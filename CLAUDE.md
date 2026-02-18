# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test Commands

```bash
cargo build              # Build the library
cargo test               # Run all tests (requires datasets/ with HDF5 files)
cargo test cmp1          # Run a single test by name
cargo test -- --nocapture  # Run tests with println output visible
```

Tests require two HDF5 files in `datasets/`:
- `frozen_pbmc_donor_c_molecule_info.h5` (~1.4 GB, scRNA-seq molecule info)
- `gene_bc_matrix.h5` (~29 MB, gene expression matrix)

The `cmp1` test also requires the `hdf5` dev-dependency, which expects `../hdf5-rust/hdf5` to exist (a local checkout of hdf5-rust bindings).

## Architecture

**h5rs** is a pure Rust, read-only HDF5 file parser targeting HTTP/object-storage access patterns. It uses `binrw` for declarative binary parsing instead of the C HDF5 library.

The official HDF5 file format spec is available at: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html

### Module layout

- **`src/lib.rs`** — High-level API: `File`, `Group`, `Object`, `Dataset`, `ChunkedDataset`. This is the user-facing layer that composes format primitives into an ergonomic read path.
- **`src/format/metadata.rs`** — HDF5 superblock, v1 B-trees (group and chunk), symbol tables, local heap, type descriptors (FixedPoint, FloatingPoint, String, VariableLength).
- **`src/format/object.rs`** — Object headers and header messages: dataspace, datatype, data layout (compact/contiguous/chunked), filter pipeline. Custom `binrw` parsers for variable-length message lists.
- **`src/format/btree.rs`** — Generic B-tree traversal with two iterator variants: `BTreeIter` (borrows reader) and `BTreeIter2` (uses `Rc<RefCell<R>>`). Abstracted via `BTree` and `HasPointer` traits.

### Read path flow

```
File::open → SuperblockV0 → root DataObjectHeader → Group (btree + local heap)
  → Group::find_obj → symbol table traversal → Object → Dataset
    → ChunkedDataset::iter_chunks → ChunkBTreeV1 → read_chunk_simple / read_chunk_filter
```

### Key design decisions

- All parsing uses `binrw` with little-endian reads (`read_le`). Structs derive `BinRead` with `#[br(...)]` attributes for magic bytes, alignment, and conditional fields.
- `modular-bitfield` handles HDF5's packed bitfield type descriptors.
- `bytemuck::cast_slice_mut` for zero-copy reinterpretation of byte buffers as typed arrays.
- Chunk decompression uses `flate2` with the `zlib-rs` feature (pure Rust zlib). Currently hardcoded for gzip+shuffle filter pipeline.
- Only HDF5 v1 B-trees and Superblock v0 are implemented. No v2 B-tree or superblock v2 support yet.
- Reader is passed around as `ObjectStoreFile` which is thin wrapper around the ObjectStore trait (object_store crate).
