# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test Commands

```bash
cargo build                            # Build the library
cargo test                             # Unit + range-read tests, no external deps
cargo test --features hdf5-compare     # Adds differential tests vs. the HDF5 C library
cargo test -- --nocapture              # Show println output
cargo clippy --all-targets -- -D warnings
cargo fmt --all -- --check
cargo build --target wasm32-unknown-unknown
```

CI (`.github/workflows/ci.yml`) runs lint, tests on Linux/macOS/Windows, the
`hdf5-compare` differential tests, a wasm32 build, rustdoc with `-D warnings`,
an MSRV check (1.86), and `cargo publish --dry-run`. Keep all of these green.

### hdf5-compare feature

Cargo has no optional dev-dependencies, so `hdf5` (the `hdf5-metno` fork, built
with a bundled libhdf5) is declared as an *optional regular* dependency gated
behind the `hdf5-compare` feature and used only from `#[cfg(test)]` code. The
round-trip tests must pin the writer to earliest format bounds
(`libver_bounds(Earliest, V18)`) — with metno's default (latest) bounds the C
library emits v2/v3 superblocks and v2 object headers that h5rs cannot parse.

### Test data

Some tests read large sample files under `datasets/` that are not in the repo:
- `frozen_pbmc_donor_c_molecule_info.h5` (~1.4 GB, scRNA-seq molecule info)
- `gene_bc_matrix.h5` (~29 MB, gene expression matrix)

These tests skip (printing `SKIP: ...`) when the files are absent — keep that
behaviour when adding dataset-driven tests, so a fresh clone stays green.

## Architecture

**h5rs** is a pure Rust, read-only HDF5 file parser targeting HTTP/object-storage access patterns. It uses `binrw` for declarative binary parsing instead of the C HDF5 library.

The official HDF5 file format spec is available at: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html

### Module layout

- **`src/lib.rs`** — High-level read path. `File`, `Group`, `Object` and `Dataset` are crate-private; the public API is `list_datasets`, `open_chunked_dataset`, `DatasetInfo`, plus `ChunkedDataset`/`NdArray` re-exported from `chunked`. Keep the public surface small — everything public needs a doc comment (`#![warn(missing_docs)]` is on).
- **`src/format/metadata.rs`** — HDF5 superblock, v1 B-trees (group and chunk), symbol tables, local heap, type descriptors (FixedPoint, FloatingPoint, String, VariableLength).
- **`src/format/object.rs`** — Object headers and header messages: dataspace, datatype, data layout (compact/contiguous/chunked), filter pipeline. Custom `binrw` parsers for variable-length message lists.
- **`src/format/btree.rs`** — Generic async B-tree traversal (`collect_btree_leaves`, `collect_btree_leaves_args`), abstracted over node types via the `BTree` and `HasPointer` traits.
- **`src/chunked.rs`** — `ChunkedDataset`: chunk B-tree traversal, filter decoding, and sub-region assembly for `read_full`/`read_range`.
- **`src/object_store.rs`** — `ObjectStoreFile` (ranged GETs against an `ObjectStore`) plus the crate-private binrw fetch-and-parse helpers.
- **`src/node_store.rs` / `src/node_fs.js`** — Node.js filesystem `ObjectStore` used only by the wasm32 test build.

### Read path flow

```
File::open → SuperblockV0 → root DataObjectHeader → Group (btree + local heap)
  → Group::find_obj → symbol table traversal → Object → Dataset
    → ChunkedDataset::read_range → ChunkBTreeV1 → read_chunk_into (filtered or raw)
```

### Key design decisions

- All parsing uses `binrw` with little-endian reads (`read_le`). Structs derive `BinRead` with `#[br(...)]` attributes for magic bytes, alignment, and conditional fields.
- Packed bitfield type descriptors are parsed as raw byte arrays with hand-written accessors (`FixedPointDescriptor`, `FloatingPointDescriptor` in `format/object.rs`).
- `bytemuck::cast_slice_mut` for zero-copy reinterpretation of byte buffers as typed arrays.
- Chunk decompression uses `flate2` with the `zlib-rs` feature (pure Rust zlib). Currently hardcoded for gzip+shuffle filter pipeline.
- Only HDF5 v1 B-trees and Superblock v0 are implemented. No v2 B-tree or superblock v2/v3 support yet. The supported/unsupported matrix lives in the crate docs and README — update both when that changes.
- Reader is passed around as `ObjectStoreFile` which is thin wrapper around the ObjectStore trait (object_store crate).
