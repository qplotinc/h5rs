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
cargo check --target wasm32-unknown-unknown --lib --tests   # examples are native-only
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
- **`src/format/btree.rs`** — Generic async version 1 B-tree traversal (`collect_btree_leaves`, `collect_btree_leaves_args`), abstracted over node types via the `BTree` and `HasPointer` traits.
- **`src/format/btree2.rs`** — Version 2 B-trees. Only enumeration is implemented (never search by key), which is all the read path needs. The node geometry — the widths of the per-child record counts — has to be recomputed exactly as the library does, since it is not stored.
- **`src/format/fractal_heap.rs`** — Fractal heaps and the doubling table, used to dereference the heap IDs that dense links and attributes are indexed by. Heap offsets are measured from the *start of the direct block*, including its prefix.
- **`src/format/chunk_index.rs`** — Enumerates a chunked dataset's chunks through any of the six index structures into a common `ChunkRecord`. Element widths are taken from the sizes recorded in each index header rather than recomputed.
- **`src/format/dense.rs`** — Links and attributes stored densely (fractal heap + v2 B-tree).
- **`src/dataset.rs`** — `Dataset`: the public reader. Dispatches `read_full`/`read_range` across the chunked, contiguous and compact layouts, decodes the filter pipeline, and assembles sub-regions with `copy_region_inner`.
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
- Both on-disk formats are supported: superblock v0-v3, object header v1 and v2, and all six chunk index structures. The supported/unsupported matrix lives in the crate docs and README — update both when that changes.
- Object header messages are dispatched explicitly by type in `parse_inner_message`, never through a `binrw` enum with a catch-all arm: a fall-through would turn a parse failure into a silently missing message.
- Page-initialisation bitmaps in the array indexes are most-significant-bit first within each byte, matching `H5VM_bit_get`.
- Reader is passed around as `ObjectStoreFile` which is thin wrapper around the ObjectStore trait (object_store crate).
- I/O is tuned for round trips, not bytes: `src/object_store.rs` caches aligned metadata blocks for pointer chasing, and `read_metadata_many`/`read_ranges` batch every case where a set of addresses is already known (B-tree levels, a group's object headers, a selection's chunks). Batched reads deliberately do *not* read ahead — the addresses are known, so rounding each up to a block would only cost bytes. `cargo test --features hdf5-compare io_tuning -- --ignored --nocapture` prints round trips and bytes across file shapes and settings; use it before changing any of these defaults.
- `read_metadata` fetches a fixed 8KB parse window, which is fine for B-tree nodes and heap headers but not for object headers — those are read in two phases via `header_chunk_extent`, since a compact dataset or a run of attributes can make one arbitrarily large.
- Chunk reads are a pipeline (`Dataset::stream_chunks_into`): fetches stay in flight while each landed request is decoded on the `ComputePool` and copied into the output. `object_store::get_ranges` is all-or-nothing (`try_collect`), so it is used only for metadata batches where every range is wanted before parsing can continue. `cargo test --features hdf5-compare --test pipeline -- --ignored --nocapture` measures link and CPU utilisation against a simulated link; the target is that the limiting resource is >90% used and the other is fully hidden.
- Where decompression runs is the host's choice (`src/compute.rs`), never the library's: a browser has one thread, a native program may have Rayon, Tokio or nothing. `ComputePool` futures are `Send` on native and `?Send` on wasm32 (`MaybeSendSync`), because a browser's scheduling primitives are not `Send` and there is no second thread to send them to. `InlineCompute` runs its job on first poll rather than at submission, which is what lets `YieldingCompute` interpose between chunks.
- `max_request_bytes` only decides when *neighbouring* chunks are merged into one request; a chunk is always fetched whole, however large. `tests/pipeline.rs::a_chunk_is_always_one_request` pins that — splitting a chunk would mean it could not be decoded until every piece arrived.
- A range read never fetches more than it needs: whole chunks for a chunked dataset, and for a contiguous one the single byte span from the first selected element to the last.
