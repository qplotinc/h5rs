# h5rs

[![CI](https://github.com/pmarks/h5rs/actions/workflows/ci.yml/badge.svg)](https://github.com/pmarks/h5rs/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/h5rs.svg)](https://crates.io/crates/h5rs)
[![docs.rs](https://img.shields.io/docsrs/h5rs)](https://docs.rs/h5rs)
[![license](https://img.shields.io/crates/l/h5rs.svg)](#license)

h5rs is a pure Rust, read-only implementation of the HDF5 file format. It can read HDF5 data from any `ObjectStore` implementation, from the [object_store](https://crates.io/crates/object_store) crate, which supports most object stores, HTTP, and POSIX filesystems.

HDF5 is an effective format for storing many kinds of scientific data. Powerful browser-based visualization tools makes it feasible to load and explore large datasets. But have you ever tried to read HDF5 in browser over HTTP?

The HDF5 library required to read and write the format has the following drawbacks:
- Very large legacy C codebase: can be difficult and slow to build
- Relies on POSIX I/O
- Synchronous, single threaded API - can't easily take advantage of multiple threads when decompressing chunks

This makes it challenging to consume HDF5 from the browser. [h5wasm](https://github.com/usnistgov/h5wasm) compiles the HDF5 library to WASM, and mounts URLs to an Emscripten Filesystem with a [special library](https://github.com/bmaranville/lazyFileLRU). This can only be used inside Web workers.

h5rs is suitable for visualization tools that access HDF5 files via object storage. Rust, and the [binrw](https://crates.io/crates/binrw) and [object_store](https://crates.io/crates/object_store) crates make it feasible to have a small and efficient HDF5 reader.

## Install

```bash
cargo add h5rs object_store
```

## Usage

```rust
use h5rs::object_store::ObjectStoreFile;
use object_store::{local::LocalFileSystem, path::Path};

let store = LocalFileSystem::new_with_prefix("/data")?;
let file = ObjectStoreFile::new(Box::new(store), Path::from("matrix.h5"));

// What's in the file?
for info in h5rs::list_datasets(&file).await? {
    println!("{} shape={:?} filters={:?}", info.path, info.shape, info.filters);
}

let ds = h5rs::open_chunked_dataset(&file, &["matrix", "data"])
    .await?
    .expect("dataset not found");

// Read everything...
let all = ds.read_full::<u32>(&file).await?;

// ...or just the slice you need. Only the overlapping chunks are fetched
// and decompressed.
let slice = ds.read_range::<u32>(&[1_000..2_000], &file).await?;
```

Swap `LocalFileSystem` for `AmazonS3`, `GoogleCloudStorage`, `MicrosoftAzure`, or `HttpStore` and the same code reads over the network, issuing one ranged GET per chunk.

## Implemented
- Support widely used HDF5 features, in both the pre-1.10 and the 1.10+ on-disk formats
- Reasonable performance profile, good multithreading support when decompressing chunked data.
- Async API
- Differential, randomized testing against the gold standard [hdf5](https://crates.io/crates/hdf5) wrapper crate.

## Format coverage

h5rs reads both of the on-disk formats the HDF5 library emits: the "earliest" encoding that `h5py` and most writers produce by default, and the "latest" encoding introduced with HDF5 1.10 and selected by `H5Pset_libver_bounds`.

| Area | Supported | Not yet |
|---|---|---|
| Superblock | v0, v1, v2, v3 | non-zero base address (user block) |
| Object header | v1, v2 (`OHDR`), continuation blocks | shared messages |
| Group links | symbol table, compact link messages, fractal heap + v2 B-tree | soft, external and user-defined links |
| Chunk index | v1 B-tree, single chunk, implicit, fixed array, extensible array, v2 B-tree | |
| Data layout | chunked (message v1-v5) | contiguous, compact and virtual are listed but not read |
| Filters | deflate (gzip), shuffle, per-chunk filter masks | szip, blosc, lzf, n-bit, scale-offset, fletcher32 |
| Datatypes | fixed-point, floating-point, string, variable-length | compound, enum, array, reference |
| Attributes | message v1-v3, compact and dense | shared datatypes and dataspaces |

Addresses and lengths must be 8 bytes wide, which is the HDF5 default; anything else is reported as unsupported rather than mis-parsed. Metadata checksums are parsed past but not verified.

Every round-trip test runs against both formats, and the fuzzer randomises which one it writes, asserting on the way out that it reached all of the chunk index structures.

## Open to contributions, but not on the roadmap
- Sync API
- Compound type support
- Support for MPI and parallel-IO.
- Exotic numeric types

## Non Goals
- Write support

## Development

```bash
cargo test          # unit + range-read tests; no external dependencies
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings

# Inspect a file
cargo run --example dump -- path/to/file.h5
cargo run --example dump -- path/to/file.h5 /group/dataset
```

### Differential tests against the HDF5 C library

The `hdf5-compare` feature enables round-trip and fuzz tests that write files
with the HDF5 C library and read them back with h5rs. It pulls in
[`hdf5-metno`](https://crates.io/crates/hdf5-metno) built against a bundled
copy of libhdf5, so no system package is needed — but the first build spends a
few minutes compiling C.

```bash
cargo test --features hdf5-compare
```

### Test data

A few tests read large sample files that are not distributed with the
repository. They skip automatically when the files are absent; to run them,
place these under `datasets/`:

- `frozen_pbmc_donor_c_molecule_info.h5` (~1.4 GB, scRNA-seq molecule info)
- `gene_bc_matrix.h5` (~29 MB, gene expression matrix)

With both files and the `hdf5-compare` feature enabled, `cargo test` also
compares every dataset and attribute in them byte-for-byte against the C
library.

### Benchmarks

```bash
cargo test --features hdf5-compare perf -- --ignored --nocapture
```

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <https://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <https://opensource.org/licenses/MIT>)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
