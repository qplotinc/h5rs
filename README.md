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

let ds = h5rs::open_dataset(&file, &["matrix", "data"])
    .await?
    .expect("dataset not found");

// Read everything...
let all = ds.read_full::<u32>(&file).await?;

// ...or just the slice you need. Only the bytes covering it are fetched:
// the overlapping chunks of a chunked dataset, or the enclosing byte span
// of a contiguous one.
let slice = ds.read_range::<u32>(&[1_000..2_000], &file).await?;
```

Swap `LocalFileSystem` for `AmazonS3`, `GoogleCloudStorage`, `MicrosoftAzure`, or `HttpStore` and the same code reads over the network, issuing one ranged GET per chunk.

For a contiguous dataset a range read is a single ranged GET covering the selection — exact for a one-dimensional range, and spanning the touched rows for higher-rank selections.

## Round trips

Over object storage a request costs far more than the bytes it carries, so h5rs is built to minimise requests rather than bytes. Measured against a 1.4 GB single-cell file:

| workload | round trips | bytes read |
|---|---|---|
| list all datasets | 1 | 0.5 MiB |
| read a 100k-element slice | 4 | 1.7 MiB |
| read a 133 MB dataset (2000+ chunks) | 6 | 134 MiB |
| list a group of 60 datasets spread across the file | 5 | 2.3 MiB |

Two things get it there. Metadata is read in aligned blocks (512 KiB by default), so following a pointer to a nearby structure usually costs no further request. And wherever a set of addresses is known at once — a B-tree level, a group's object headers, the chunks a selection overlaps — they are fetched together: merged into one request where they lie close, issued in parallel where they do not.

### Overlapping I/O and decompression

Reading compressed chunks is two kinds of work at once, and h5rs pipelines them: requests stay in flight while each one that lands is decompressed and copied into the output. Whichever resource is scarcer sets the pace and the other disappears behind it.

Where the decompression runs is your choice, because it depends on the host. The default runs it on the async task — right for a browser, and still overlapped with I/O, but one core. `ThreadPoolCompute` spreads it over OS threads and needs no async runtime; anything else (Rayon, a Tokio blocking pool, web workers) is a two-method `ComputePool` trait.

```rust
use h5rs::compute::ThreadPoolCompute;
use std::sync::Arc;

let file = ObjectStoreFile::new(Box::new(store), Path::from("matrix.h5"))
    .with_compute(Arc::new(ThreadPoolCompute::with_available_parallelism()));
```

Measured against a simulated link (`cargo test --features hdf5-compare --test pipeline -- --ignored --nocapture`):

| scenario | limit | wall | ideal | limiting resource used |
|---|---|---|---|---|
| 200 MiB / 50 chunks over 30 MB/s, 50 ms latency, 1 core | link | 3.58 s | 3.51 s | 98% |
| 512 MiB / 100 chunks from 2 GB/s flash, 1 core | CPU | 278 ms | 237 ms | 85% |
| 512 MiB / 100 chunks from 2 GB/s flash, 8 cores | link | 145 ms | 135 ms | 93% |

In the first, 235 ms of decompression hides entirely inside 3.5 s of download. In the last, 237 ms of decompression across 8 cores hides inside 135 ms of reading. The single-core row is limited by decompression and the copy into the output array sharing one thread.

`max_inflight_bytes` caps what a read holds — bytes still in flight plus bytes fetched but not yet decoded — so a fast link feeding a slow decoder cannot buffer the whole dataset.

`ObjectStoreFile::stats()` reports what a read cost, and `ReadOptions` tunes the trade-off:

```rust
use h5rs::object_store::{ObjectStoreFile, ReadOptions};

let file = ObjectStoreFile::with_options(
    Box::new(store),
    Path::from("matrix.h5"),
    ReadOptions { metadata_block_size: 1 << 20, ..Default::default() },
);
// ... read ...
println!("{:?}", file.stats());
```

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
| Data layout | chunked, contiguous, compact (message v1-v5) | virtual |
| Filters | deflate (gzip), shuffle, per-chunk filter masks | szip, blosc, lzf, n-bit, scale-offset, fletcher32 |
| Dataspaces | simple, scalar, null (message v1-v2) | permutation indices |
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

# Round trips and bytes across read-ahead settings and file shapes
cargo test --features hdf5-compare io_tuning -- --ignored --nocapture
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
