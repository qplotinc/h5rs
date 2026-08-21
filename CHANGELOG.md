# Changelog

All notable changes to this project are documented in this file. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Read files written in HDF5's "latest" on-disk format (`H5Pset_libver_bounds`),
  as emitted by HDF5 1.10 and later:
  - superblock versions 2 and 3;
  - version 2 object headers (`OHDR`) and continuation blocks (`OCHK`);
  - "new style" groups, with links stored compactly as object header messages
    or densely in a fractal heap indexed by a version 2 B-tree;
  - all five version 4/5 chunk index structures — single chunk, implicit, fixed
    array, extensible array and version 2 B-tree;
  - densely stored attributes, and attribute messages versions 2 and 3;
  - dataspace message version 2, filter pipeline message version 2, and fill
    value message version 3.
- Honour the per-chunk filter mask, so a chunk stored with a filter skipped
  decodes correctly instead of failing or returning garbage.
- `ChunkedDataset::chunk_index_name` reports which index structure a dataset
  uses.
- A `dump` example that lists a file's datasets and reads one.

### Changed

- Round-trip and fuzz tests now run against both on-disk formats.
- Unsupported filter pipelines are rejected before any chunk is fetched, rather
  than being fed to the zlib decoder.

## [0.1.0]

Initial release.

- Read-only HDF5 parsing in pure Rust, with no dependency on the HDF5 C library.
- Async reads against any `object_store` backend: local filesystem, S3, GCS,
  Azure, HTTP.
- `list_datasets` to walk a file's group tree, `open_chunked_dataset` to open a
  dataset by path.
- `ChunkedDataset::read_full` and `read_range` for whole-dataset and
  rectangular sub-region reads; a range read fetches and decompresses only the
  chunks that overlap the selection.
- Superblock v0, v1 B-trees (group and chunk), v1 object headers, contiguous
  and chunked layouts, and the deflate + shuffle filter pipeline.
- All failures are returned as `H5Error`; malformed or unsupported input never
  panics. Unimplemented format features report `H5Error::Unsupported` rather
  than returning wrong data.
- `wasm32-unknown-unknown` support.
- Dual licensed under MIT OR Apache-2.0.
