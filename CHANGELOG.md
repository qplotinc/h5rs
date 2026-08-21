# Changelog

All notable changes to this project are documented in this file. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
