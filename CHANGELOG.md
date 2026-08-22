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
- Read paths for the contiguous and compact storage layouts, alongside the
  existing chunked one. A contiguous range read fetches a single byte span
  covering the selection rather than the whole dataset.
- Support for scalar and null dataspaces.
- `Dataset::layout_name`, and a `layout` field on `DatasetInfo`.
- Round-trip minimisation for object storage. Metadata is read in aligned
  512 KiB blocks and cached, so following a pointer to a nearby structure
  usually costs no request; and wherever a set of addresses is known at once —
  a B-tree level, a group's object headers, the chunks a selection overlaps —
  they are fetched together, merged where close and issued in parallel where
  not. Listing every dataset in a 1.4 GB file now costs one round trip, and
  reading a 133 MB dataset over 2000 chunks costs six rather than 2125.
- `ObjectStoreFile::stats` reports requests, round trips and bytes read;
  `ReadOptions` tunes read-ahead, cache size, batch size and coalescing.
- A dataset remembers its chunk index, so repeated range reads on one dataset
  do not re-walk it.
- Chunk reads are now a pipeline: requests stay in flight while each one that
  lands is decompressed and copied into the output, so downloading and decoding
  overlap instead of running in sequence. `object_store`'s `get_ranges` waits
  for every range before returning, so bulk reads issue and consume their own
  requests.
- `compute::ComputePool` lets the host say where CPU-bound work runs.
  `InlineCompute` (the default) runs it on the async task, which is right for
  WASM; `ThreadPoolCompute` spreads it over OS threads without needing an async
  runtime; anything else is a two-method trait.
- `compute::YieldingCompute` hands control back to a single-threaded host
  between chunks, roughly every N decoded bytes — decompression has no await
  points inside it, so without this a large read holds the thread until every
  chunk is done. `compute::host_yield` posts through a `MessageChannel` on the
  web, ending the current task so the browser can paint, and
  `YieldingCompute::with_yield` takes a yield of the host's own. On `wasm32`
  neither the pool nor its futures need to be `Send`, so one built from JS
  promises or workers fits the trait.
- `ReadOptions` gains `io_concurrency`, `max_coalesced_bytes` and
  `max_inflight_bytes`, which together decide how well a high-latency link is
  saturated and how much memory a read holds. `max_coalesced_bytes` is a
  ceiling on merging neighbouring ranges, not on request size: a chunk is
  always fetched whole, however large.
- A `dump` example that lists a file's datasets and reads one.

### Changed

- **Breaking:** `ChunkedDataset` is now `Dataset` and `open_chunked_dataset` is
  now `open_dataset`, since reads are no longer limited to chunked datasets.
  `chunk_shape` and `chunk_index_name` return `Option`, being `None` for a
  dataset that is not chunked.
- Speculative reads are trimmed to the object's size, learned from the first
  response, rather than being rejected by the store for starting past the end
  of the file.
- Object headers are read in two phases — prefix first, then the exact extent —
  so a header larger than the default metadata fetch (a big compact dataset, or
  a long run of attributes) no longer fails to parse.
- Round-trip and fuzz tests now run against both on-disk formats and all three
  storage layouts.
- Unfiltered chunks are copied straight out of the fetched bytes rather than
  going through the compute pool.
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
