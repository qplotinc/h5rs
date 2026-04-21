
h5rs is a pure Rust, read-only, implementation of the HDF5 file format.  It can read HDF5 data from any `ObjectStore` implementation, from the [object_store](https://crates.io/crates/object_store) crate, which support most object stores, HTTP, and POSIX filesystems. 

HDF5 is an effective format for storing many kinds of scientific data. Powerful browser-based visualization tools makes it feasible to load and explore large datasets. But have you ever tried to read HDF5 in browser over HTTP?

The HDF5 library required to read and write the format has the following drawbacks:
- Very large legacy C codebase: can be difficult and slow to build
- Relies on POSIX I/O
- Synchronous, single threaded API - can't easily take advantage of multiple threads when decompressing chunks

This makes it challenging to consume HDF5 from the browser. [h5wasm](https://github.com/usnistgov/h5wasm) compiles the HDF5 library to WASM, and mounts URLs to an Emscripten Filesystem with a [special library](https://github.com/bmaranville/lazyFileLRU). This can only be used inside Web workers.

h5rs is suitable for visualization tools that access HDF5 files via object storage. Rust, and the [binrw](https://crates.io/crates/binrw) and [object_store](https://crates.io/crates/object_store) crates make it feasible to have a small and efficient HDF5 reader.

# Implemented
- Support widely used HDF5 features
- Reasonable performance profile, good multithreading support when decompressing chunked data.
- Async API
- Differential, randomized testing against the gold standard (hdf5)[https://crates.io/crates/hdf5] wrapper crate.

# Open to contributions, but not on the roadmap
- Sync API
- Compound type support
- Support for MPI and parallel-IO.
- Exotic numeric types

# Non Goals
- Write support
