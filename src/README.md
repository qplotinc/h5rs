
HDF5 is an effective format for storing many kinds of scientific data. Powerful browser-based visualization tools makes it feasible to load and explore large datasets. But have you ever tried to read HDF5 in browser over HTTP?

The HDF5 library required to read and write the format has the following drawbacks:
- Very large legacy C codebase: can be difficult and slow to build
- Relies on POSIX I/O
- Synchronous, single threaded API - can't easily take advantage of multiple threads when decompressing chunks

This makes it challenging to consume HDF5 from the browser. [h5wasm](https://github.com/usnistgov/h5wasm) compiles the HDF5 library to WASM, and mounts URLs to an Emscripten Filesystem with a [special library](https://github.com/bmaranville/lazyFileLRU). This can only be used inside Web workers.

h5rs is attempt to have a pure Rust HDF5 reader, suitable for visualization tools that access HDF5 files via object storage. Rust, and the incredible [binrw](https://crates.io/crates/binrw) crate now seem to make it feasible to have a small-ish library that lets us efficiently read HDF5 in the library.

# Goals
- Support widely used HDF5 features
- Reasonable performance profile, good multithreading support when decompressing chunked data.
- Compatible with HTTP & object storage IO. (e.g. via object_store crate)
- Async API (TBD how to support a sync API)

# Open to contributions, but not on my roadmap
- Compound type support
- Any of the MPI / HPC stuff
- Exotic numeric types

# Non Goals
- Write support
