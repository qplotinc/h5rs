//! Read-only [`ObjectStore`] backed by Node.js `fs` APIs.
//!
//! Only available on `wasm32-unknown-unknown` running under Node.js.

use std::fmt;

use async_trait::async_trait;
use bytes::Bytes;
use chrono::Utc;
use futures::StreamExt;
use futures::stream::{self, BoxStream};
use object_store::path::Path;
use object_store::*;
use wasm_bindgen::prelude::*;

// ---------------------------------------------------------------------------
// JS bindings (synchronous Node.js fs calls)
// ---------------------------------------------------------------------------

#[wasm_bindgen(module = "/src/node_fs.js")]
extern "C" {
    #[wasm_bindgen(js_name = "readFileRangeSync")]
    fn js_read_range(path: &str, offset: f64, length: f64) -> js_sys::Uint8Array;

    #[wasm_bindgen(js_name = "fileSizeSync")]
    fn js_file_size(path: &str) -> f64;

    #[wasm_bindgen(js_name = "getCwd")]
    fn js_get_cwd() -> String;
}

/// Synchronous helper: read `length` bytes at `offset` from a local file.
/// Converts the JS Uint8Array to an owned `Vec<u8>` so no `JsValue` escapes.
fn read_bytes(path: &str, offset: u64, length: u64) -> Vec<u8> {
    js_read_range(path, offset as f64, length as f64).to_vec()
}

/// Synchronous helper: get the file size in bytes.
fn file_len(path: &str) -> u64 {
    js_file_size(path) as u64
}

/// Returns `process.cwd()`.
pub fn cwd() -> String {
    js_get_cwd()
}

// ---------------------------------------------------------------------------
// NodeFileSystem
// ---------------------------------------------------------------------------

/// A read-only [`ObjectStore`] that reads from the local filesystem via
/// Node.js synchronous `fs` calls. Analogous to `LocalFileSystem` but works
/// inside `wasm32-unknown-unknown` running under Node.js.
#[derive(Debug, Clone)]
pub struct NodeFileSystem {
    prefix: String,
}

impl NodeFileSystem {
    pub fn new_with_prefix(prefix: &str) -> Self {
        Self {
            prefix: prefix.to_string(),
        }
    }

    /// Create a `NodeFileSystem` rooted at `process.cwd()`.
    pub fn cwd() -> Self {
        Self::new_with_prefix(&cwd())
    }

    fn full_path(&self, location: &Path) -> String {
        let loc = location.as_ref();
        if loc.is_empty() {
            self.prefix.clone()
        } else {
            format!("{}/{}", self.prefix, loc)
        }
    }
}

impl fmt::Display for NodeFileSystem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "NodeFileSystem({})", self.prefix)
    }
}

fn not_impl(op: &str) -> Error {
    Error::NotImplemented {
        operation: op.to_string(),
        implementer: "NodeFileSystem".to_string(),
    }
}

#[async_trait]
impl ObjectStore for NodeFileSystem {
    async fn put_opts(
        &self,
        _location: &Path,
        _payload: PutPayload,
        _opts: PutOptions,
    ) -> Result<PutResult> {
        Err(not_impl("put_opts"))
    }

    async fn put_multipart_opts(
        &self,
        _location: &Path,
        _opts: PutMultipartOptions,
    ) -> Result<Box<dyn MultipartUpload>> {
        Err(not_impl("put_multipart_opts"))
    }

    async fn get_opts(&self, location: &Path, options: GetOptions) -> Result<GetResult> {
        let fs_path = self.full_path(location);
        let total_size = file_len(&fs_path);

        let meta = ObjectMeta {
            location: location.clone(),
            last_modified: Utc::now(),
            size: total_size,
            e_tag: None,
            version: None,
        };

        options.check_preconditions(&meta)?;

        let range = match options.range {
            Some(range) => range
                .as_range(total_size)
                .map_err(|source| Error::Generic {
                    store: "NodeFileSystem",
                    source: source.into(),
                })?,
            None => 0..total_size,
        };

        let data = read_bytes(&fs_path, range.start, range.end - range.start);
        let bytes = Bytes::from(data);
        let payload =
            GetResultPayload::Stream(stream::once(futures::future::ready(Ok(bytes))).boxed());

        Ok(GetResult {
            payload,
            meta,
            range,
            attributes: Attributes::new(),
            extensions: Default::default(),
        })
    }

    fn delete_stream(
        &self,
        _locations: BoxStream<'static, Result<Path>>,
    ) -> BoxStream<'static, Result<Path>> {
        stream::once(futures::future::ready(Err(not_impl("delete_stream")))).boxed()
    }

    fn list(&self, _prefix: Option<&Path>) -> BoxStream<'static, Result<ObjectMeta>> {
        stream::empty().boxed()
    }

    async fn list_with_delimiter(&self, _prefix: Option<&Path>) -> Result<ListResult> {
        Err(not_impl("list_with_delimiter"))
    }

    async fn copy_opts(&self, _from: &Path, _to: &Path, _options: CopyOptions) -> Result<()> {
        Err(not_impl("copy_opts"))
    }
}
