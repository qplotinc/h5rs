//! Byte-range access to an HDF5 file living in an [`object_store`] backend.
//!
//! [`ObjectStoreFile`] pairs an [`ObjectStore`] implementation with the path of
//! a single object, and exposes the small set of ranged-read operations the
//! rest of the crate needs. Everything h5rs does — parsing the superblock,
//! walking B-trees, fetching chunks — is expressed as ranged GETs against this
//! type, which is what makes the reader usable over HTTP and object storage.

use std::io::Cursor;

use binrw::BinRead;
use bytes::Bytes;
use futures_core::stream::BoxStream;
use object_store::{GetOptions, GetRange, ObjectMeta};
use object_store::{ObjectStore, path::Path};

use crate::error::H5Result;

/// Default fetch size for metadata structures of unknown size
/// (object headers, B-tree nodes, symbol tables). 8KB covers all
/// realistic HDF5 metadata blocks.
const METADATA_FETCH_SIZE: u64 = 8192;

/// A single HDF5 file addressed within an [`ObjectStore`].
///
/// Cloning is cheap: the underlying store is shared behind an [`Arc`].
///
/// ```no_run
/// use h5rs::object_store::ObjectStoreFile;
/// use object_store::{local::LocalFileSystem, path::Path};
///
/// let store = LocalFileSystem::new_with_prefix("/data").unwrap();
/// let file = ObjectStoreFile::new(Box::new(store), Path::from("example.h5"));
/// ```
///
/// [`Arc`]: std::sync::Arc
#[derive(Clone)]
pub struct ObjectStoreFile {
    path: Path,
    os: std::sync::Arc<Box<dyn ObjectStore>>,
}

impl ObjectStoreFile {
    /// Address the object at `path` within `os`.
    pub fn new(os: Box<dyn ObjectStore>, path: Path) -> ObjectStoreFile {
        ObjectStoreFile {
            path,
            os: std::sync::Arc::new(os),
        }
    }

    /// The underlying object store.
    pub fn store(&self) -> &dyn ObjectStore {
        self.os.as_ref()
    }

    /// The object's path within the store.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The last path segment, or `"default"` if the path has none.
    pub fn filename(&self) -> String {
        self.path.filename().unwrap_or("default").to_string()
    }

    /// Fetch the object's metadata (size, etag, last-modified).
    pub async fn metadata(&self) -> Result<ObjectMeta, object_store::Error> {
        // We make a GET request with a minimal range
        // because S3 pre-signed URLs can't be used with HEAD.
        let opts = GetOptions {
            range: Some(GetRange::Bounded(0..1)),
            ..Default::default()
        };
        let r = self.os.get_opts(&self.path, opts).await?;
        Ok(r.meta)
    }

    /// The object's size in bytes.
    pub async fn size(&self) -> Result<u64, object_store::Error> {
        Ok(self.metadata().await?.size)
    }

    /// Fetch a byte range as a stream, without buffering the whole range.
    pub async fn get_range_stream(
        &self,
        range: std::ops::Range<u64>,
    ) -> Result<BoxStream<'static, Result<Bytes, object_store::Error>>, object_store::Error> {
        let opts = GetOptions {
            range: Some(GetRange::Bounded(range)),
            ..Default::default()
        };
        let r = self.os.get_opts(&self.path, opts).await?;
        Ok(r.into_stream())
    }

    /// Fetch a byte range into memory.
    pub async fn get_range(
        &self,
        range: std::ops::Range<u64>,
    ) -> Result<Bytes, object_store::Error> {
        let opts = GetOptions {
            range: Some(GetRange::Bounded(range)),
            ..Default::default()
        };
        let r = self.os.get_opts(&self.path, opts).await?;
        r.bytes().await
    }
}

/// Fetch `len` bytes at `offset` and parse with binrw (no args).
pub(crate) async fn read_and_parse<T: for<'a> BinRead<Args<'a> = ()>>(
    file: &ObjectStoreFile,
    offset: u64,
    len: u64,
) -> H5Result<T> {
    let bytes = file.get_range(offset..offset + len).await?;
    let mut cursor = Cursor::new(bytes);
    Ok(T::read_le(&mut cursor)?)
}

/// Fetch `len` bytes at `offset` and parse with binrw args.
pub(crate) async fn read_and_parse_args<T: for<'a> BinRead<Args<'a> = A>, A: Clone>(
    file: &ObjectStoreFile,
    offset: u64,
    len: u64,
    args: A,
) -> H5Result<T> {
    let bytes = file.get_range(offset..offset + len).await?;
    let mut cursor = Cursor::new(bytes);
    Ok(T::read_le_args(&mut cursor, args)?)
}

/// Fetch a metadata structure of unknown size at `offset`.
pub(crate) async fn read_metadata<T: for<'a> BinRead<Args<'a> = ()>>(
    file: &ObjectStoreFile,
    offset: u64,
) -> H5Result<T> {
    read_and_parse(file, offset, METADATA_FETCH_SIZE).await
}

/// Fetch a metadata structure of unknown size at `offset` with args.
pub(crate) async fn read_metadata_args<T: for<'a> BinRead<Args<'a> = A>, A: Clone>(
    file: &ObjectStoreFile,
    offset: u64,
    args: A,
) -> H5Result<T> {
    read_and_parse_args(file, offset, METADATA_FETCH_SIZE, args).await
}

/// Fetch the default metadata block at `offset` without parsing it, for
/// structures whose true length is only known once their prefix is read.
pub(crate) async fn fetch_metadata_block(file: &ObjectStoreFile, offset: u64) -> H5Result<Bytes> {
    fetch_exact(file, offset, METADATA_FETCH_SIZE).await
}

/// Fetch raw bytes at `offset` of length `len`.
pub(crate) async fn fetch_exact(file: &ObjectStoreFile, offset: u64, len: u64) -> H5Result<Bytes> {
    Ok(file.get_range(offset..offset + len).await?)
}
