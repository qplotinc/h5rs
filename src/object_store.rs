use std::io::Cursor;

use binrw::BinRead;
use bytes::Bytes;
use futures::stream::BoxStream;
use object_store::{GetOptions, GetRange, ObjectMeta};
use object_store::{ObjectStore, path::Path};

use crate::error::H5Result;

/// Default fetch size for metadata structures of unknown size
/// (object headers, B-tree nodes, symbol tables). 8KB covers all
/// realistic HDF5 metadata blocks.
const METADATA_FETCH_SIZE: u64 = 8192;

#[derive(Clone)]
pub struct ObjectStoreFile {
    pub path: Path,
    pub os: std::sync::Arc<Box<dyn ObjectStore>>,
}

impl ObjectStoreFile {
    pub fn new(os: Box<dyn ObjectStore>, path: Path) -> ObjectStoreFile {
        ObjectStoreFile {
            path,
            os: std::sync::Arc::new(os),
        }
    }

    pub fn store(&self) -> &dyn ObjectStore {
        self.os.as_ref()
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn filename(&self) -> String {
        self.path.filename().unwrap_or("default").to_string()
    }

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

    pub async fn size(&self) -> Result<u64, object_store::Error> {
        Ok(self.metadata().await?.size)
    }

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
pub async fn read_and_parse<T: for<'a> BinRead<Args<'a> = ()>>(
    file: &ObjectStoreFile,
    offset: u64,
    len: u64,
) -> H5Result<T> {
    let bytes = file.get_range(offset..offset + len).await?;
    let mut cursor = Cursor::new(bytes);
    Ok(T::read_le(&mut cursor)?)
}

/// Fetch `len` bytes at `offset` and parse with binrw args.
pub async fn read_and_parse_args<T: for<'a> BinRead<Args<'a> = A>, A: Clone>(
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
pub async fn read_metadata<T: for<'a> BinRead<Args<'a> = ()>>(
    file: &ObjectStoreFile,
    offset: u64,
) -> H5Result<T> {
    read_and_parse(file, offset, METADATA_FETCH_SIZE).await
}

/// Fetch a metadata structure of unknown size at `offset` with args.
pub async fn read_metadata_args<T: for<'a> BinRead<Args<'a> = A>, A: Clone>(
    file: &ObjectStoreFile,
    offset: u64,
    args: A,
) -> H5Result<T> {
    read_and_parse_args(file, offset, METADATA_FETCH_SIZE, args).await
}

/// Fetch raw bytes at `offset` of length `len`.
pub async fn fetch_exact(file: &ObjectStoreFile, offset: u64, len: u64) -> H5Result<Bytes> {
    Ok(file.get_range(offset..offset + len).await?)
}
