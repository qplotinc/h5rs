//! A pure Rust, read-only HDF5 reader built for object storage and HTTP.
//!
//! h5rs parses the [HDF5 file format] directly with [`binrw`], so it needs no C
//! library, no `libhdf5` build step, and no POSIX file handle. Every read is a
//! ranged GET against an [`object_store`] backend, which means the same code
//! reads a local file, an S3/GCS/Azure object, a plain HTTP URL, or — compiled
//! to `wasm32` — a file fetched from the browser.
//!
//! # Getting started
//!
//! Point an [`ObjectStoreFile`] at your file, then list what is inside it or
//! open a dataset by path:
//!
//! ```no_run
//! use h5rs::object_store::ObjectStoreFile;
//! use object_store::{local::LocalFileSystem, path::Path};
//!
//! # async fn run() -> h5rs::error::H5Result<()> {
//! let store = LocalFileSystem::new_with_prefix("/data")?;
//! let file = ObjectStoreFile::new(Box::new(store), Path::from("matrix.h5"));
//!
//! for info in h5rs::list_datasets(&file).await? {
//!     println!("{} shape={:?} filters={:?}", info.path, info.shape, info.filters);
//! }
//!
//! let ds = h5rs::open_dataset(&file, &["matrix", "data"])
//!     .await?
//!     .expect("dataset not found");
//!
//! // Read everything...
//! let all = ds.read_full::<u32>(&file).await?;
//! println!("{:?} -> {} values", all.shape, all.data.len());
//!
//! // ...or just the slice you need. Only the bytes covering it are fetched.
//! let slice = ds.read_range::<u32>(&[1_000..2_000], &file).await?;
//! println!("{:?}", slice.shape);
//! # Ok(())
//! # }
//! ```
//!
//! Element types are checked against the on-disk datatype at read time via the
//! [`H5Type`] trait, which is implemented for the fixed-width integer and float
//! primitives. Asking for the wrong type returns [`H5Error::TypeMismatch`]
//! rather than reinterpreting the bytes.
//!
//! # Round trips
//!
//! Over object storage a request costs far more than the bytes it carries, so
//! h5rs is built to minimise them rather than to minimise bytes read. Listing
//! every dataset in a 1.4 GB file costs one round trip; reading a 133 MB
//! dataset spread over 2000 chunks costs about six.
//!
//! Two things get it there. Metadata is read in aligned blocks, so following a
//! pointer to a nearby structure usually needs no further request. And wherever
//! a set of addresses is known at once — a B-tree level, a group's object
//! headers, the chunks a selection overlaps — they are fetched together, merged
//! where they lie close and issued in parallel where they do not.
//!
//! [`ObjectStoreFile::stats`] reports what a read actually cost, and
//! [`ReadOptions`] tunes the trade-off.
//!
//! # Overlapping I/O and decompression
//!
//! Reading compressed chunks is two kinds of work at once, and h5rs pipelines
//! them: requests are kept in flight while each one that lands is decoded and
//! copied into the output. Whichever resource is scarcer sets the pace and the
//! other disappears behind it.
//!
//! Where the decompression runs is the caller's choice, because it depends
//! entirely on the host — see [`compute`]. The default runs it on the async
//! task, which still overlaps with I/O but uses one core; handing h5rs a
//! [`ThreadPoolCompute`] or a pool of your own spreads it over many.
//!
//! [`ObjectStoreFile::stats`]: crate::object_store::ObjectStoreFile::stats
//! [`ReadOptions`]: crate::object_store::ReadOptions
//! [`ThreadPoolCompute`]: crate::compute::ThreadPoolCompute
//!
//! # Errors
//!
//! h5rs reads files it did not write, usually over a network, so every failure
//! mode is a returned [`H5Error`] — malformed input never panics. A file that
//! uses a part of the format h5rs has not implemented yields
//! [`H5Error::Unsupported`] rather than silently returning wrong data.
//!
//! # Format coverage
//!
//! h5rs reads both of the on-disk formats the HDF5 library emits: the
//! "earliest" encoding that `h5py` and most writers produce by default, and the
//! "latest" encoding introduced with HDF5 1.10 and selected by
//! `H5Pset_libver_bounds`.
//!
//! | Area | Supported | Not yet |
//! |---|---|---|
//! | Superblock | v0, v1, v2, v3 | non-zero base address (user block) |
//! | Object header | v1, v2 (`OHDR`), continuation blocks | shared messages |
//! | Group links | symbol table, compact link messages, fractal heap + v2 B-tree | soft, external and user-defined links |
//! | Chunk index | v1 B-tree, single chunk, implicit, fixed array, extensible array, v2 B-tree | |
//! | Data layout | chunked, contiguous, compact (message v1-v5) | virtual |
//! | Filters | deflate (gzip), shuffle, per-chunk filter masks | szip, blosc, lzf, n-bit, scale-offset, fletcher32 |
//! | Dataspaces | simple, scalar, null (message v1-v2) | permutation indices |
//! | Datatypes | fixed-point, floating-point, string, variable-length | compound, enum, array, reference |
//! | Attributes | message v1-v3, compact and dense | shared datatypes and dataspaces |
//!
//! Addresses and lengths must be 8 bytes wide, which is the HDF5 default;
//! anything else is reported as unsupported rather than mis-parsed. Metadata
//! checksums are parsed past but not verified.
//!
//! Writing is out of scope, by design.
//!
//! [HDF5 file format]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html
//! [`ObjectStoreFile`]: crate::object_store::ObjectStoreFile
//! [`H5Type`]: crate::h5type::H5Type
//! [`H5Error`]: crate::error::H5Error
//! [`H5Error::Parse`]: crate::error::H5Error::Parse
//! [`H5Error::TypeMismatch`]: crate::error::H5Error::TypeMismatch
//! [`H5Error::Unsupported`]: crate::error::H5Error::Unsupported

#![deny(unsafe_code)]
#![warn(missing_docs)]

use crate::error::{H5Error, H5Result};
use crate::format::{
    btree::collect_btree_leaves,
    metadata::{GroupBTreeV1, GroupPointerV1, GroupSymbolTableNode, LoadedLocalHeap, Superblock},
    object::{AttributeMessage, DataObjectHeader, LinkMessage, LinkTarget},
};
use binrw::BinRead;

use crate::object_store::ObjectStoreFile;

pub mod compute;
pub(crate) mod dataset;
pub mod error;
pub(crate) mod format;
pub mod h5type;
#[cfg(all(test, target_arch = "wasm32"))]
mod node_store;
pub mod object_store;

pub use dataset::{Dataset, NdArray};

/// Metadata about a dataset found during HDF5 tree walking.
#[derive(Debug, Clone)]
pub struct DatasetInfo {
    /// Full internal path, e.g. "/group1/dataset".
    pub path: String,
    /// Dataset shape (dimensions).
    pub shape: Vec<u64>,
    /// Chunk shape, if the dataset is chunked.
    pub chunk_shape: Option<Vec<u64>>,
    /// How the raw data is stored: `"chunked"`, `"contiguous"`, `"compact"`
    /// or `"virtual"`.
    pub layout: &'static str,
    /// (is_float, is_signed, byte_size) for the scalar type.
    pub dtype_info: (bool, bool, usize),
    /// Names of HDF5 filters applied (e.g. "Deflate", "Shuffle").
    pub filters: Vec<String>,
}

/// Walk an HDF5 file's group tree and return info about all datasets.
pub async fn list_datasets(
    file: &crate::object_store::ObjectStoreFile,
) -> error::H5Result<Vec<DatasetInfo>> {
    let f = File::open(file).await?;
    let mut results = Vec::new();
    let mut stack: Vec<(Group, String)> = vec![(f.root_group, String::new())];

    while let Some((group, prefix)) = stack.pop() {
        let refs = group.object_refs(file).await?;
        // Every member's address is known before any of their headers is read,
        // so read them together instead of one round trip per object.
        let addresses: Vec<u64> = refs.iter().map(|(_, address)| *address).collect();
        let headers = read_object_headers(file, &addresses).await?;

        for ((name, _), header) in refs.iter().zip(headers) {
            let child_path = if prefix.is_empty() {
                format!("/{name}")
            } else {
                format!("{prefix}/{name}")
            };

            if let Some(g) = header.to_group(file).await? {
                stack.push((g, child_path));
            } else if let Some(ds) = header.to_dataset(name.clone(), file).await? {
                let dtype_info = match ds.dtype_info() {
                    (false, false, 0) => (false, false, ds.datatype.element_size()?),
                    info => info,
                };
                results.push(DatasetInfo {
                    path: child_path,
                    shape: ds.shape(),
                    chunk_shape: ds.chunk_shape(),
                    layout: ds.layout_name(),
                    dtype_info,
                    filters: ds.filter_names(),
                });
            }
        }
    }

    results.sort_by(|a, b| a.path.cmp(&b.path));
    Ok(results)
}

/// Open an HDF5 file and navigate to a dataset by internal path.
///
/// `internal_path` is a slice of group/dataset names, e.g. `["group1", "dataset"]`.
/// Returns `None` if the path does not exist or does not name a dataset.
pub async fn open_dataset(
    file: &crate::object_store::ObjectStoreFile,
    internal_path: &[&str],
) -> error::H5Result<Option<Dataset>> {
    let Some((dataset_name, groups)) = internal_path.split_last() else {
        return Ok(None);
    };

    let f = File::open(file).await?;
    let mut current_group = f.root_group;

    // Navigate through groups (all but last segment)
    for &segment in groups {
        match current_group.find_obj(segment, file).await? {
            Some(obj) => match obj.to_group(file).await? {
                Some(g) => current_group = g,
                None => return Ok(None),
            },
            None => return Ok(None),
        }
    }

    // Last segment should be a dataset
    match current_group.find_obj(*dataset_name, file).await? {
        Some(obj) => obj.header.to_dataset(dataset_name.to_string(), file).await,
        None => Ok(None),
    }
}

/// Read an object header at `address` and follow its continuation blocks.
async fn read_object_header(file: &ObjectStoreFile, address: u64) -> H5Result<DataObjectHeader> {
    Ok(read_object_headers(file, &[address]).await?.remove(0))
}

/// Read several object headers at once.
///
/// An object header has no bounded size: a compact dataset or a run of
/// attributes can make one arbitrarily large. So a default-sized prefix is read
/// first to learn each header's extent, and only the headers that turn out to
/// be longer are read again. Both passes are batched, so a whole group's worth
/// of headers costs one or two round trips rather than one per object.
async fn read_object_headers(
    file: &ObjectStoreFile,
    addresses: &[u64],
) -> H5Result<Vec<DataObjectHeader>> {
    let prefix_spans: Vec<(u64, u64)> = addresses
        .iter()
        .map(|&a| (a, crate::object_store::METADATA_FETCH_SIZE))
        .collect();
    let mut blocks = file.read_metadata_many(&prefix_spans).await?;

    let mut extents = Vec::with_capacity(addresses.len());
    for bytes in &blocks {
        extents.push(
            crate::format::object::header_chunk_extent(bytes)
                .ok_or_else(|| H5Error::corrupt("truncated object header prefix"))?,
        );
    }

    let long: Vec<usize> = (0..addresses.len())
        .filter(|&i| extents[i] > blocks[i].len() as u64)
        .collect();
    if !long.is_empty() {
        let spans: Vec<(u64, u64)> = long.iter().map(|&i| (addresses[i], extents[i])).collect();
        for (&i, bytes) in long.iter().zip(file.read_metadata_many(&spans).await?) {
            blocks[i] = bytes;
        }
    }

    let mut headers = Vec::with_capacity(addresses.len());
    for bytes in blocks {
        let mut header = DataObjectHeader::read_le(&mut std::io::Cursor::new(bytes))?;
        // Continuation blocks are rare, and each one's address is only known
        // once its parent is parsed, so these stay one at a time.
        header.load_continuation_messages(file).await?;
        headers.push(header);
    }
    Ok(headers)
}

struct File {
    pub root_group: Group,
}

impl File {
    pub async fn open(file: &ObjectStoreFile) -> H5Result<File> {
        let sb = Superblock::read(file).await?;
        let root_group = read_object_header(file, sb.root_group_address).await?;

        let rg = root_group
            .to_group(file)
            .await?
            .ok_or_else(|| H5Error::corrupt("root object is not a group"))?;

        Ok(File { root_group: rg })
    }
}

struct Object {
    header: DataObjectHeader,
}

impl Object {
    pub async fn to_group(&self, file: &ObjectStoreFile) -> H5Result<Option<Group>> {
        self.header.to_group(file).await
    }
}

/// How much of a symbol table node to read before its length is known. A node
/// holds at most `2 * group_leaf_node_k` entries of 40 bytes plus a small
/// header, which fits comfortably.
const SYMBOL_TABLE_FETCH_SIZE: u64 = 8192;

/// Keep only the hard links, which are the ones h5rs can follow directly.
///
/// Soft, external and user-defined links are skipped rather than reported as
/// errors: a group containing one should still list its other members.
fn hard_links(links: &[LinkMessage]) -> Vec<(String, u64)> {
    links
        .iter()
        .filter_map(|l| match l.target {
            LinkTarget::Hard(address) => Some((l.name.clone(), address)),
            _ => None,
        })
        .collect()
}

/// How a group stores the links to its members.
///
/// Files written before HDF5 1.8 (and any file using the "earliest" format
/// bounds) use a symbol table; newer groups store links either as messages in
/// the group's own object header or, once there are enough of them, in a
/// fractal heap indexed by a version 2 B-tree.
enum GroupLinks {
    SymbolTable {
        btree: GroupBTreeV1,
        heap: LoadedLocalHeap,
    },
    Compact(Vec<LinkMessage>),
    Dense {
        fractal_heap_address: u64,
        name_btree_address: u64,
    },
}

struct Group {
    links: GroupLinks,
    /// Read by the differential tests, which compare every attribute against
    /// the HDF5 C library.
    #[allow(dead_code)]
    pub attributes: Vec<AttributeMessage>,
}

impl Group {
    /// Every member of this group, as `(name, object header address)`.
    async fn object_refs(&self, file: &ObjectStoreFile) -> H5Result<Vec<(String, u64)>> {
        match &self.links {
            GroupLinks::SymbolTable { btree, heap } => {
                let ptrs: Vec<GroupPointerV1> = collect_btree_leaves(file, btree.clone()).await?;
                // Read every symbol table node in one batched request rather
                // than walking them one at a time.
                let spans: Vec<(u64, u64)> = ptrs
                    .iter()
                    .map(|p| (p.child_pointer, SYMBOL_TABLE_FETCH_SIZE))
                    .collect();
                let blocks = file.read_metadata_many(&spans).await?;

                let mut res = vec![];
                for bytes in blocks {
                    let st = GroupSymbolTableNode::read_le(&mut std::io::Cursor::new(bytes))?;
                    for e in &st.entries {
                        let name = heap.get_string(e.link_name_offset)?;
                        res.push((name, e.object_header_address));
                    }
                }
                Ok(res)
            }
            GroupLinks::Compact(links) => Ok(hard_links(links)),
            GroupLinks::Dense {
                fractal_heap_address,
                name_btree_address,
            } => {
                let links =
                    format::dense::read_links(file, *fractal_heap_address, *name_btree_address)
                        .await?;
                Ok(hard_links(&links))
            }
        }
    }

    async fn find_obj(
        &self,
        name: impl AsRef<str>,
        file: &ObjectStoreFile,
    ) -> H5Result<Option<Object>> {
        let r = self.object_refs(file).await?;
        let Some((_, address)) = r.iter().find(|(n, _)| n == name.as_ref()) else {
            return Ok(None);
        };

        Ok(Some(Object {
            header: read_object_header(file, *address).await?,
        }))
    }
}

#[cfg(all(test, not(target_arch = "wasm32"), feature = "hdf5-compare"))]
mod roundtrip {
    use std::fmt::Debug;

    use object_store::{local::LocalFileSystem, path::Path};

    use crate::error::H5Result;
    use crate::h5type::H5Type;
    use crate::object_store::ObjectStoreFile;

    pub(crate) fn test_file_abs(path: &std::path::Path) -> ObjectStoreFile {
        let parent = path.parent().unwrap();
        let filename = path.file_name().unwrap().to_str().unwrap();
        let store = LocalFileSystem::new_with_prefix(parent).unwrap();
        ObjectStoreFile::new(Box::new(store), Path::from(filename))
    }

    /// Trait for generating deterministic test values from a flat index.
    trait TestValue: Sized {
        fn from_index(i: usize) -> Self;
    }

    macro_rules! impl_test_value {
        ($($ty:ty),*) => {
            $(impl TestValue for $ty {
                fn from_index(i: usize) -> Self { (i % 251) as $ty }
            })*
        };
    }

    impl_test_value!(u8, u16, u32, u64, i8, i16, i32, i64, f32, f64);

    /// Extract elements from a flat row-major array at the given N-dimensional sub-range.
    fn collect_subrange<T: Copy>(
        data: &[T],
        shape: &[usize],
        ranges: &[std::ops::Range<u64>],
    ) -> Vec<T> {
        let ndim = shape.len();
        let range_shape: Vec<usize> = ranges.iter().map(|r| (r.end - r.start) as usize).collect();
        let total: usize = range_shape.iter().product();

        (0..total)
            .map(|linear| {
                let mut remaining = linear;
                let mut src_linear = 0usize;
                for d in 0..ndim {
                    let range_stride: usize = range_shape[d + 1..].iter().product();
                    let idx_in_range = remaining / range_stride;
                    remaining %= range_stride;
                    let src_dim_idx = ranges[d].start as usize + idx_in_range;
                    let src_stride: usize = shape[d + 1..].iter().product();
                    src_linear += src_dim_idx * src_stride;
                }
                data[src_linear]
            })
            .collect()
    }

    /// Which on-disk format the HDF5 C library should write.
    ///
    /// The two bounds produce genuinely different files: `Earliest` gives a v0
    /// superblock, v1 object headers and v1 B-tree chunk indexes, while
    /// `Latest` gives a v3 superblock, v2 object headers, link messages and one
    /// of the newer chunk indexes. Every round-trip case runs against both.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub(crate) enum LibVer {
        Earliest,
        Latest,
    }

    impl LibVer {
        pub(crate) const ALL: [LibVer; 2] = [LibVer::Earliest, LibVer::Latest];

        /// A file builder pinned to these format bounds.
        pub(crate) fn builder(self) -> hdf5::FileBuilder {
            use hdf5::plist::file_access::LibraryVersion as V;
            let mut b = hdf5::File::with_options();
            match self {
                LibVer::Earliest => {
                    b.with_fapl(|f| f.libver_bounds(V::Earliest, V::V18));
                }
                LibVer::Latest => {
                    b.with_fapl(|f| f.libver_bounds(V::latest(), V::latest()));
                }
            }
            b
        }
    }

    /// How the C library should store the raw data.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Storage {
        /// Chunked, with the chunk shape given by `RoundtripTest::chunk`.
        Chunked,
        /// One contiguous run outside the object header.
        Contiguous,
        /// Inline in the object header.
        Compact,
    }

    /// Maximum extent of one dimension, used to force an unlimited dimension
    /// and so exercise the extensible array chunk index.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum MaxDims {
        /// Maximum equals the current extent.
        Fixed,
        /// The outermost dimension is unlimited.
        UnlimitedFirst,
    }

    struct RoundtripTest {
        shape: Vec<usize>,
        chunk: Option<Vec<usize>>,
        deflate: Option<u8>,
        shuffle: bool,
        libver: LibVer,
        max_dims: MaxDims,
        storage: Storage,
        /// Assert the file really uses this chunk index, so a case cannot
        /// silently stop exercising the structure it was written for.
        expect_index: Option<&'static str>,
    }

    impl RoundtripTest {
        fn new() -> Self {
            Self {
                shape: vec![],
                chunk: None,
                deflate: None,
                shuffle: false,
                libver: LibVer::Earliest,
                max_dims: MaxDims::Fixed,
                storage: Storage::Chunked,
                expect_index: None,
            }
        }

        fn shape(mut self, s: &[usize]) -> Self {
            self.shape = s.to_vec();
            self
        }

        fn chunk(mut self, c: &[usize]) -> Self {
            self.chunk = Some(c.to_vec());
            self
        }

        fn deflate(mut self, level: u8) -> Self {
            self.deflate = Some(level);
            self
        }

        fn shuffle(mut self) -> Self {
            self.shuffle = true;
            self
        }

        fn libver(mut self, libver: LibVer) -> Self {
            self.libver = libver;
            self
        }

        fn unlimited_first(mut self) -> Self {
            self.max_dims = MaxDims::UnlimitedFirst;
            self
        }

        fn expect_index(mut self, name: &'static str) -> Self {
            self.expect_index = Some(name);
            self
        }

        fn storage(mut self, storage: Storage) -> Self {
            self.storage = storage;
            self
        }

        /// Returns the name of the chunk index the file actually used, so
        /// callers can check that a case exercised what it meant to.
        async fn run<T>(&self) -> H5Result<&'static str>
        where
            T: H5Type + hdf5::H5Type + TestValue + PartialEq + Debug,
        {
            let tmp = tempfile::NamedTempFile::new().unwrap();
            let path = tmp.path();

            let total: usize = self.shape.iter().product();
            let data: Vec<T> = (0..total).map(T::from_index).collect();

            // Write with the HDF5 C library, in whichever on-disk format this
            // case is exercising.
            {
                let hf = self.libver.builder().create(path).unwrap();
                let extents: Vec<(usize, Option<usize>)> = self
                    .shape
                    .iter()
                    .enumerate()
                    .map(|(d, &n)| match self.max_dims {
                        MaxDims::UnlimitedFirst if d == 0 => (n, None),
                        _ => (n, Some(n)),
                    })
                    .collect();
                let mut builder = hf.new_dataset::<T>().shape(&extents[..]);
                match self.storage {
                    Storage::Chunked => {
                        if let Some(ref c) = self.chunk {
                            builder = builder.chunk(&c[..]);
                        }
                        if self.shuffle {
                            builder = builder.shuffle();
                        }
                        if let Some(level) = self.deflate {
                            builder = builder.deflate(level);
                        }
                    }
                    Storage::Contiguous => {
                        builder = builder.layout(hdf5::plist::dataset_create::Layout::Contiguous);
                    }
                    Storage::Compact => {
                        builder = builder.layout(hdf5::plist::dataset_create::Layout::Compact);
                    }
                }
                let ds = builder.create("data").unwrap();
                ds.write_raw(&data).unwrap();
            }

            // Read with h5rs
            let file = test_file_abs(path);
            let f = super::File::open(&file).await?;
            let obj = f.root_group.find_obj("data", &file).await?.unwrap();
            let ds = obj
                .header
                .to_dataset("data".to_string(), &file)
                .await?
                .unwrap();
            let cds = ds;

            if let Some(expected) = self.expect_index {
                assert_eq!(
                    cds.chunk_index_name().unwrap_or("none"),
                    expected,
                    "{:?}: expected a {expected} chunk index",
                    self.libver
                );
            }
            let expected_layout = match self.storage {
                Storage::Chunked => "chunked",
                Storage::Contiguous => "contiguous",
                Storage::Compact => "compact",
            };
            assert_eq!(
                cds.layout_name(),
                expected_layout,
                "{:?}: expected a {expected_layout} layout",
                self.libver
            );

            // Full read
            let result = cds.read_full::<T>(&file).await?;
            assert_eq!(result.shape, self.shape, "shape mismatch");
            assert_eq!(&result.data[..], &data[..], "data mismatch");

            // Sub-range read (middle quarter in each dimension)
            if total > 0 {
                let sel: Vec<std::ops::Range<u64>> = self
                    .shape
                    .iter()
                    .map(|&d| {
                        let start = (d / 4) as u64;
                        let end = (3 * d / 4).max(d / 4 + 1) as u64;
                        start..end
                    })
                    .collect();

                let range_result = cds.read_range::<T>(&sel, &file).await?;
                let expected_shape: Vec<usize> =
                    sel.iter().map(|r| (r.end - r.start) as usize).collect();
                assert_eq!(range_result.shape, expected_shape, "range shape mismatch");

                let expected_data = collect_subrange(&data, &self.shape, &sel);
                assert_eq!(
                    &range_result.data[..],
                    &expected_data[..],
                    "range data mismatch"
                );
            }

            Ok(cds.chunk_index_name().unwrap_or("none"))
        }
    }

    /// Run a case against both on-disk formats, so a case can only pass if
    /// h5rs reads the old and the new encoding identically.
    macro_rules! roundtrip {
        ($name:ident, $T:ty, $builder:expr) => {
            #[tokio::test]
            async fn $name() -> H5Result<()> {
                for libver in LibVer::ALL {
                    $builder
                        .libver(libver)
                        .run::<$T>()
                        .await
                        .unwrap_or_else(|e| panic!("{libver:?}: {e}"));
                }
                Ok(())
            }
        };
    }

    // -- Dimensions --
    roundtrip!(
        dim_1d,
        u32,
        RoundtripTest::new().shape(&[10000]).chunk(&[1000])
    );
    roundtrip!(
        dim_2d,
        f64,
        RoundtripTest::new().shape(&[100, 200]).chunk(&[32, 64])
    );
    roundtrip!(
        dim_3d,
        u8,
        RoundtripTest::new().shape(&[10, 20, 30]).chunk(&[4, 8, 16])
    );

    // -- Filters --
    roundtrip!(
        filter_none,
        u32,
        RoundtripTest::new().shape(&[10000]).chunk(&[1000])
    );
    roundtrip!(
        filter_deflate,
        u32,
        RoundtripTest::new()
            .shape(&[10000])
            .chunk(&[1000])
            .deflate(1)
    );
    roundtrip!(
        filter_shuffle_deflate,
        u32,
        RoundtripTest::new()
            .shape(&[10000])
            .chunk(&[1000])
            .shuffle()
            .deflate(1)
    );

    // -- Data types --
    roundtrip!(
        type_u8,
        u8,
        RoundtripTest::new().shape(&[5000]).chunk(&[500]).deflate(1)
    );
    roundtrip!(
        type_u32,
        u32,
        RoundtripTest::new().shape(&[5000]).chunk(&[500]).deflate(1)
    );
    roundtrip!(
        type_u64,
        u64,
        RoundtripTest::new().shape(&[5000]).chunk(&[500]).deflate(1)
    );
    roundtrip!(
        type_i32,
        i32,
        RoundtripTest::new().shape(&[5000]).chunk(&[500]).deflate(1)
    );
    roundtrip!(
        type_f32,
        f32,
        RoundtripTest::new().shape(&[5000]).chunk(&[500]).deflate(1)
    );
    roundtrip!(
        type_f64,
        f64,
        RoundtripTest::new().shape(&[5000]).chunk(&[500]).deflate(1)
    );

    // -- Edge cases --
    roundtrip!(
        edge_partial_chunks,
        u32,
        RoundtripTest::new().shape(&[1003]).chunk(&[100])
    );
    roundtrip!(
        edge_single_element_chunks,
        u32,
        RoundtripTest::new().shape(&[100]).chunk(&[1])
    );
    roundtrip!(
        edge_chunk_equals_dim,
        u32,
        RoundtripTest::new().shape(&[50]).chunk(&[50])
    );
    roundtrip!(
        edge_single_element,
        u32,
        RoundtripTest::new().shape(&[1]).chunk(&[1])
    );

    /// Build a small chunked u32 dataset and hand back everything needed to
    /// exercise the error paths against it.
    async fn error_case_dataset(
        tmp: &tempfile::NamedTempFile,
    ) -> H5Result<(ObjectStoreFile, super::Dataset)> {
        let path = tmp.path();
        {
            let hf = hdf5::File::with_options()
                .with_fapl(|f| {
                    f.libver_bounds(
                        hdf5::plist::file_access::LibraryVersion::Earliest,
                        hdf5::plist::file_access::LibraryVersion::V18,
                    )
                })
                .create(path)
                .unwrap();
            let ds = hf
                .new_dataset::<u32>()
                .shape(&[100][..])
                .chunk(&[10][..])
                .create("data")
                .unwrap();
            ds.write_raw(&(0u32..100).collect::<Vec<_>>()).unwrap();
        }

        let file = test_file_abs(path);
        let f = super::File::open(&file).await?;
        let obj = f.root_group.find_obj("data", &file).await?.unwrap();
        let ds = obj
            .header
            .to_dataset("data".to_string(), &file)
            .await?
            .unwrap();
        Ok((file, ds))
    }

    /// Reading a u32 dataset as f64 must report a mismatch, not reinterpret
    /// the bytes and not panic.
    #[tokio::test]
    async fn wrong_element_type_is_an_error() -> H5Result<()> {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let (file, cds) = error_case_dataset(&tmp).await?;

        let err = cds.read_full::<f64>(&file).await.unwrap_err();
        assert!(
            matches!(err, crate::error::H5Error::TypeMismatch { .. }),
            "expected TypeMismatch, got {err:?}"
        );

        // The correct type still works.
        assert_eq!(cds.read_full::<u32>(&file).await?.data.len(), 100);
        Ok(())
    }

    /// A selection with the wrong number of dimensions must report an error,
    /// not trip an assertion.
    #[tokio::test]
    async fn bad_selection_arity_is_an_error() -> H5Result<()> {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let (file, cds) = error_case_dataset(&tmp).await?;

        let err = cds
            .read_range::<u32>(&[0..10, 0..10], &file)
            .await
            .unwrap_err();
        assert!(
            matches!(err, crate::error::H5Error::InvalidSelection(_)),
            "expected InvalidSelection, got {err:?}"
        );
        Ok(())
    }

    // -- Chunk index structures --
    //
    // Which index the C library picks depends on the dataset's shape and
    // filters, so each of these cases both writes a file with the right shape
    // and asserts that the index it got is the one under test.

    /// Only reachable in the new format; the old format always uses a v1 B-tree.
    #[tokio::test]
    async fn index_single_chunk() -> H5Result<()> {
        RoundtripTest::new()
            .shape(&[64])
            .chunk(&[64])
            .libver(LibVer::Earliest)
            .expect_index("v1 btree")
            .run::<u32>()
            .await?;
        RoundtripTest::new()
            .shape(&[64])
            .chunk(&[64])
            .libver(LibVer::Latest)
            .expect_index("single chunk")
            .run::<u32>()
            .await
            .map(|_| ())
    }

    #[tokio::test]
    async fn index_single_chunk_filtered() -> H5Result<()> {
        RoundtripTest::new()
            .shape(&[4000])
            .chunk(&[4000])
            .shuffle()
            .deflate(1)
            .libver(LibVer::Latest)
            .expect_index("single chunk")
            .run::<u32>()
            .await
            .map(|_| ())
    }

    #[tokio::test]
    async fn index_fixed_array() -> H5Result<()> {
        RoundtripTest::new()
            .shape(&[10_000])
            .chunk(&[100])
            .libver(LibVer::Latest)
            .expect_index("fixed array")
            .run::<u32>()
            .await
            .map(|_| ())
    }

    #[tokio::test]
    async fn index_fixed_array_filtered() -> H5Result<()> {
        RoundtripTest::new()
            .shape(&[10_000])
            .chunk(&[100])
            .shuffle()
            .deflate(1)
            .libver(LibVer::Latest)
            .expect_index("fixed array")
            .run::<u32>()
            .await
            .map(|_| ())
    }

    /// More than 1024 chunks, which pushes the fixed array past one data block
    /// page and exercises the page bitmap.
    #[tokio::test]
    async fn index_fixed_array_paged() -> H5Result<()> {
        RoundtripTest::new()
            .shape(&[300_000])
            .chunk(&[100])
            .libver(LibVer::Latest)
            .expect_index("fixed array")
            .run::<u8>()
            .await
            .map(|_| ())
    }

    /// One unlimited dimension selects the extensible array.
    #[tokio::test]
    async fn index_extensible_array() -> H5Result<()> {
        RoundtripTest::new()
            .shape(&[10_000])
            .chunk(&[100])
            .unlimited_first()
            .libver(LibVer::Latest)
            .expect_index("extensible array")
            .run::<u32>()
            .await
            .map(|_| ())
    }

    #[tokio::test]
    async fn index_extensible_array_filtered() -> H5Result<()> {
        RoundtripTest::new()
            .shape(&[10_000])
            .chunk(&[100])
            .unlimited_first()
            .shuffle()
            .deflate(1)
            .libver(LibVer::Latest)
            .expect_index("extensible array")
            .run::<u32>()
            .await
            .map(|_| ())
    }

    /// Enough chunks to reach the extensible array's secondary blocks and
    /// paged data blocks.
    #[tokio::test]
    async fn index_extensible_array_deep() -> H5Result<()> {
        RoundtripTest::new()
            .shape(&[500_000])
            .chunk(&[64])
            .unlimited_first()
            .libver(LibVer::Latest)
            .expect_index("extensible array")
            .run::<u8>()
            .await
            .map(|_| ())
    }

    /// An unlimited dimension in the old format still uses a v1 B-tree.
    #[tokio::test]
    async fn index_unlimited_old_format() -> H5Result<()> {
        RoundtripTest::new()
            .shape(&[10_000])
            .chunk(&[100])
            .unlimited_first()
            .libver(LibVer::Earliest)
            .expect_index("v1 btree")
            .run::<u32>()
            .await
            .map(|_| ())
    }

    // -- Storage layouts --

    /// Contiguous storage: one row-major run of raw data outside the object
    /// header, with no chunk index at all.
    #[tokio::test]
    async fn layout_contiguous_1d() -> H5Result<()> {
        for libver in LibVer::ALL {
            RoundtripTest::new()
                .shape(&[10_000])
                .storage(Storage::Contiguous)
                .libver(libver)
                .run::<u32>()
                .await
                .unwrap_or_else(|e| panic!("{libver:?}: {e}"));
        }
        Ok(())
    }

    #[tokio::test]
    async fn layout_contiguous_2d() -> H5Result<()> {
        for libver in LibVer::ALL {
            RoundtripTest::new()
                .shape(&[64, 96])
                .storage(Storage::Contiguous)
                .libver(libver)
                .run::<f64>()
                .await
                .unwrap_or_else(|e| panic!("{libver:?}: {e}"));
        }
        Ok(())
    }

    #[tokio::test]
    async fn layout_contiguous_3d() -> H5Result<()> {
        for libver in LibVer::ALL {
            RoundtripTest::new()
                .shape(&[7, 11, 13])
                .storage(Storage::Contiguous)
                .libver(libver)
                .run::<i16>()
                .await
                .unwrap_or_else(|e| panic!("{libver:?}: {e}"));
        }
        Ok(())
    }

    /// Compact storage keeps the raw data inline in the object header, so it is
    /// limited to roughly 64 KiB.
    #[tokio::test]
    async fn layout_compact_1d() -> H5Result<()> {
        for libver in LibVer::ALL {
            RoundtripTest::new()
                .shape(&[500])
                .storage(Storage::Compact)
                .libver(libver)
                .run::<u32>()
                .await
                .unwrap_or_else(|e| panic!("{libver:?}: {e}"));
        }
        Ok(())
    }

    #[tokio::test]
    async fn layout_compact_2d() -> H5Result<()> {
        for libver in LibVer::ALL {
            RoundtripTest::new()
                .shape(&[20, 30])
                .storage(Storage::Compact)
                .libver(libver)
                .run::<f32>()
                .await
                .unwrap_or_else(|e| panic!("{libver:?}: {e}"));
        }
        Ok(())
    }

    /// A range read of a contiguous dataset must fetch only the enclosing byte
    /// span, and must land on the right elements for every offset and length.
    #[tokio::test]
    async fn contiguous_range_reads() -> H5Result<()> {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path();
        let total = 5000usize;
        let data: Vec<u32> = (0..total as u32).collect();

        {
            let hf = LibVer::Latest.builder().create(path).unwrap();
            let ds = hf
                .new_dataset::<u32>()
                .shape(&[total][..])
                .layout(hdf5::plist::dataset_create::Layout::Contiguous)
                .create("data")
                .unwrap();
            ds.write_raw(&data).unwrap();
        }

        let file = test_file_abs(path);
        let ds = crate::open_dataset(&file, &["data"])
            .await?
            .expect("dataset should open");
        assert_eq!(ds.layout_name(), "contiguous");
        assert_eq!(ds.chunk_shape(), None);

        let t = total as u64;
        for range in [
            0..1,
            0..100,
            1..2,
            999..1001,
            t - 1..t,
            t - 100..t,
            0..t,
            // Clamped past the end, and entirely past the end.
            t - 10..t + 50,
            t..t + 10,
            // Empty.
            0..0,
            100..100,
        ] {
            let got = ds
                .read_range::<u32>(std::slice::from_ref(&range), &file)
                .await?;
            let start = range.start.min(t) as usize;
            let end = range.end.min(t) as usize;
            assert_eq!(got.shape, vec![end - start], "range {range:?}: shape");
            assert_eq!(&got.data[..], &data[start..end], "range {range:?}: data");
        }
        Ok(())
    }

    /// A scalar dataset has a rank-zero dataspace and takes an empty selection.
    #[tokio::test]
    async fn scalar_dataset() -> H5Result<()> {
        for libver in LibVer::ALL {
            let tmp = tempfile::NamedTempFile::new().unwrap();
            {
                let hf = libver.builder().create(tmp.path()).unwrap();
                hf.new_dataset::<f64>()
                    .shape(())
                    .create("scalar")
                    .unwrap()
                    .write_raw(&[2.5f64])
                    .unwrap();
            }

            let file = test_file_abs(tmp.path());
            let ds = crate::open_dataset(&file, &["scalar"])
                .await?
                .expect("scalar dataset should open");
            assert_eq!(ds.ndim(), 0, "{libver:?}");
            assert!(ds.shape().is_empty(), "{libver:?}");

            let got = ds.read_full::<f64>(&file).await?;
            assert!(got.shape.is_empty(), "{libver:?}");
            assert_eq!(got.data, vec![2.5f64], "{libver:?}");
        }
        Ok(())
    }

    /// Fixed dimensions, no filters and early allocation select the implicit
    /// index, where chunk addresses are computed rather than stored.
    #[tokio::test]
    // `&[a..b]` is a one-dimensional selection, not a mis-typed range literal.
    #[allow(clippy::single_range_in_vec_init)]
    async fn index_implicit() -> H5Result<()> {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path();
        let data: Vec<u32> = (0..5000).collect();

        {
            let hf = LibVer::Latest.builder().create(path).unwrap();
            let ds = hf
                .new_dataset::<u32>()
                .shape(&[5000][..])
                .chunk(&[125][..])
                .alloc_time(Some(hdf5::plist::dataset_create::AllocTime::Early))
                .create("data")
                .unwrap();
            ds.write_raw(&data).unwrap();
        }

        let file = test_file_abs(path);
        let ds = crate::open_dataset(&file, &["data"])
            .await?
            .expect("dataset should open");
        assert_eq!(ds.chunk_index_name(), Some("implicit"));
        let result = ds.read_full::<u32>(&file).await?;
        assert_eq!(&result.data[..], &data[..]);

        // A sub-range must skip the chunks it does not need.
        let slice = ds.read_range::<u32>(&[1000..2500], &file).await?;
        assert_eq!(&slice.data[..], &data[1000..2500]);
        Ok(())
    }

    /// Two unlimited dimensions select the version 2 B-tree chunk index.
    #[tokio::test]
    async fn index_btree_v2() -> H5Result<()> {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path();
        let shape = [200usize, 300];
        let data: Vec<u32> = (0..shape[0] * shape[1]).map(|i| (i % 251) as u32).collect();

        {
            let hf = LibVer::Latest.builder().create(path).unwrap();
            let extents = [(shape[0], None), (shape[1], None)];
            let ds = hf
                .new_dataset::<u32>()
                .shape(&extents[..])
                .chunk(&[16, 32][..])
                .create("data")
                .unwrap();
            ds.write_raw(&data).unwrap();
        }

        let file = test_file_abs(path);
        let ds = crate::open_dataset(&file, &["data"])
            .await?
            .expect("dataset should open");
        assert_eq!(ds.chunk_index_name(), Some("v2 btree"));
        let result = ds.read_full::<u32>(&file).await?;
        assert_eq!(result.shape, shape.to_vec());
        assert_eq!(&result.data[..], &data[..]);
        Ok(())
    }

    /// Build a file exercising the structures that differ most between the two
    /// formats — nested groups, enough links to force dense link storage,
    /// enough attributes to force dense attribute storage, and a mix of
    /// datatypes — then compare every dataset and attribute against the C
    /// library.
    async fn structural_roundtrip(libver: LibVer) -> H5Result<()> {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path();

        {
            let hf = libver.builder().create(path).unwrap();

            // Root attributes, including a fixed-length string.
            hf.new_attr::<u32>()
                .shape(&[3][..])
                .create("root_nums")
                .unwrap()
                .write_raw(&[7u32, 8, 9])
                .unwrap();
            let label = hf
                .new_attr::<hdf5::types::FixedAscii<16>>()
                .shape(&[1][..])
                .create("root_label")
                .unwrap();
            label
                .write_raw(&[hdf5::types::FixedAscii::<16>::from_ascii(b"h5rs").unwrap()])
                .unwrap();

            // A nested group holding a chunked, filtered dataset with its own
            // attributes.
            let grp = hf.create_group("nested").unwrap();
            let sub = grp.create_group("deeper").unwrap();
            let ds = sub
                .new_dataset::<f64>()
                .shape(&[40, 50][..])
                .chunk(&[8, 16][..])
                .shuffle()
                .deflate(1)
                .create("values")
                .unwrap();
            let values: Vec<f64> = (0..2000).map(|i| i as f64 * 0.5).collect();
            ds.write_raw(&values).unwrap();
            ds.new_attr::<i64>()
                .shape(&[2][..])
                .create("range")
                .unwrap()
                .write_raw(&[-5i64, 5])
                .unwrap();

            // Enough links in one group that the new format switches from
            // compact link messages to a fractal heap indexed by a v2 B-tree.
            let many = hf.create_group("many_links").unwrap();
            for i in 0..40 {
                let d = many
                    .new_dataset::<u16>()
                    .shape(&[8][..])
                    .chunk(&[8][..])
                    .create(format!("item{i:03}").as_str())
                    .unwrap();
                d.write_raw(&(0u16..8).map(|v| v + i as u16).collect::<Vec<_>>())
                    .unwrap();
            }

            // Non-chunked layouts, so the comparison covers those read paths too.
            let contig = hf
                .new_dataset::<i32>()
                .shape(&[30, 40][..])
                .layout(hdf5::plist::dataset_create::Layout::Contiguous)
                .create("contiguous")
                .unwrap();
            contig
                .write_raw(&(0..1200).map(|i| i - 600).collect::<Vec<i32>>())
                .unwrap();

            let compact = hf
                .new_dataset::<u16>()
                .shape(&[24][..])
                .layout(hdf5::plist::dataset_create::Layout::Compact)
                .create("compact")
                .unwrap();
            compact.write_raw(&(0u16..24).collect::<Vec<_>>()).unwrap();

            // Enough attributes that the new format stores them densely too.
            let attrs = hf
                .new_dataset::<u8>()
                .shape(&[16][..])
                .chunk(&[16][..])
                .create("many_attrs")
                .unwrap();
            attrs.write_raw(&(0u8..16).collect::<Vec<_>>()).unwrap();
            for i in 0..40 {
                attrs
                    .new_attr::<u32>()
                    .shape(&[1][..])
                    .create(format!("attr{i:03}").as_str())
                    .unwrap()
                    .write_raw(&[i as u32])
                    .unwrap();
            }
        }

        super::test::compare_object_store_file(test_file_abs(path), path).await?;

        // The dense paths only carry their weight if they actually ran: 40
        // links in one group and 40 attributes on one dataset are past the
        // point where the new format switches to fractal heap storage.
        let file = test_file_abs(path);
        let datasets = crate::list_datasets(&file).await?;
        assert_eq!(
            datasets.len(),
            44,
            "{libver:?}: expected 44 datasets, found {:?}",
            datasets.iter().map(|d| &d.path).collect::<Vec<_>>()
        );

        let layout_of = |name: &str| {
            datasets
                .iter()
                .find(|d| d.path == name)
                .unwrap_or_else(|| panic!("{name} missing"))
                .layout
        };
        assert_eq!(layout_of("/contiguous"), "contiguous", "{libver:?}");
        assert_eq!(layout_of("/compact"), "compact", "{libver:?}");

        let f = super::File::open(&file).await?;
        let obj = f
            .root_group
            .find_obj("many_attrs", &file)
            .await?
            .expect("many_attrs should exist");
        let attrs = obj.header.all_attributes(&file).await?;
        assert_eq!(attrs.len(), 40, "{libver:?}: expected 40 attributes");

        Ok(())
    }

    #[tokio::test]
    async fn structural_old_format() -> H5Result<()> {
        structural_roundtrip(LibVer::Earliest).await
    }

    #[tokio::test]
    async fn structural_new_format() -> H5Result<()> {
        structural_roundtrip(LibVer::Latest).await
    }

    /// The two formats must present the same tree to callers.
    #[tokio::test]
    async fn both_formats_list_the_same_datasets() -> H5Result<()> {
        let mut listings = vec![];
        let mut files = vec![];
        for libver in LibVer::ALL {
            let tmp = tempfile::NamedTempFile::new().unwrap();
            {
                let hf = libver.builder().create(tmp.path()).unwrap();
                let grp = hf.create_group("g").unwrap();
                for i in 0..30 {
                    let d = grp
                        .new_dataset::<u32>()
                        .shape(&[100][..])
                        .chunk(&[10][..])
                        .create(format!("d{i:02}").as_str())
                        .unwrap();
                    d.write_raw(&(0u32..100).collect::<Vec<_>>()).unwrap();
                }
            }
            let file = test_file_abs(tmp.path());
            let mut paths: Vec<String> = crate::list_datasets(&file)
                .await?
                .into_iter()
                .map(|d| format!("{} {:?} {:?}", d.path, d.shape, d.chunk_shape))
                .collect();
            paths.sort();
            listings.push(paths);
            files.push(tmp);
        }
        assert_eq!(listings[0], listings[1], "listings differ between formats");
        assert_eq!(listings[0].len(), 30);
        Ok(())
    }

    #[tokio::test]
    async fn fuzz_roundtrip() -> H5Result<()> {
        const SEED: u64 = 1235;
        const ITERS: usize = 60;

        struct Prng(u64);
        impl Prng {
            fn next(&mut self) -> u64 {
                // Knuth multiplicative LCG, then a murmur3 finalizer. The raw
                // low bits of an LCG have very short periods — bit 0 simply
                // alternates — which would lock the choices below together.
                self.0 = self
                    .0
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let mut x = self.0;
                x = (x ^ (x >> 33)).wrapping_mul(0xff51afd7ed558ccd);
                x = (x ^ (x >> 33)).wrapping_mul(0xc4ceb9fe1a85ec53);
                x ^ (x >> 33)
            }
            fn range(&mut self, lo: u64, hi: u64) -> u64 {
                lo + self.next() % (hi - lo)
            }
            fn bool(&mut self) -> bool {
                self.next() & 1 == 0
            }
        }

        let mut rng = Prng(SEED);
        let mut indexes_seen: Vec<&'static str> = vec![];
        let mut layouts_seen: Vec<Storage> = vec![];

        for i in 0..ITERS {
            let ndim = rng.range(1, 4) as usize; // 1, 2, or 3
            let max_dim: u64 = match ndim {
                1 => 5000,
                2 => 500,
                _ => 50,
            };
            let shape: Vec<usize> = (0..ndim)
                .map(|_| rng.range(1, max_dim + 1) as usize)
                .collect();
            // One case in eight makes the chunk cover the whole dataset, which
            // is what selects the single-chunk index in the new format.
            let whole_chunk = rng.range(0, 8) == 0;
            let chunk: Vec<usize> = shape
                .iter()
                .map(|&d| {
                    if whole_chunk {
                        d
                    } else {
                        rng.range(1, d as u64 + 1) as usize
                    }
                })
                .collect();
            let deflate = if rng.bool() { Some(1u8) } else { None };
            let shuffle = deflate.is_some() && rng.bool();
            let type_idx = rng.range(0, 10) as usize;
            // Vary the on-disk format and the maximum extents, so the fuzzer
            // reaches every chunk index structure rather than just one.
            let libver = if rng.bool() {
                LibVer::Earliest
            } else {
                LibVer::Latest
            };
            let unlimited = rng.bool();

            // Vary the storage layout too, within what the C library allows:
            // an unlimited dimension requires chunking, and compact storage has
            // to fit in the object header.
            let elem_bytes: usize = match type_idx {
                0 | 4 => 1,
                1 | 5 => 2,
                2 | 6 | 8 => 4,
                _ => 8,
            };
            let total_bytes: usize = shape.iter().product::<usize>() * elem_bytes;
            let storage = match rng.range(0, 4) {
                // A whole-dataset chunk shape is how the single-chunk index is
                // reached, so those cases stay chunked.
                _ if whole_chunk => Storage::Chunked,
                0 if !unlimited => Storage::Contiguous,
                1 if !unlimited && total_bytes <= 16 * 1024 => Storage::Compact,
                _ => Storage::Chunked,
            };

            let mut cfg = RoundtripTest::new()
                .shape(&shape)
                .chunk(&chunk)
                .libver(libver)
                .storage(storage);
            if let Some(level) = deflate {
                cfg = cfg.deflate(level);
            }
            if shuffle {
                cfg = cfg.shuffle();
            }
            if unlimited {
                cfg = cfg.unlimited_first();
            }

            let result = match type_idx {
                0 => cfg.run::<u8>().await,
                1 => cfg.run::<u16>().await,
                2 => cfg.run::<u32>().await,
                3 => cfg.run::<u64>().await,
                4 => cfg.run::<i8>().await,
                5 => cfg.run::<i16>().await,
                6 => cfg.run::<i32>().await,
                7 => cfg.run::<i64>().await,
                8 => cfg.run::<f32>().await,
                _ => cfg.run::<f64>().await,
            };

            let index = result.unwrap_or_else(|e| {
                panic!(
                    "iter {i}: {libver:?} {storage:?} shape={shape:?} chunk={chunk:?} \
                     deflate={deflate:?} shuffle={shuffle} unlimited={unlimited} \
                     type_idx={type_idx}: {e}"
                )
            });
            if !indexes_seen.contains(&index) {
                indexes_seen.push(index);
            }
            if !layouts_seen.contains(&storage) {
                layouts_seen.push(storage);
            }
        }

        indexes_seen.sort_unstable();
        println!("chunk indexes exercised: {indexes_seen:?}");
        println!("storage layouts exercised: {layouts_seen:?}");
        for expected in [Storage::Chunked, Storage::Contiguous, Storage::Compact] {
            assert!(
                layouts_seen.contains(&expected),
                "fuzzer never produced {expected:?} storage; saw {layouts_seen:?}"
            );
        }
        // Guard against the generator drifting into a corner that only ever
        // produces one kind of file.
        for expected in [
            "v1 btree",
            "single chunk",
            "fixed array",
            "extensible array",
        ] {
            assert!(
                indexes_seen.contains(&expected),
                "fuzzer never produced a {expected} index; saw {indexes_seen:?}"
            );
        }

        Ok(())
    }
}

/// Conditional async test attribute: `#[tokio::test]` on native, `#[wasm_bindgen_test]` on WASM.
///
/// Usage:
/// ```ignore
/// #[crate::async_test]
/// async fn my_test() { ... }
/// ```
#[cfg(all(test, not(target_arch = "wasm32")))]
#[allow(unused_imports)]
pub(crate) use tokio::test as async_test;

#[cfg(test)]
mod test {

    #[cfg(all(not(target_arch = "wasm32"), feature = "hdf5-compare"))]
    use std::fmt::Debug;

    use object_store::path::Path;

    use crate::error::H5Result;
    use crate::h5type::H5Type;
    use crate::object_store::ObjectStoreFile;

    const MOL_INFO_FILE: &str = "datasets/frozen_pbmc_donor_c_molecule_info.h5";
    #[cfg(all(not(target_arch = "wasm32"), feature = "hdf5-compare"))]
    const MATRIX_FILE: &str = "datasets/gene_bc_matrix.h5";

    fn test_file(path: &str) -> ObjectStoreFile {
        #[cfg(not(target_arch = "wasm32"))]
        {
            use object_store::local::LocalFileSystem;
            let cwd = std::env::current_dir().unwrap();
            let store = LocalFileSystem::new_with_prefix(&cwd).unwrap();
            ObjectStoreFile::new(Box::new(store), Path::from(path))
        }
        #[cfg(target_arch = "wasm32")]
        {
            let store = crate::node_store::NodeFileSystem::cwd();
            ObjectStoreFile::new(Box::new(store), Path::from(path))
        }
    }

    /// Skip the enclosing test (returning `Ok(())`) when a large sample dataset
    /// is not checked out locally. The files under `datasets/` are gigabytes and
    /// are not distributed with the crate; see the README for how to fetch them.
    macro_rules! require_dataset {
        ($path:expr) => {{
            #[cfg(not(target_arch = "wasm32"))]
            {
                if !std::path::Path::new($path).exists() {
                    println!("SKIP: {} not present (see README: Test data)", $path);
                    return Ok(());
                }
            }
        }};
    }

    // ---- hdf5-dependent tests (native only) ----
    #[cfg(all(not(target_arch = "wasm32"), feature = "hdf5-compare"))]
    use crate::format::object::{AttributeMessage, TypeDescriptor};

    /// Read all chunks of a dataset via h5rs and compare byte-for-byte
    /// against the hdf5 C library (gold standard).
    #[cfg(all(not(target_arch = "wasm32"), feature = "hdf5-compare"))]
    async fn compare_typed<T>(
        cds: &super::Dataset,
        hdf5_ds: &hdf5::Dataset,
        path: &str,
        file: &ObjectStoreFile,
    ) -> H5Result<()>
    where
        T: H5Type + hdf5::H5Type + PartialEq + Debug,
    {
        let expected: Vec<T> = hdf5_ds.read_raw::<T>().unwrap();
        let result = cds.read_full::<T>(file).await?;

        assert_eq!(result.data.len(), expected.len(), "{path}: length mismatch");
        assert_eq!(&result.data[..], &expected[..], "{path}: data mismatch");
        println!(
            "  OK {path} ({} values, shape {:?})",
            expected.len(),
            result.shape
        );

        Ok(())
    }

    /// Dispatch to the correct typed comparison based on the HDF5 datatype.
    #[cfg(all(not(target_arch = "wasm32"), feature = "hdf5-compare"))]
    async fn compare_dataset(
        cds: &super::Dataset,
        hdf5_ds: &hdf5::Dataset,
        path: &str,
        file: &ObjectStoreFile,
    ) -> H5Result<()> {
        match &cds.datatype.type_desc {
            TypeDescriptor::FixedPoint(fp) => match (fp.signed(), fp.size()) {
                (0, 1) => compare_typed::<u8>(cds, hdf5_ds, path, file).await,
                (0, 2) => compare_typed::<u16>(cds, hdf5_ds, path, file).await,
                (0, 4) => compare_typed::<u32>(cds, hdf5_ds, path, file).await,
                (0, 8) => compare_typed::<u64>(cds, hdf5_ds, path, file).await,
                (1, 1) => compare_typed::<i8>(cds, hdf5_ds, path, file).await,
                (1, 2) => compare_typed::<i16>(cds, hdf5_ds, path, file).await,
                (1, 4) => compare_typed::<i32>(cds, hdf5_ds, path, file).await,
                (1, 8) => compare_typed::<i64>(cds, hdf5_ds, path, file).await,
                (s, sz) => panic!("{path}: unsupported FixedPoint signed={s} size={sz}"),
            },
            TypeDescriptor::FloatingPoint(fp) => match fp.size() {
                4 => compare_typed::<f32>(cds, hdf5_ds, path, file).await,
                8 => compare_typed::<f64>(cds, hdf5_ds, path, file).await,
                sz => panic!("{path}: unsupported FloatingPoint size={sz}"),
            },
            other => {
                println!("  SKIP {path}: unsupported type {other:?}");
                Ok(())
            }
        }
    }

    /// Walk a group's children, collecting chunked datasets and sub-groups.
    #[cfg(all(not(target_arch = "wasm32"), feature = "hdf5-compare"))]
    async fn collect_from_group(
        group: &super::Group,
        path: &str,
        file: &ObjectStoreFile,
        hf: &hdf5::File,
        datasets: &mut Vec<(String, super::Dataset)>,
        group_stack: &mut Vec<(super::Group, String)>,
    ) -> H5Result<()> {
        let refs = group.object_refs(file).await?;

        // Check that h5rs found *every* member, not just that the ones it found
        // are correct — otherwise a group whose links h5rs failed to read would
        // silently pass as an empty group.
        let expected = if path == "/" {
            hf.len()
        } else {
            hf.group(path).unwrap().len()
        };
        assert_eq!(
            refs.len() as u64,
            expected,
            "{path}: h5rs found {} members, the C library reports {expected}",
            refs.len()
        );

        for (name, address) in &refs {
            let child_path = if path == "/" {
                format!("/{name}")
            } else {
                format!("{path}/{name}")
            };

            let header = super::read_object_header(file, *address).await?;

            if let Some(g) = header.to_group(file).await? {
                let hdf5_group = hf.group(&child_path).unwrap();
                compare_attrs(&g.attributes, &hdf5_group, &child_path);
                group_stack.push((g, child_path));
            } else if let Some(ds) = header.to_dataset(name.clone(), file).await? {
                let hdf5_ds = hf.dataset(&child_path).unwrap();
                compare_attrs(&ds.attributes, &hdf5_ds, &child_path);
                datasets.push((child_path, ds));
            }
        }
        Ok(())
    }

    #[cfg(all(not(target_arch = "wasm32"), feature = "hdf5-compare"))]
    fn compare_attr_typed<T>(attr: &AttributeMessage, hdf5_attr: &hdf5::Attribute, path: &str)
    where
        T: H5Type + hdf5::H5Type + PartialEq + Debug,
    {
        let ours: Vec<T> = attr.read::<T>().unwrap();
        let expected: Vec<T> = hdf5_attr.read_raw::<T>().unwrap();
        assert_eq!(ours, expected, "{path}@{}: data mismatch", attr.name());
        println!("  OK {path}@{} ({} values)", attr.name(), ours.len());
    }

    /// FixedAscii<N> requires a compile-time size, so we dispatch via macro.
    #[cfg(all(not(target_arch = "wasm32"), feature = "hdf5-compare"))]
    macro_rules! compare_fixed_strings {
        ($attr:expr, $hdf5_attr:expr, $path:expr, $( $n:literal ),*) => {
            match $attr.datatype.element_size().unwrap() {
                $( $n => {
                    let ours = $attr.read_strings().unwrap();
                    let expected: Vec<hdf5::types::FixedAscii<$n>> =
                        $hdf5_attr.read_raw().unwrap();
                    assert_eq!(
                        ours.len(), expected.len(),
                        "{}@{}: string count mismatch", $path, $attr.name()
                    );
                    for (a, b) in ours.iter().zip(expected.iter()) {
                        assert_eq!(
                            a.as_str(), b.as_str(),
                            "{}@{}: string mismatch", $path, $attr.name()
                        );
                    }
                    println!(
                        "  OK {}@{} ({} string{})", $path, $attr.name(),
                        ours.len(), if ours.len() == 1 { "" } else { "s" }
                    );
                }, )*
                sz => println!(
                    "  SKIP {}@{}: unhandled fixed string size {sz}",
                    $path, $attr.name()
                ),
            }
        };
    }

    #[cfg(all(not(target_arch = "wasm32"), feature = "hdf5-compare"))]
    fn compare_attr_strings(attr: &AttributeMessage, hdf5_attr: &hdf5::Attribute, path: &str) {
        compare_fixed_strings!(
            attr, hdf5_attr, path, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
            19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 48, 64, 128, 256
        );
    }

    #[cfg(all(not(target_arch = "wasm32"), feature = "hdf5-compare"))]
    fn compare_attr_untyped(attr: &AttributeMessage, hdf5_attr: &hdf5::Attribute, path: &str) {
        match &attr.datatype.type_desc {
            TypeDescriptor::FixedPoint(fp) => match (fp.signed(), fp.size()) {
                (0, 1) => compare_attr_typed::<u8>(attr, hdf5_attr, path),
                (0, 2) => compare_attr_typed::<u16>(attr, hdf5_attr, path),
                (0, 4) => compare_attr_typed::<u32>(attr, hdf5_attr, path),
                (0, 8) => compare_attr_typed::<u64>(attr, hdf5_attr, path),
                (1, 1) => compare_attr_typed::<i8>(attr, hdf5_attr, path),
                (1, 2) => compare_attr_typed::<i16>(attr, hdf5_attr, path),
                (1, 4) => compare_attr_typed::<i32>(attr, hdf5_attr, path),
                (1, 8) => compare_attr_typed::<i64>(attr, hdf5_attr, path),
                (s, sz) => println!(
                    "  SKIP {path}@{}: unsupported FixedPoint signed={s} size={sz}",
                    attr.name()
                ),
            },
            TypeDescriptor::FloatingPoint(fp) => match fp.size() {
                4 => compare_attr_typed::<f32>(attr, hdf5_attr, path),
                8 => compare_attr_typed::<f64>(attr, hdf5_attr, path),
                sz => println!(
                    "  SKIP {path}@{}: unsupported FloatingPoint size={sz}",
                    attr.name()
                ),
            },
            TypeDescriptor::String(_) => compare_attr_strings(attr, hdf5_attr, path),
            other => {
                println!(
                    "  SKIP {path}@{}: unsupported attr type {other:?}",
                    attr.name()
                );
            }
        }
    }

    #[cfg(all(not(target_arch = "wasm32"), feature = "hdf5-compare"))]
    fn compare_attrs(attrs: &[AttributeMessage], hdf5_loc: &hdf5::Location, path: &str) {
        for attr in attrs {
            let name = attr.name();
            let hdf5_attr = hdf5_loc.attr(&name).unwrap();
            compare_attr_untyped(attr, &hdf5_attr, path);
        }
    }

    /// Open an HDF5 file, walk all groups, and compare every chunked
    /// dataset against the hdf5 C library.
    #[cfg(all(not(target_arch = "wasm32"), feature = "hdf5-compare"))]
    async fn compare_file(path: &str) -> H5Result<()> {
        require_dataset!(path);
        compare_object_store_file(test_file(path), std::path::Path::new(path)).await
    }

    /// Compare every group, dataset and attribute reachable in `file` against
    /// what the HDF5 C library reads from the same file on disk.
    #[cfg(all(not(target_arch = "wasm32"), feature = "hdf5-compare"))]
    pub(crate) async fn compare_object_store_file(
        file: ObjectStoreFile,
        fs_path: &std::path::Path,
    ) -> H5Result<()> {
        let path = fs_path.display().to_string();
        let f = super::File::open(&file).await?;
        let hf = hdf5::File::open(fs_path).unwrap();

        // Compare root group attributes
        compare_attrs(&f.root_group.attributes, &hf, "/");

        let mut datasets: Vec<(String, super::Dataset)> = vec![];
        let mut group_stack: Vec<(super::Group, String)> = vec![];

        collect_from_group(
            &f.root_group,
            "/",
            &file,
            &hf,
            &mut datasets,
            &mut group_stack,
        )
        .await?;
        while let Some((group, gpath)) = group_stack.pop() {
            collect_from_group(&group, &gpath, &file, &hf, &mut datasets, &mut group_stack).await?;
        }

        println!("{path}: found {} chunked datasets", datasets.len());

        for (ds_path, cds) in &datasets {
            let hdf5_ds = hf.dataset(ds_path).unwrap();
            compare_dataset(cds, &hdf5_ds, ds_path, &file).await?;
        }

        Ok(())
    }

    #[cfg(all(not(target_arch = "wasm32"), feature = "hdf5-compare"))]
    #[tokio::test]
    async fn mol_info_file() -> H5Result<()> {
        compare_file(MOL_INFO_FILE).await
    }

    #[cfg(all(not(target_arch = "wasm32"), feature = "hdf5-compare"))]
    #[tokio::test]
    async fn matrix_file() -> H5Result<()> {
        compare_file(MATRIX_FILE).await
    }

    /// Helper: do a full read of a 1D chunked dataset, returning the flat data.
    async fn full_read_1d<T: H5Type>(
        cds: &super::Dataset,
        file: &ObjectStoreFile,
    ) -> H5Result<Vec<T>> {
        Ok(cds.read_full::<T>(file).await?.data)
    }

    /// Open a specific 1D dataset from MOL_INFO_FILE and return it with
    /// the ObjectStoreFile and full reference data.
    async fn setup_range_test(
        dataset_name: &str,
    ) -> H5Result<(ObjectStoreFile, super::Dataset, Vec<u8>)> {
        let file = test_file(MOL_INFO_FILE);
        let f = super::File::open(&file).await?;
        let obj = f.root_group.find_obj(dataset_name, &file).await?.unwrap();
        let ds = obj
            .header
            .to_dataset(dataset_name.to_string(), &file)
            .await?
            .unwrap();
        let cds = ds;
        let full_data = full_read_1d::<u8>(&cds, &file).await?;
        Ok((file, cds, full_data))
    }

    /// Verify that read_range for a 1D range matches the corresponding
    /// slice of the full exhaustive read.
    fn assert_range_eq(
        result: &super::NdArray<u8>,
        full_data: &[u8],
        range: &std::ops::Range<u64>,
        total: u64,
    ) {
        let start = (range.start.min(total)) as usize;
        let end = (range.end.min(total)) as usize;
        let expected = &full_data[start..end];
        assert_eq!(
            result.shape,
            vec![end - start],
            "range {range:?}: shape mismatch"
        );
        assert_eq!(
            result.data.len(),
            expected.len(),
            "range {range:?}: length mismatch"
        );
        assert_eq!(&result.data[..], expected, "range {range:?}: data mismatch");
    }

    #[crate::async_test]
    async fn read_range_basic() -> H5Result<()> {
        require_dataset!(MOL_INFO_FILE);
        let (file, cds, full_data) = setup_range_test("gem_group").await?;
        let total = full_data.len() as u64;
        let chunk_size = cds.chunk_shape().unwrap()[0];
        println!(
            "gem_group: {total} elements, chunk_size={chunk_size}, {} chunks",
            total.div_ceil(chunk_size)
        );

        let ranges: Vec<std::ops::Range<u64>> = vec![
            // Basic slices
            0..100,
            0..1,
            total - 100..total,
            total - 1..total,
            1000..2000,
            // Full dataset
            0..total,
            // Empty ranges
            0..0,
            100..100,
            total..total,
        ];

        for range in &ranges {
            let result = cds
                .read_range::<u8>(std::slice::from_ref(range), &file)
                .await?;
            assert_range_eq(&result, &full_data, range, total);
            println!("  OK read_range {range:?} ({} values)", result.data.len());
        }

        Ok(())
    }

    #[crate::async_test]
    async fn read_range_chunk_boundaries() -> H5Result<()> {
        require_dataset!(MOL_INFO_FILE);
        let (file, cds, full_data) = setup_range_test("gem_group").await?;
        let total = full_data.len() as u64;
        let cs = cds.chunk_shape().unwrap()[0];
        println!("gem_group: {total} elements, chunk_size={cs}");

        let ranges: Vec<std::ops::Range<u64>> = vec![
            // Exactly one chunk
            0..cs,
            cs..2 * cs,
            // Straddle one chunk boundary
            cs - 1..cs + 1,
            cs - 10..cs + 10,
            // Within a single chunk (no boundary crossing)
            10..cs - 10,
            cs + 10..2 * cs - 10,
            // Exactly two chunks
            0..2 * cs,
            // Exactly three chunks
            cs..4 * cs,
            // Start at chunk boundary
            cs..cs + 50,
            2 * cs..2 * cs + 1,
            // End at chunk boundary
            50..cs,
            cs + 50..2 * cs,
            // Range near end of dataset (edge chunk handling)
            total - cs..total,
            total - 1..total,
            total - cs / 2..total,
        ];

        for range in &ranges {
            let result = cds
                .read_range::<u8>(std::slice::from_ref(range), &file)
                .await?;
            assert_range_eq(&result, &full_data, range, total);
            println!("  OK read_range {range:?} ({} values)", result.data.len());
        }

        Ok(())
    }

    #[crate::async_test]
    async fn read_range_various_sizes() -> H5Result<()> {
        require_dataset!(MOL_INFO_FILE);
        let (file, cds, full_data) = setup_range_test("gem_group").await?;
        let total = full_data.len() as u64;
        let cs = cds.chunk_shape().unwrap()[0];

        // Test power-of-two sizes and offsets
        let sizes = [
            1,
            2,
            7,
            63,
            64,
            65,
            1023,
            1024,
            1025,
            cs - 1,
            cs,
            cs + 1,
            cs * 3 + 7,
        ];
        let offsets = [0, 1, cs - 1, cs, cs + 1, total / 2];

        for &offset in &offsets {
            for &size in &sizes {
                let end = (offset + size).min(total);
                if offset >= total {
                    continue;
                }
                let range = offset..end;
                let result = cds
                    .read_range::<u8>(std::slice::from_ref(&range), &file)
                    .await?;
                assert_range_eq(&result, &full_data, &range, total);
            }
        }

        println!(
            "  OK read_range_various_sizes ({} combinations)",
            offsets.len() * sizes.len()
        );
        Ok(())
    }

    #[crate::async_test]
    async fn read_range_clamping() -> H5Result<()> {
        require_dataset!(MOL_INFO_FILE);
        let (file, cds, full_data) = setup_range_test("gem_group").await?;
        let total = full_data.len() as u64;

        // Range extends past dataset extent — should be clamped
        let range = total - 10..total + 100;
        let result = cds
            .read_range::<u8>(std::slice::from_ref(&range), &file)
            .await?;
        assert_range_eq(&result, &full_data, &range, total);
        println!(
            "  OK clamped range {range:?} → {} values (expected 10)",
            result.data.len()
        );

        // Completely past the end
        let range = total..total + 100;
        let result = cds
            .read_range::<u8>(std::slice::from_ref(&range), &file)
            .await?;
        assert_eq!(result.data.len(), 0);
        assert_eq!(result.shape, vec![0]);
        println!("  OK fully out-of-bounds range → empty");

        Ok(())
    }

    #[crate::async_test]
    async fn read_range_u32_dataset() -> H5Result<()> {
        // Test read_range on a u32 dataset (barcode_corrected_reads)
        require_dataset!(MOL_INFO_FILE);
        let file = test_file(MOL_INFO_FILE);
        let f = super::File::open(&file).await?;
        let obj = f
            .root_group
            .find_obj("barcode_corrected_reads", &file)
            .await?
            .unwrap();
        let ds = obj
            .header
            .to_dataset("barcode_corrected_reads".to_string(), &file)
            .await?
            .unwrap();
        let cds = ds;

        let full_data = full_read_1d::<u32>(&cds, &file).await?;
        let total = full_data.len() as u64;
        let cs = cds.chunk_shape().unwrap()[0];
        println!(
            "barcode_corrected_reads: {total} u32 values, chunk_size={cs}, filtered={}",
            cds.filter.is_some()
        );

        let ranges: Vec<std::ops::Range<u64>> = vec![
            0..100,
            cs - 5..cs + 5,
            total - 50..total,
            cs * 2..cs * 2 + 1000,
        ];

        for range in &ranges {
            let result = cds
                .read_range::<u32>(std::slice::from_ref(range), &file)
                .await?;
            let start = range.start as usize;
            let end = range.end.min(total) as usize;
            let expected = &full_data[start..end];
            assert_eq!(
                result.shape,
                vec![end - start],
                "range {range:?}: shape mismatch"
            );
            assert_eq!(&result.data[..], expected, "range {range:?}: data mismatch");
            println!("  OK read_range {range:?} ({} values)", result.data.len());
        }

        Ok(())
    }

    #[crate::async_test]
    async fn read_range_u64_dataset() -> H5Result<()> {
        // Test read_range on a u64 dataset (barcode) to cover larger element types
        require_dataset!(MOL_INFO_FILE);
        let file = test_file(MOL_INFO_FILE);
        let f = super::File::open(&file).await?;
        let obj = f.root_group.find_obj("barcode", &file).await?.unwrap();
        let ds = obj
            .header
            .to_dataset("barcode".to_string(), &file)
            .await?
            .unwrap();
        let cds = ds;

        let full_data = full_read_1d::<u64>(&cds, &file).await?;
        let total = full_data.len() as u64;
        let cs = cds.chunk_shape().unwrap()[0];
        println!("barcode: {total} u64 values, chunk_size={cs}");

        let ranges: Vec<std::ops::Range<u64>> =
            vec![0..50, cs - 1..cs + 1, total - 10..total, cs..cs * 2 + 7];

        for range in &ranges {
            let result = cds
                .read_range::<u64>(std::slice::from_ref(range), &file)
                .await?;
            let start = range.start as usize;
            let end = range.end.min(total) as usize;
            let expected = &full_data[start..end];
            assert_eq!(
                result.shape,
                vec![end - start],
                "range {range:?}: shape mismatch"
            );
            assert_eq!(&result.data[..], expected, "range {range:?}: data mismatch");
            println!("  OK read_range {range:?} ({} values)", result.data.len());
        }

        Ok(())
    }

    #[crate::async_test]
    async fn read_range_chunk_skip_count() -> H5Result<()> {
        require_dataset!(MOL_INFO_FILE);
        // Verify that read_range skips chunks outside the selection.
        // We do this by comparing the number of chunks that overlap with
        // the selection vs the total chunk count.
        let (file, cds, _full_data) = setup_range_test("gem_group").await?;
        let total = cds.shape()[0];
        let cs = cds.chunk_shape().unwrap()[0];
        let all_chunks = cds.collect_chunks(&file).await?;
        let total_chunks = all_chunks.len();

        // A range covering ~3 chunks in the middle
        let start = cs * 10;
        let end = cs * 13;
        let range = start..end;

        let overlapping = all_chunks
            .iter()
            .filter(|c| {
                let co = c.offsets[0];
                let ce = (co + cs).min(total);
                co < end && ce > start
            })
            .count();

        let result = cds
            .read_range::<u8>(std::slice::from_ref(&range), &file)
            .await?;
        assert_eq!(result.data.len(), (end - start) as usize);

        println!("  range {range:?}: {overlapping} chunks needed out of {total_chunks} total");
        assert!(
            overlapping < total_chunks,
            "expected fewer chunks than total"
        );
        assert_eq!(overlapping, 3, "expected exactly 3 chunks for this range");

        Ok(())
    }

    // ---- I/O shape ----

    /// Guard the round-trip count against regressions.
    ///
    /// The absolute numbers matter less than their scale: reading a dataset
    /// spread over hundreds of chunks, and listing a group of hundreds of
    /// datasets, must both stay in the single digits rather than growing with
    /// the number of chunks or objects.
    #[cfg(all(not(target_arch = "wasm32"), feature = "hdf5-compare"))]
    #[tokio::test]
    // `&[a..b]` is a one-dimensional selection, not a mis-typed range literal.
    #[allow(clippy::single_range_in_vec_init)]
    async fn round_trips_do_not_scale_with_object_count() -> H5Result<()> {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        {
            let hf = hdf5::File::create(tmp.path()).unwrap();
            let grp = hf.create_group("g").unwrap();
            for i in 0..120 {
                let d = grp
                    .new_dataset::<u32>()
                    .shape(&[4096][..])
                    .chunk(&[64][..])
                    .create(format!("d{i:03}").as_str())
                    .unwrap();
                d.write_raw(&(0u32..4096).collect::<Vec<_>>()).unwrap();
            }
        }
        let path = tmp.path();

        // 120 datasets, each with 64 chunks.
        let file = crate::roundtrip::test_file_abs(path);
        let listed = crate::list_datasets(&file).await?;
        assert_eq!(listed.len(), 120);
        let listing = file.stats();
        assert!(
            listing.batches <= 12,
            "listing 120 datasets took {} round trips: {listing:?}",
            listing.batches
        );

        let file = crate::roundtrip::test_file_abs(path);
        let ds = crate::open_dataset(&file, &["g", "d000"]).await?.unwrap();
        let all = ds.read_full::<u32>(&file).await?;
        assert_eq!(all.data.len(), 4096);
        let full = file.stats();
        assert!(
            full.batches <= 8,
            "reading 64 chunks took {} round trips: {full:?}",
            full.batches
        );

        // A second range read reuses the chunk index, so it costs at most the
        // one fetch of the chunk it needs.
        let before = file.stats().batches;
        ds.read_range::<u32>(&[100..200], &file).await?;
        let extra = file.stats().batches - before;
        assert!(
            extra <= 1,
            "a repeat range read took {extra} extra round trips"
        );

        Ok(())
    }

    /// Measure how many round trips and bytes each workload costs across a
    /// range of read-ahead settings, so the defaults are chosen from
    /// measurements rather than guessed.
    ///
    /// `waits` is the number of times the reader had to stop and wait on the
    /// store; requests issued within one wait go in parallel, so that column is
    /// what decides elapsed time over a high-latency link.
    ///
    /// Run with: `cargo test --features hdf5-compare io_tuning -- --ignored --nocapture`
    #[cfg(all(not(target_arch = "wasm32"), feature = "hdf5-compare"))]
    #[tokio::test]
    #[ignore]
    async fn io_tuning() -> H5Result<()> {
        use crate::object_store::ReadOptions;

        require_dataset!(MOL_INFO_FILE);
        require_dataset!(MATRIX_FILE);
        const KIB: u64 = 1024;
        const BLOCKS: [u64; 6] = [
            8 * KIB,
            64 * KIB,
            128 * KIB,
            256 * KIB,
            512 * KIB,
            1024 * KIB,
        ];

        // A file whose metadata is densely packed, which is where read-ahead
        // has the most to gain.
        let dense = tempfile::NamedTempFile::new().unwrap();
        {
            let hf = hdf5::File::create(dense.path()).unwrap();
            let grp = hf.create_group("g").unwrap();
            for i in 0..200 {
                let d = grp
                    .new_dataset::<u32>()
                    .shape(&[64][..])
                    .chunk(&[64][..])
                    .create(format!("d{i:03}").as_str())
                    .unwrap();
                d.write_raw(&(0u32..64).collect::<Vec<_>>()).unwrap();
                d.new_attr::<u32>()
                    .shape(&[1][..])
                    .create("n")
                    .unwrap()
                    .write_raw(&[i as u32])
                    .unwrap();
            }
        }
        let dense_path = dense.path().to_path_buf();

        // The adversarial shape for read-ahead: many datasets whose headers are
        // pushed far apart by the bulk data written between them, so a
        // speculative block rarely contains anything else that is wanted.
        let spread = tempfile::NamedTempFile::new().unwrap();
        {
            let hf = hdf5::File::create(spread.path()).unwrap();
            let grp = hf.create_group("g").unwrap();
            let payload: Vec<u32> = (0..250_000).collect();
            for i in 0..60 {
                let d = grp
                    .new_dataset::<u32>()
                    .shape(&[payload.len()][..])
                    .chunk(&[payload.len()][..])
                    .create(format!("d{i:03}").as_str())
                    .unwrap();
                d.write_raw(&payload).unwrap();
            }
        }
        let spread_path = spread.path().to_path_buf();

        let header: String = BLOCKS
            .iter()
            .map(|b| format!("{:>14}", format!("{}K", b / 1024)))
            .collect();
        println!("\n{:<34}{header}", "workload (waits / MB read)");
        println!("{:-<104}", "");

        // Each workload is run once per block size on a fresh handle, so no
        // cache carries over between measurements.
        for (name, workload) in workloads(&dense_path, &spread_path) {
            let mut row = String::new();
            for block in BLOCKS {
                let options = ReadOptions {
                    metadata_block_size: block,
                    ..ReadOptions::default()
                };
                let file = workload.open(options);
                workload.run(&file).await?;
                let s = file.stats();
                row.push_str(&format!(
                    "{:>14}",
                    format!(
                        "{} / {:.1}",
                        s.batches,
                        s.bytes_fetched as f64 / (1024.0 * 1024.0)
                    )
                ));
            }
            println!("{name:<34}{row}");
        }

        // The same workloads with batching switched off, to show what the
        // level-wise and per-chunk batching is worth on its own.
        println!(
            "\n{:<34}{:>14}{:>14}",
            "batching off vs on (waits)", "one-by-one", "batched"
        );
        println!("{:-<62}", "");
        for (name, workload) in workloads(&dense_path, &spread_path) {
            let mut row = String::new();
            for (max_request_bytes, io_concurrency) in [(1, 1), (8 * 1024 * 1024, 16)] {
                let options = ReadOptions {
                    metadata_block_size: 128 * KIB,
                    max_request_bytes,
                    io_concurrency,
                    ..ReadOptions::default()
                };
                let file = workload.open(options);
                workload.run(&file).await?;
                row.push_str(&format!("{:>14}", file.stats().batches));
            }
            println!("{name:<34}{row}");
        }

        Ok(())
    }

    /// One measurable read pattern.
    #[cfg(all(not(target_arch = "wasm32"), feature = "hdf5-compare"))]
    struct Workload {
        path: std::path::PathBuf,
        kind: WorkloadKind,
    }

    #[cfg(all(not(target_arch = "wasm32"), feature = "hdf5-compare"))]
    enum WorkloadKind {
        /// Walk the whole group tree, touching metadata only.
        List,
        /// Open one dataset and take a slice out of the middle of it.
        Slice { dataset: &'static str, count: u64 },
        /// Read one dataset end to end.
        Full { dataset: &'static str },
    }

    #[cfg(all(not(target_arch = "wasm32"), feature = "hdf5-compare"))]
    impl Workload {
        fn open(&self, options: crate::object_store::ReadOptions) -> ObjectStoreFile {
            use object_store::local::LocalFileSystem;
            let path = self.path.canonicalize().unwrap();
            let store = LocalFileSystem::new_with_prefix(path.parent().unwrap()).unwrap();
            let name = path.file_name().unwrap().to_str().unwrap().to_string();
            ObjectStoreFile::with_options(Box::new(store), Path::from(name), options)
        }

        async fn run(&self, file: &ObjectStoreFile) -> H5Result<()> {
            match &self.kind {
                WorkloadKind::List => {
                    crate::list_datasets(file).await?;
                }
                #[allow(clippy::single_range_in_vec_init)]
                WorkloadKind::Slice { dataset, count } => {
                    let ds = crate::open_dataset(file, &[dataset]).await?.unwrap();
                    let mid = ds.shape()[0] / 2;
                    ds.read_range::<u32>(&[mid..mid + count], file).await?;
                }
                WorkloadKind::Full { dataset } => {
                    let ds = crate::open_dataset(file, &[dataset]).await?.unwrap();
                    ds.read_full::<u32>(file).await?;
                }
            }
            Ok(())
        }
    }

    #[cfg(all(not(target_arch = "wasm32"), feature = "hdf5-compare"))]
    fn workloads(dense: &std::path::Path, spread: &std::path::Path) -> Vec<(String, Workload)> {
        let mol = std::path::PathBuf::from(MOL_INFO_FILE);
        let matrix = std::path::PathBuf::from(MATRIX_FILE);
        vec![
            (
                "mol_info: list".to_string(),
                Workload {
                    path: mol.clone(),
                    kind: WorkloadKind::List,
                },
            ),
            (
                "mol_info: slice 100k".to_string(),
                Workload {
                    path: mol.clone(),
                    kind: WorkloadKind::Slice {
                        dataset: "barcode_corrected_reads",
                        count: 100_000,
                    },
                },
            ),
            (
                "mol_info: read_full 34M".to_string(),
                Workload {
                    path: mol,
                    kind: WorkloadKind::Full {
                        dataset: "barcode_corrected_reads",
                    },
                },
            ),
            (
                "matrix: list".to_string(),
                Workload {
                    path: matrix,
                    kind: WorkloadKind::List,
                },
            ),
            (
                "dense 200 datasets: list".to_string(),
                Workload {
                    path: dense.to_path_buf(),
                    kind: WorkloadKind::List,
                },
            ),
            (
                "spread 60 datasets: list".to_string(),
                Workload {
                    path: spread.to_path_buf(),
                    kind: WorkloadKind::List,
                },
            ),
        ]
    }

    // ---- Performance comparison tests (native only) ----

    #[cfg(all(not(target_arch = "wasm32"), feature = "hdf5-compare"))]
    fn median(times: &[std::time::Duration]) -> std::time::Duration {
        let mut sorted: Vec<_> = times.to_vec();
        sorted.sort();
        sorted[sorted.len() / 2]
    }

    #[cfg(all(not(target_arch = "wasm32"), feature = "hdf5-compare"))]
    fn fmt_duration(d: std::time::Duration) -> String {
        let ms = d.as_secs_f64() * 1000.0;
        if ms >= 1000.0 {
            format!("{:.2}s", ms / 1000.0)
        } else {
            format!("{:.1}ms", ms)
        }
    }

    #[cfg(all(not(target_arch = "wasm32"), feature = "hdf5-compare"))]
    struct BenchResult {
        name: String,
        h5rs_times: Vec<std::time::Duration>,
        hdf5_times: Vec<std::time::Duration>,
    }

    #[cfg(all(not(target_arch = "wasm32"), feature = "hdf5-compare"))]
    impl BenchResult {
        fn print(&self) {
            let h = median(&self.h5rs_times);
            let c = median(&self.hdf5_times);
            let ratio = h.as_secs_f64() / c.as_secs_f64();
            let all_h: Vec<_> = self.h5rs_times.iter().map(|t| fmt_duration(*t)).collect();
            let all_c: Vec<_> = self.hdf5_times.iter().map(|t| fmt_duration(*t)).collect();
            println!("  {}", self.name);
            println!("    h5rs: {:>9}  [{}]", fmt_duration(h), all_h.join(", "));
            println!("    hdf5: {:>9}  [{}]", fmt_duration(c), all_c.join(", "));
            println!("    ratio: {ratio:.2}x");
        }
    }

    #[cfg(all(not(target_arch = "wasm32"), feature = "hdf5-compare"))]
    #[tokio::test]
    #[ignore] // Run with: cargo test perf -- --ignored --nocapture
    #[allow(unused)]
    // `&[a..b]` is a one-dimensional selection, not a mis-typed range literal.
    #[allow(clippy::single_range_in_vec_init)]
    async fn perf() -> H5Result<()> {
        require_dataset!(MOL_INFO_FILE);
        use std::time::Instant;

        const ITERS: usize = 5;

        // --- warm page cache ---
        println!("Warming page cache...");
        {
            let buf = std::fs::read(MOL_INFO_FILE).unwrap();
            println!("  {} bytes read", buf.len());
            std::hint::black_box(&buf);
        }

        // --- open files once ---
        let osf = test_file(MOL_INFO_FILE);
        let f = super::File::open(&osf).await?;
        let hf = hdf5::File::open(MOL_INFO_FILE).unwrap();

        // --- pre-open datasets ---
        // gem_group: u8, ~34.7M values, unfiltered
        let gem_obj = f.root_group.find_obj("gem_group", &osf).await?.unwrap();
        let gem_cds = gem_obj
            .header
            .to_dataset("gem_group".to_string(), &osf)
            .await?
            .unwrap();
        let gem_hdf5 = hf.dataset("/gem_group").unwrap();
        let gem_total = gem_cds.shape()[0];

        // barcode_corrected_reads: u32, ~34.7M values, gzip+shuffle
        let bcr_obj = f
            .root_group
            .find_obj("barcode_corrected_reads", &osf)
            .await?
            .unwrap();
        let bcr_cds = bcr_obj
            .header
            .to_dataset("barcode_corrected_reads".to_string(), &osf)
            .await?
            .unwrap();
        let bcr_hdf5 = hf.dataset("/barcode_corrected_reads").unwrap();

        // barcode: u64, ~34.7M values
        let bc_obj = f.root_group.find_obj("barcode", &osf).await?.unwrap();
        let bc_cds = bc_obj
            .header
            .to_dataset("barcode".to_string(), &osf)
            .await?
            .unwrap();
        let bc_hdf5 = hf.dataset("/barcode").unwrap();

        let mut results: Vec<BenchResult> = Vec::new();

        // ========================================================
        // Full array reads
        // ========================================================

        // 1. Full read: gem_group (u8, unfiltered)
        if false {
            let mut h5rs_t = Vec::new();
            let mut hdf5_t = Vec::new();
            for _ in 0..ITERS {
                let t = Instant::now();
                let d = full_read_1d::<u8>(&gem_cds, &osf).await?;
                h5rs_t.push(t.elapsed());
                std::hint::black_box(&d);

                let t = Instant::now();
                let d: Vec<u8> = gem_hdf5.read_raw().unwrap();
                hdf5_t.push(t.elapsed());
                std::hint::black_box(&d);
            }
            results.push(BenchResult {
                name: format!(
                    "Full read gem_group (u8, {}M, {})",
                    gem_total / 1_000_000,
                    if gem_cds.filter.is_some() {
                        "filtered"
                    } else {
                        "no filter"
                    }
                ),
                h5rs_times: h5rs_t,
                hdf5_times: hdf5_t,
            });
        }

        // 2. Full read: barcode_corrected_reads (u32, gzip+shuffle)
        if false {
            let mut h5rs_t = Vec::new();
            let mut hdf5_t = Vec::new();
            for _ in 0..ITERS {
                let t = Instant::now();
                let d = full_read_1d::<u32>(&bcr_cds, &osf).await?;
                h5rs_t.push(t.elapsed());
                std::hint::black_box(&d);

                let t = Instant::now();
                let d: Vec<u32> = bcr_hdf5.read_raw().unwrap();
                hdf5_t.push(t.elapsed());
                std::hint::black_box(&d);
            }
            results.push(BenchResult {
                name: format!(
                    "Full read barcode_corrected_reads (u32, {}M, {})",
                    bcr_cds.shape()[0] / 1_000_000,
                    if bcr_cds.filter.is_some() {
                        "filtered"
                    } else {
                        "no filter"
                    }
                ),
                h5rs_times: h5rs_t,
                hdf5_times: hdf5_t,
            });
        }

        // 3. Full read: barcode (u64, large elements)
        {
            let mut h5rs_t = Vec::new();
            let mut hdf5_t = Vec::new();
            for _ in 0..30 {
                let t = Instant::now();
                let d = full_read_1d::<u64>(&bc_cds, &osf).await?;
                h5rs_t.push(t.elapsed());
                std::hint::black_box(&d);

                let t = Instant::now();
                let d: Vec<u64> = bc_hdf5.read_raw().unwrap();
                hdf5_t.push(t.elapsed());
                std::hint::black_box(&d);
            }
            results.push(BenchResult {
                name: format!(
                    "Full read barcode (u64, {}M, {})",
                    bc_cds.shape()[0] / 1_000_000,
                    if bc_cds.filter.is_some() {
                        "filtered"
                    } else {
                        "no filter"
                    }
                ),
                h5rs_times: h5rs_t,
                hdf5_times: hdf5_t,
            });
        }

        // ========================================================
        // Range reads
        // ========================================================

        // 4. Small range: 1M elements from middle of gem_group
        if false {
            let start = gem_total / 2;
            let end = start + 1_000_000;
            let mut h5rs_t = Vec::new();
            let mut hdf5_t = Vec::new();
            for _ in 0..ITERS {
                let t = Instant::now();
                let d = gem_cds.read_range::<u8>(&[start..end], &osf).await?;
                h5rs_t.push(t.elapsed());
                std::hint::black_box(&d);

                let t = Instant::now();
                let d: ndarray::Array1<u8> = gem_hdf5
                    .read_slice_1d(start as usize..end as usize)
                    .unwrap();
                hdf5_t.push(t.elapsed());
                std::hint::black_box(&d);
            }
            results.push(BenchResult {
                name: "Range read gem_group 1M elements (u8, no filter)".to_string(),
                h5rs_times: h5rs_t,
                hdf5_times: hdf5_t,
            });
        }

        // 5. Large range: 10M elements from gem_group
        if false {
            let start = gem_total / 4;
            let end = start + 10_000_000;
            let mut h5rs_t = Vec::new();
            let mut hdf5_t = Vec::new();
            for _ in 0..ITERS {
                let t = Instant::now();
                let d = gem_cds.read_range::<u8>(&[start..end], &osf).await?;
                h5rs_t.push(t.elapsed());
                std::hint::black_box(&d);

                let t = Instant::now();
                let d: ndarray::Array1<u8> = gem_hdf5
                    .read_slice_1d(start as usize..end as usize)
                    .unwrap();
                hdf5_t.push(t.elapsed());
                std::hint::black_box(&d);
            }
            results.push(BenchResult {
                name: "Range read gem_group 10M elements (u8, no filter)".to_string(),
                h5rs_times: h5rs_t,
                hdf5_times: hdf5_t,
            });
        }

        // 6. Range read: 1M elements from barcode_corrected_reads (filtered)
        if false {
            let start = bcr_cds.shape()[0] / 2;
            let end = start + 1_000_000;
            let mut h5rs_t = Vec::new();
            let mut hdf5_t = Vec::new();
            for _ in 0..ITERS {
                let t = Instant::now();
                let d = bcr_cds.read_range::<u32>(&[start..end], &osf).await?;
                h5rs_t.push(t.elapsed());
                std::hint::black_box(&d);

                let t = Instant::now();
                let d: ndarray::Array1<u32> = bcr_hdf5
                    .read_slice_1d(start as usize..end as usize)
                    .unwrap();
                hdf5_t.push(t.elapsed());
                std::hint::black_box(&d);
            }
            results.push(BenchResult {
                name: "Range read barcode_corrected_reads 1M (u32, gzip+shuffle)".to_string(),
                h5rs_times: h5rs_t,
                hdf5_times: hdf5_t,
            });
        }

        // ========================================================
        // Attribute reads
        // ========================================================

        // Helper macro: read an attribute with correct h5rs type dispatch
        macro_rules! bench_read_attr_h5rs {
            ($attr:expr) => {
                match &$attr.datatype.type_desc {
                    TypeDescriptor::FixedPoint(fp) => match (fp.signed(), fp.size()) {
                        (0, 1) => {
                            std::hint::black_box($attr.read::<u8>());
                        }
                        (0, 2) => {
                            std::hint::black_box($attr.read::<u16>());
                        }
                        (0, 4) => {
                            std::hint::black_box($attr.read::<u32>());
                        }
                        (0, 8) => {
                            std::hint::black_box($attr.read::<u64>());
                        }
                        (1, 1) => {
                            std::hint::black_box($attr.read::<i8>());
                        }
                        (1, 2) => {
                            std::hint::black_box($attr.read::<i16>());
                        }
                        (1, 4) => {
                            std::hint::black_box($attr.read::<i32>());
                        }
                        (1, 8) => {
                            std::hint::black_box($attr.read::<i64>());
                        }
                        _ => {}
                    },
                    TypeDescriptor::FloatingPoint(fp) => match fp.size() {
                        4 => {
                            std::hint::black_box($attr.read::<f32>());
                        }
                        8 => {
                            std::hint::black_box($attr.read::<f64>());
                        }
                        _ => {}
                    },
                    TypeDescriptor::String(_) => {
                        std::hint::black_box($attr.read_strings());
                    }
                    _ => {}
                }
            };
        }

        // Helper macro: read an attribute with correct hdf5 type dispatch
        macro_rules! bench_read_attr_hdf5 {
            ($attr:expr, $ha:expr) => {
                match &$attr.datatype.type_desc {
                    TypeDescriptor::FixedPoint(fp) => match (fp.signed(), fp.size()) {
                        (0, 1) => {
                            std::hint::black_box($ha.read_raw::<u8>().unwrap());
                        }
                        (0, 2) => {
                            std::hint::black_box($ha.read_raw::<u16>().unwrap());
                        }
                        (0, 4) => {
                            std::hint::black_box($ha.read_raw::<u32>().unwrap());
                        }
                        (0, 8) => {
                            std::hint::black_box($ha.read_raw::<u64>().unwrap());
                        }
                        (1, 1) => {
                            std::hint::black_box($ha.read_raw::<i8>().unwrap());
                        }
                        (1, 2) => {
                            std::hint::black_box($ha.read_raw::<i16>().unwrap());
                        }
                        (1, 4) => {
                            std::hint::black_box($ha.read_raw::<i32>().unwrap());
                        }
                        (1, 8) => {
                            std::hint::black_box($ha.read_raw::<i64>().unwrap());
                        }
                        _ => {}
                    },
                    TypeDescriptor::FloatingPoint(fp) => match fp.size() {
                        4 => {
                            std::hint::black_box($ha.read_raw::<f32>().unwrap());
                        }
                        8 => {
                            std::hint::black_box($ha.read_raw::<f64>().unwrap());
                        }
                        _ => {}
                    },
                    TypeDescriptor::String(_) => {
                        // skip strings for hdf5 C lib (FixedAscii size dispatch too complex)
                    }
                    _ => {}
                }
            };
        }

        // 7. Read all numeric/string attributes from root group (100x)
        {
            const ATTR_ITERS: usize = 100;
            let root_attrs: Vec<_> = f.root_group.attributes.iter().collect();
            let mut h5rs_t = Vec::new();
            let mut hdf5_t = Vec::new();
            for _ in 0..ITERS {
                let t = Instant::now();
                for _ in 0..ATTR_ITERS {
                    for attr in &root_attrs {
                        bench_read_attr_h5rs!(attr);
                    }
                }
                h5rs_t.push(t.elapsed());

                let t = Instant::now();
                for _ in 0..ATTR_ITERS {
                    for attr in &root_attrs {
                        let ha = hf.attr(&attr.name()).unwrap();
                        bench_read_attr_hdf5!(attr, ha);
                    }
                }
                hdf5_t.push(t.elapsed());
            }
            results.push(BenchResult {
                name: format!(
                    "Attribute reads, root group ({} attrs x {ATTR_ITERS} iters)",
                    root_attrs.len()
                ),
                h5rs_times: h5rs_t,
                hdf5_times: hdf5_t,
            });
        }

        // 8. Read all attributes from all datasets (100x)
        {
            const ATTR_ITERS: usize = 100;
            let all_ds_attrs: Vec<(&str, &[AttributeMessage])> = vec![
                ("gem_group", &gem_cds.attributes),
                ("barcode_corrected_reads", &bcr_cds.attributes),
                ("barcode", &bc_cds.attributes),
            ];
            let total_attrs: usize = all_ds_attrs.iter().map(|(_, a)| a.len()).sum();
            if total_attrs > 0 {
                let mut h5rs_t = Vec::new();
                let mut hdf5_t = Vec::new();
                for _ in 0..ITERS {
                    let t = Instant::now();
                    for _ in 0..ATTR_ITERS {
                        for (_, attrs) in &all_ds_attrs {
                            for attr in *attrs {
                                bench_read_attr_h5rs!(attr);
                            }
                        }
                    }
                    h5rs_t.push(t.elapsed());

                    let t = Instant::now();
                    for _ in 0..ATTR_ITERS {
                        for (ds_name, attrs) in &all_ds_attrs {
                            let hds = hf.dataset(&format!("/{ds_name}")).unwrap();
                            for attr in *attrs {
                                let ha = hds.attr(&attr.name()).unwrap();
                                bench_read_attr_hdf5!(attr, ha);
                            }
                        }
                    }
                    hdf5_t.push(t.elapsed());
                }
                results.push(BenchResult {
                    name: format!(
                        "Attribute reads, datasets ({total_attrs} attrs x {ATTR_ITERS} iters)"
                    ),
                    h5rs_times: h5rs_t,
                    hdf5_times: hdf5_t,
                });
            }
        }

        // ========================================================
        // Print results
        // ========================================================
        println!("\n{:=<72}", "");
        println!("PERF RESULTS  ({ITERS} iterations, alternating h5rs/hdf5)");
        println!("{:=<72}", "");
        for r in &results {
            r.print();
            println!();
        }

        Ok(())
    }
}

#[cfg(all(test, target_arch = "wasm32"))]
pub(crate) use wasm_bindgen_test::wasm_bindgen_test as async_test;
