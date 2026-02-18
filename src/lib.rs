#![allow(dead_code)]

use crate::error::H5Result;
use crate::format::{
    btree::{collect_btree_leaves, collect_btree_leaves_args},
    metadata::{
        ChunkBTreeV1, ChunkPointerV1, GroupBTreeV1, GroupPointerV1, GroupSymbolTableNode,
        LoadedLocalHeap, SuperblockV0, SymbolTableEntry,
    },
    object::{
        AttributeMessage, DataLayoutChunked, DataLayoutInner, DataLayoutMessage, DataLayoutV3,
        DataObjectHeader, DataspaceMessage, DatatypeMessage, FilterMessage, FilterType,
        LayoutInner,
    },
};
use crate::h5type::H5Type;
use crate::object_store::{ObjectStoreFile, fetch_exact, read_metadata, read_metadata_args};

pub mod error;
pub(crate) mod format;
pub mod h5type;
pub(crate) mod object_store;

struct File {
    superblock: SuperblockV0,
    pub root_group: Group,
}

impl File {
    pub async fn open(file: &ObjectStoreFile) -> H5Result<File> {
        let sb: SuperblockV0 = read_metadata(file, 0).await?;

        let mut root_group: DataObjectHeader =
            read_metadata(file, sb.root_group_symbol_table_entry.object_header_address).await?;
        root_group.load_continuation_messages(file).await?;

        let rg = root_group.to_group("/".to_string(), file).await.unwrap()?;

        Ok(File {
            superblock: sb,
            root_group: rg,
        })
    }
}

struct Object {
    name: String,
    header: DataObjectHeader,
}

impl Object {
    pub async fn to_group(&self, file: &ObjectStoreFile) -> H5Result<Option<Group>> {
        match self.header.to_group(self.name.clone(), file).await {
            Some(r) => Ok(Some(r?)),
            None => Ok(None),
        }
    }
}

struct Group {
    name: String,
    btree: GroupBTreeV1,
    loaded_local_heap: LoadedLocalHeap,
    pub attributes: Vec<AttributeMessage>,
}

impl Group {
    async fn object_refs(
        &self,
        file: &ObjectStoreFile,
    ) -> H5Result<Vec<(String, SymbolTableEntry)>> {
        let ptrs: Vec<GroupPointerV1> = collect_btree_leaves(file, self.btree.clone()).await?;

        let mut res = vec![];

        for p in ptrs {
            let st = self.load_symbol_table(&p, file).await?;
            for e in &st.entries {
                let name = self.loaded_local_heap.get_string(e.link_name_offset)?;
                let e = e.clone();
                res.push((name, e));
            }
        }

        Ok(res)
    }

    async fn load_symbol_table(
        &self,
        ptr: &GroupPointerV1,
        file: &ObjectStoreFile,
    ) -> H5Result<GroupSymbolTableNode> {
        read_metadata(file, ptr.child_pointer).await
    }

    async fn load_object(
        &self,
        ptr: &SymbolTableEntry,
        file: &ObjectStoreFile,
    ) -> H5Result<Object> {
        let name = self.loaded_local_heap.get_string(ptr.link_name_offset)?;
        let header: DataObjectHeader = read_metadata(file, ptr.object_header_address).await?;

        Ok(Object { name, header })
    }

    async fn find_obj(
        &self,
        name: impl AsRef<str>,
        file: &ObjectStoreFile,
    ) -> H5Result<Option<Object>> {
        let r = self.object_refs(file).await?;
        let Some((name, ste)) = r.iter().find(|(n, _)| n == name.as_ref()) else {
            println!("didn't find object: {}", name.as_ref());
            return Ok(None);
        };

        let mut header: DataObjectHeader = read_metadata(file, ste.object_header_address).await?;
        header.load_continuation_messages(file).await?;

        Ok(Some(Object {
            name: name.clone(),
            header,
        }))
    }
}

struct Dataset {
    name: String,
    dataspace: DataspaceMessage,
    datatype: DatatypeMessage,
    layout: DataLayoutMessage,
    filter: Option<FilterMessage>,
    pub attributes: Vec<AttributeMessage>,
}

impl Dataset {
    pub fn chunked(&self) -> Option<ChunkedDataset> {
        let DataLayoutMessage {
            inner:
                DataLayoutInner::V3(DataLayoutV3 {
                    layout_inner: LayoutInner::Chunked(chunk),
                    ..
                }),
            ..
        } = self.layout.clone()
        else {
            return None;
        };

        Some(ChunkedDataset {
            name: self.name.clone(),
            dataspace: self.dataspace.clone(),
            datatype: self.datatype.clone(),
            layout: self.layout.clone(),
            filter: self.filter.clone(),
            chunks_layout: chunk,
        })
    }
}

struct ChunkedDataset {
    name: String,
    dataspace: DataspaceMessage,
    datatype: DatatypeMessage,
    layout: DataLayoutMessage,
    chunks_layout: DataLayoutChunked,
    filter: Option<FilterMessage>,
}

impl ChunkedDataset {
    pub async fn read_chunk_simple<T: H5Type>(
        &self,
        c: &ChunkPointerV1,
        file: &ObjectStoreFile,
    ) -> H5Result<Vec<T>> {
        T::check_dtype(&self.datatype);

        let byte_count: usize = self
            .chunks_layout
            .dimension_sizes
            .iter()
            .map(|&d| d as usize)
            .product();

        assert_eq!(byte_count, c.key.chunk_size as usize);

        let bytes = fetch_exact(file, c.child_pointer, byte_count as u64).await?;

        let elem_size = std::mem::size_of::<T>();
        let mut result = vec![T::zeroed(); byte_count / elem_size];
        let read_target: &mut [u8] = bytemuck::cast_slice_mut(&mut result);
        read_target.copy_from_slice(&bytes);

        Ok(result)
    }

    pub async fn read_chunk_filter<T: H5Type>(
        &self,
        c: &ChunkPointerV1,
        file: &ObjectStoreFile,
    ) -> H5Result<Vec<T>> {
        use std::io::Read;

        T::check_dtype(&self.datatype);

        let uncompressed_bytes: usize = self
            .chunks_layout
            .dimension_sizes
            .iter()
            .map(|&d| d as usize)
            .product();

        let compressed = fetch_exact(file, c.child_pointer, c.key.chunk_size as u64).await?;

        // Inflate
        let mut data = Vec::with_capacity(uncompressed_bytes);
        let mut decoder = flate2::read::ZlibDecoder::new(&compressed[..]);
        decoder
            .read_to_end(&mut data)
            .map_err(|e| binrw::Error::Custom {
                pos: c.child_pointer,
                err: Box::new(e),
            })?;

        assert_eq!(data.len(), uncompressed_bytes);

        // Un-shuffle
        if self.filter.as_ref().is_some_and(|f| {
            f.filters
                .iter()
                .any(|fd| matches!(fd.filter_type, FilterType::Shuffle))
        }) {
            let element_size = *self.chunks_layout.dimension_sizes.last().unwrap() as usize;
            let num_elements = uncompressed_bytes / element_size;
            let mut unshuffled = vec![0u8; uncompressed_bytes];

            for i in 0..num_elements {
                for b in 0..element_size {
                    unshuffled[i * element_size + b] = data[b * num_elements + i];
                }
            }
            data = unshuffled;
        }

        let result: Vec<T> = bytemuck::cast_slice(&data).to_vec();
        Ok(result)
    }

    pub async fn collect_chunks(&self, file: &ObjectStoreFile) -> H5Result<Vec<ChunkPointerV1>> {
        let btree: ChunkBTreeV1 = read_metadata_args(
            file,
            self.chunks_layout.address,
            (self.dataspace.dimensionality,),
        )
        .await?;

        collect_btree_leaves_args(file, btree).await
    }
}

#[cfg(test)]
mod test {
    use std::fmt::Debug;

    use object_store::{local::LocalFileSystem, path::Path};

    use crate::error::H5Result;
    use crate::format::object::{AttributeMessage, DataObjectHeader, TypeDescriptor};
    use crate::h5type::H5Type;
    use crate::object_store::{ObjectStoreFile, read_metadata};

    const MOL_INFO_FILE: &str = "datasets/frozen_pbmc_donor_c_molecule_info.h5";
    const MATRIX_FILE: &str = "datasets/gene_bc_matrix.h5";

    fn test_file(path: &str) -> ObjectStoreFile {
        let cwd = std::env::current_dir().unwrap();
        let store = LocalFileSystem::new_with_prefix(&cwd).unwrap();
        ObjectStoreFile::new(Box::new(store), Path::from(path))
    }

    /// Read all chunks of a dataset via h5rs and compare byte-for-byte
    /// against the hdf5 C library (gold standard).
    async fn compare_typed<T>(
        cds: &super::ChunkedDataset,
        hdf5_ds: &hdf5::Dataset,
        path: &str,
        file: &ObjectStoreFile,
    ) -> H5Result<()>
    where
        T: H5Type + hdf5::H5Type + PartialEq + Debug,
    {
        let expected: Vec<T> = hdf5_ds.read_raw::<T>().unwrap();
        let chunks = cds.collect_chunks(file).await?;

        let has_filters = cds.filter.is_some();
        let mut all_data: Vec<T> = Vec::new();
        for chunk in &chunks {
            let data: Vec<T> = if has_filters {
                cds.read_chunk_filter(chunk, file).await?
            } else {
                cds.read_chunk_simple(chunk, file).await?
            };
            all_data.extend_from_slice(&data);
        }

        // Last chunk may have padding beyond the actual dataset size
        all_data.truncate(expected.len());

        assert_eq!(all_data.len(), expected.len(), "{path}: length mismatch");
        assert_eq!(&all_data[..], &expected[..], "{path}: data mismatch");
        println!(
            "  OK {path} ({} values, {} chunks)",
            expected.len(),
            chunks.len()
        );

        Ok(())
    }

    /// Dispatch to the correct typed comparison based on the HDF5 datatype.
    async fn compare_dataset(
        cds: &super::ChunkedDataset,
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
    async fn collect_from_group(
        group: &super::Group,
        path: &str,
        file: &ObjectStoreFile,
        hf: &hdf5::File,
        datasets: &mut Vec<(String, super::ChunkedDataset)>,
        group_stack: &mut Vec<(super::Group, String)>,
    ) -> H5Result<()> {
        let refs = group.object_refs(file).await?;
        for (name, ste) in &refs {
            let child_path = if path == "/" {
                format!("/{name}")
            } else {
                format!("{path}/{name}")
            };

            let mut header: DataObjectHeader =
                read_metadata(file, ste.object_header_address).await?;
            header.load_continuation_messages(file).await?;

            if let Some(result) = header.to_group(name.clone(), file).await {
                let g = result?;
                let hdf5_group = hf.group(&child_path).unwrap();
                compare_attrs(&g.attributes, &hdf5_group, &child_path);
                group_stack.push((g, child_path));
            } else if let Some(ds) = header.to_dataset(name.clone()) {
                let hdf5_ds = hf.dataset(&child_path).unwrap();
                compare_attrs(&ds.attributes, &hdf5_ds, &child_path);
                if let Some(cds) = ds.chunked() {
                    datasets.push((child_path, cds));
                } else {
                    println!("  SKIP {child_path}: non-chunked layout");
                }
            }
        }
        Ok(())
    }

    fn compare_attr_typed<T>(attr: &AttributeMessage, hdf5_attr: &hdf5::Attribute, path: &str)
    where
        T: H5Type + hdf5::H5Type + PartialEq + Debug,
    {
        let ours: Vec<T> = attr.read::<T>();
        let expected: Vec<T> = hdf5_attr.read_raw::<T>().unwrap();
        assert_eq!(ours, expected, "{path}@{}: data mismatch", attr.name());
        println!("  OK {path}@{} ({} values)", attr.name(), ours.len());
    }

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
            other => {
                println!(
                    "  SKIP {path}@{}: unsupported attr type {other:?}",
                    attr.name()
                );
            }
        }
    }

    fn compare_attrs(attrs: &[AttributeMessage], hdf5_loc: &hdf5::Location, path: &str) {
        for attr in attrs {
            let name = attr.name();
            let hdf5_attr = hdf5_loc.attr(&name).unwrap();
            compare_attr_untyped(attr, &hdf5_attr, path);
        }
    }

    /// Open an HDF5 file, walk all groups, and compare every chunked
    /// dataset against the hdf5 C library.
    async fn compare_file(path: &str) -> H5Result<()> {
        let file = test_file(path);
        let f = super::File::open(&file).await?;
        let hf = hdf5::File::open(path).unwrap();

        // Compare root group attributes
        compare_attrs(&f.root_group.attributes, &hf, "/");

        let mut datasets: Vec<(String, super::ChunkedDataset)> = vec![];
        let mut group_stack: Vec<(super::Group, String)> = vec![];

        collect_from_group(&f.root_group, "/", &file, &hf, &mut datasets, &mut group_stack)
            .await?;
        while let Some((group, gpath)) = group_stack.pop() {
            collect_from_group(&group, &gpath, &file, &hf, &mut datasets, &mut group_stack)
                .await?;
        }

        println!("{path}: found {} chunked datasets", datasets.len());

        for (ds_path, cds) in &datasets {
            let hdf5_ds = hf.dataset(ds_path).unwrap();
            compare_dataset(cds, &hdf5_ds, ds_path, &file).await?;
        }

        Ok(())
    }

    #[tokio::test]
    async fn mol_info_file() -> H5Result<()> {
        compare_file(MOL_INFO_FILE).await
    }

    #[tokio::test]
    async fn matrix_file() -> H5Result<()> {
        compare_file(MATRIX_FILE).await
    }
}
