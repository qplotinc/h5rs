#![allow(dead_code)]

use crate::error::H5Result;
use crate::format::{
    btree::{collect_btree_leaves, collect_btree_leaves_args},
    metadata::{
        ChunkBTreeV1, ChunkPointerV1, GroupBTreeV1, GroupPointerV1, GroupSymbolTableNode,
        LoadedLocalHeap, SuperblockV0, SymbolTableEntry,
    },
    object::{
        DataLayoutChunked, DataLayoutInner, DataLayoutMessage, DataLayoutV3, DataObjectHeader,
        DataspaceMessage, DatatypeMessage, FilterMessage, FilterType, LayoutInner,
    },
};
use crate::object_store::{ObjectStoreFile, fetch_exact, read_metadata, read_metadata_args};

pub mod error;
pub(crate) mod format;
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
    pub async fn read_chunk_simple(
        &self,
        c: &ChunkPointerV1,
        file: &ObjectStoreFile,
    ) -> H5Result<Vec<u32>> {
        let byte_count: usize = self
            .chunks_layout
            .dimension_sizes
            .iter()
            .map(|&d| d as usize)
            .product();

        assert_eq!(byte_count, c.key.chunk_size as usize);

        let bytes = fetch_exact(file, c.child_pointer, byte_count as u64).await?;

        let mut result = vec![0u32; byte_count / 4];
        let read_target: &mut [u8] = bytemuck::cast_slice_mut(&mut result);
        read_target.copy_from_slice(&bytes);

        Ok(result)
    }

    pub async fn read_chunk_filter(
        &self,
        c: &ChunkPointerV1,
        file: &ObjectStoreFile,
    ) -> H5Result<Vec<u32>> {
        use std::io::Read;

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

        let result: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
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

    use hdf5::Result;
    use ndarray::s;
    use object_store::{local::LocalFileSystem, path::Path};

    use crate::error::H5Result;
    use crate::object_store::ObjectStoreFile;

    const MOL_INFO_FILE: &str = "datasets/frozen_pbmc_donor_c_molecule_info.h5";
    const MATRIX_FILE: &str = "datasets/gene_bc_matrix.h5";

    fn test_file(path: &str) -> ObjectStoreFile {
        let cwd = std::env::current_dir().unwrap();
        let store = LocalFileSystem::new_with_prefix(&cwd).unwrap();
        ObjectStoreFile::new(Box::new(store), Path::from(path))
    }

    #[tokio::test]
    async fn cmp1() -> H5Result<()> {
        // Read first chunk via hdf5 crate (ground truth)
        let hf = hdf5::File::open(MOL_INFO_FILE).unwrap();
        let hds = hf.dataset("/umi").unwrap();
        let expected = hds.read_slice_1d::<u32, _>(s![..16384]).unwrap();

        // Read first chunk via h5rs
        let file = test_file(MOL_INFO_FILE);
        let f = super::File::open(&file).await?;

        let obj = f.root_group.find_obj("umi", &file).await?;
        let o = obj.unwrap();
        let ds = o.header.to_dataset("umi".to_string()).unwrap();
        let cds = ds.chunked().unwrap();

        let chunks = cds.collect_chunks(&file).await?;
        let chunk_ptr = &chunks[0];

        let chunk = cds.read_chunk_simple(chunk_ptr, &file).await?;

        assert_eq!(chunk.len(), expected.len());
        assert_eq!(&chunk[..], expected.as_slice().unwrap());
        println!(
            "SUCCESS: first chunk of umi ({} values) matches hdf5 reference",
            chunk.len()
        );

        Ok(())
    }

    #[tokio::test]
    async fn filters() -> H5Result<()> {
        let file = test_file(MATRIX_FILE);
        let f = super::File::open(&file).await?;

        let obj = f.root_group.find_obj("matrix", &file).await?;
        let o = obj.unwrap();
        let g = o.to_group(&file).await?.unwrap();

        let data = g.find_obj("data", &file).await?.unwrap();

        println!("data matrix: {:#?}", data.header);
        Ok(())
    }

    #[tokio::test]
    async fn read_filtered_data() -> H5Result<()> {
        // Read via hdf5 crate (ground truth)
        let hf = hdf5::File::open(MATRIX_FILE).unwrap();
        let hds = hf.dataset("matrix/data").unwrap();
        let expected = hds.read_raw::<i32>().unwrap();

        // Read via h5rs
        let file = test_file(MATRIX_FILE);
        let f = super::File::open(&file).await?;

        let matrix_obj = f.root_group.find_obj("matrix", &file).await?.unwrap();
        let matrix_group = matrix_obj.to_group(&file).await?.unwrap();
        let data_obj = matrix_group.find_obj("data", &file).await?.unwrap();

        let ds = data_obj.header.to_dataset("data".to_string()).unwrap();
        println!("filter: {:#?}", ds.filter);
        println!("dataspace: {:#?}", ds.dataspace);
        println!("datatype: {:#?}", ds.datatype);

        let cds = ds.chunked().unwrap();

        let chunks = cds.collect_chunks(&file).await?;

        println!("num chunks: {}", chunks.len());

        let mut all_data: Vec<i32> = Vec::new();
        for chunk in &chunks {
            let data = cds.read_chunk_filter(chunk, &file).await?;
            let data_i32: Vec<i32> = data.iter().map(|&v| v as i32).collect();
            all_data.extend_from_slice(&data_i32);
        }

        all_data.truncate(expected.len());

        assert_eq!(all_data.len(), expected.len(), "length mismatch");
        assert_eq!(&all_data[..], &expected[..], "data mismatch");

        println!(
            "SUCCESS: read {} values matching hdf5 reference",
            all_data.len()
        );
        Ok(())
    }
}
