#![allow(dead_code)]
use std::{
    cell::RefCell,
    io::{Read, Seek, SeekFrom},
    rc::Rc,
};

use binrw::{BinRead, BinResult};

use crate::format::{
    btree::{BTreeIter, BTreeIter2},
    metadata::{
        ChunkBTreeV1, ChunkPointerV1, GroupBTreeV1, GroupPointerV1, GroupSymbolTableNode,
        LoadedLocalHeap, SuperblockV0, SymbolTableEntry,
    },
    object::{
        DataLayoutChunked, DataLayoutInner, DataLayoutMessage, DataLayoutV3, DataObjectHeader,
        DataspaceMessage, DatatypeMessage, FilterMessage, LayoutInner,
    },
};

pub(crate) mod format;

struct File {
    superblock: SuperblockV0,
    pub root_group: Group,
}

impl File {
    pub fn open<R: Read + Seek>(reader: &mut R) -> BinResult<File> {
        let sb = SuperblockV0::read_le(reader)?;

        reader.seek(SeekFrom::Start(
            sb.root_group_symbol_table_entry.object_header_address,
        ))?;
        let root_group = DataObjectHeader::read_le(reader);
        let mut root_group = root_group.unwrap();
        root_group.load_continuation_messages(reader).unwrap();

        let rg = root_group
            .to_group("/".to_string(), reader)
            .unwrap()
            .unwrap();

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
    pub fn to_group<R: Read + Seek>(&self, r: &mut R) -> BinResult<Option<Group>> {
        self.header.to_group(self.name.clone(), r).transpose()
    }
}

struct Group {
    name: String,
    btree: GroupBTreeV1,
    loaded_local_heap: LoadedLocalHeap,
}

struct GroupObjectIter<'a, R> {
    group: &'a Group,
    reader: RefCell<R>,
    btree_iter: BTreeIter2<GroupBTreeV1, R>,
    current_symbol_table: Option<GroupSymbolTableNode>,
    symbol_table_pos: usize,
}

impl Group {
    fn object_refs<R: Read + Seek>(
        &self,
        reader: &mut R,
    ) -> BinResult<Vec<(String, SymbolTableEntry)>> {
        let it: BinResult<Vec<GroupPointerV1>> =
            BTreeIter::new(reader, self.btree.clone()).collect();

        let ptrs = it?;

        let mut res = vec![];

        for p in ptrs {
            let st = self.load_symbol_table(&p, reader)?;
            for e in &st.entries {
                let name = self.loaded_local_heap.get_string(e.link_name_offset)?;
                let e = e.clone();
                res.push((name, e));
            }
        }

        Ok(res)
    }

    fn load_symbol_table<R: Read + Seek>(
        &self,
        ptr: &GroupPointerV1,
        reader: &mut R,
    ) -> BinResult<GroupSymbolTableNode> {
        reader.seek(SeekFrom::Start(ptr.child_pointer))?;
        GroupSymbolTableNode::read_le(reader)
    }

    fn iter_symbol_table<R: Read + Seek>(
        &self,
        symbol_table: &GroupSymbolTableNode,
        reader: &mut R,
    ) -> impl Iterator<Item = BinResult<Object>> {
        symbol_table
            .entries
            .iter()
            .map(|e| self.load_object(e, reader))
    }

    fn load_object<R: Read + Seek>(
        &self,
        ptr: &SymbolTableEntry,
        reader: &mut R,
    ) -> BinResult<Object> {
        let name = self.loaded_local_heap.get_string(ptr.link_name_offset)?;

        reader.seek(SeekFrom::Start(ptr.object_header_address))?;
        let header = DataObjectHeader::read_le(reader)?;

        Ok(Object { name, header })
    }

    // FIXME: this visits every symbol table entry in the group --
    // we will want more efficient object finding if there are a lot of objects in groups.
    fn find_obj<R: Read + Seek>(
        &self,
        name: impl AsRef<str>,
        reader: &mut R,
    ) -> BinResult<Option<Object>> {
        let r = self.object_refs(reader)?;
        let Some((name, ste)) = r.iter().find(|(n, _)| n == name.as_ref()) else {
            println!("didn't find object: {}", name.as_ref());
            return Ok(None);
        };

        reader.seek(SeekFrom::Start(ste.object_header_address))?;
        let mut header = DataObjectHeader::read_le(reader)?;
        header.load_continuation_messages(reader)?;

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
    // FIXME - support different datatypes
    pub fn read_chunk_simple<R: Read + Seek>(
        &self,
        c: &ChunkPointerV1,
        reader: &mut R,
    ) -> BinResult<Vec<u32>> {
        let n: u32 = self.chunks_layout.dimension_sizes.iter().product();

        let mut result = vec![0u32; n as usize];
        let read_target: &mut [u8] = bytemuck::cast_slice_mut(&mut result);
        // make sure we got the right size to read
        assert_eq!(read_target.len(), c.key.chunk_size as usize);

        reader.seek(SeekFrom::Start(c.child_pointer))?;
        reader.read_exact(read_target)?;

        Ok(result)
    }

    // FIXME - support different datatypes
    pub fn read_chunk_filter<R: Read + Seek>(
        &self,
        c: &ChunkPointerV1,
        reader: &mut R,
    ) -> BinResult<Vec<u32>> {
        let n: u32 = self.chunks_layout.dimension_sizes.iter().product();
        let mut result = vec![0u32; n as usize];

        let read_target: &mut [u8] = bytemuck::cast_slice_mut(&mut result);
        // make sure we got the right size to read
        assert_eq!(read_target.len(), c.key.chunk_size as usize);

        // FIXME - just hardcoding gzip + shuffle right now.
        reader.seek(SeekFrom::Start(c.child_pointer))?;
        let mut gz = flate2::read::GzDecoder::new(reader);

        gz.read_exact(read_target)?;

        // now de-shuffle somehow.

        Ok(result)
    }

    pub fn iter_chunks<R: Read + Seek>(
        &self,
        reader: Rc<RefCell<R>>,
    ) -> BinResult<impl Iterator<Item = BinResult<ChunkPointerV1>>> {
        reader
            .borrow_mut()
            .seek(SeekFrom::Start(self.chunks_layout.address))?;
        let btree = ChunkBTreeV1::read_le_args(
            &mut *reader.borrow_mut(),
            (self.dataspace.dimensionality,),
        )?;

        Ok(BTreeIter2::new(reader, btree))
    }
}

#[cfg(test)]
mod test {

    use std::{cell::RefCell, rc::Rc};

    use binrw::BinResult;
    use hdf5::Result;
    use ndarray::s;

    const MOL_INFO_FILE: &str = "datasets/frozen_pbmc_donor_c_molecule_info.h5";
    const MATRIX_FILE: &str = "datasets/gene_bc_matrix.h5";

    fn load_ds() -> Result<()> {
        let f = hdf5::File::open(MOL_INFO_FILE).unwrap();
        let ds = f.dataset("/umi").unwrap();

        let v = ds.read_slice_1d::<u32, _>(s![..16])?;

        println!("{v:?}");
        Ok(())
    }

    #[test]
    fn cmp1() -> BinResult<()> {
        load_ds().unwrap();

        let mut rdr = std::io::BufReader::new(std::fs::File::open(MOL_INFO_FILE).unwrap());
        let f = super::File::open(&mut rdr)?;

        let obj = f.root_group.find_obj("umi", &mut rdr)?;

        let o = obj.unwrap();
        let ds = o.header.to_dataset("adf".to_string()).unwrap();

        let cds = ds.chunked().unwrap();

        let rc = Rc::new(RefCell::new(rdr));
        let chunk = cds
            .iter_chunks(rc.clone())
            .unwrap()
            .next()
            .unwrap()
            .unwrap();

        let chunk = cds
            .read_chunk_simple(&chunk, &mut *rc.borrow_mut())
            .unwrap();
        println!("my chunk: {:?}", &chunk[..16]);

        Ok(())
    }

    #[test]
    fn filters() -> BinResult<()> {
        let mut rdr = std::io::BufReader::new(std::fs::File::open(MATRIX_FILE).unwrap());
        let f = super::File::open(&mut rdr)?;

        let obj = f.root_group.find_obj("matrix", &mut rdr)?;
        let o = obj.unwrap();
        let g = o.to_group(&mut rdr)?.unwrap();

        let data = g.find_obj("data", &mut rdr)?.unwrap();

        println!("data matrix: {:#?}", data.header);
        Ok(())
    }
}
