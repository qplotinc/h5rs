#![allow(dead_code)]
use std::io::{Cursor, Read, Seek, SeekFrom};

use binrw::{BinRead, BinResult, NullString};

use crate::format::btree::*;

#[derive(BinRead, Debug)]
#[br(magic = b"\x89HDF\r\n\x1a\n")]
pub struct SuperblockV0 {
    pub superblock_version: u8,
    pub free_space_version: u8,
    pub root_group_table_version: u8,

    #[br(align_before = 12)]
    pub shared_header_version: u8,
    pub size_of_offsets: u8,
    pub size_of_lengths: u8,

    #[br(align_before = 16)]
    pub group_leaf_node_k: u16,
    pub group_internal_node_k: u16,
    pub file_consistency_flags: u32,
    // only in v1 of superblock
    //indexed_storage_internal_node_k: u16,
    #[br(align_before = 24)]
    pub base_address: u64,
    pub free_space_info_address: u64,
    pub end_of_file_address: u64,
    pub driver_info_address: u64,
    pub root_group_symbol_table_entry: SymbolTableEntry,
}

#[derive(BinRead, Debug)]
#[br(magic = b"\x89HDF\r\n\x1a\n")]
pub struct SuperblockV2 {
    superblock_version: u8,
    size_of_offsets: u8,
    size_of_lengths: u8,
    file_consistency_flags: u8,
    base_address: u64,
    superblock_extension_address: [u8; 8],
    end_of_file_address: [u8; 8],
    root_group_object_header_address: [u8; 8],
    superblock_checksum: u32,
}

/// v1 BTrees
/// https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_infra_btrees_v1
#[derive(BinRead, Debug, Clone)]
#[br(magic = b"TREE")]
pub struct GroupBTreeV1 {
    #[br(assert(node_type == 0))]
    node_type: u8,
    node_level: u8,
    entries_used: u16,
    left_address: u64,
    right_address: u64,

    #[br(count = entries_used)]
    pub children: Vec<GroupPointerV1>,
}

impl BTree for GroupBTreeV1 {
    type Leaf = GroupPointerV1;

    type Args = ();

    fn children(&self) -> &[Self::Leaf] {
        &self.children
    }

    fn args(&self) -> Self::Args {
        ()
    }

    fn node_level(&self) -> u8 {
        self.node_level
    }
}

#[derive(BinRead, Debug, Clone)]
pub struct GroupPointerV1 {
    pub key: u64,
    pub child_pointer: u64,
}

impl HasPointer for GroupPointerV1 {
    fn child_pointer(&self) -> u64 {
        self.child_pointer
    }
}

#[derive(BinRead, Debug)]
#[br(magic = b"TREE")]
#[br(import(dim: u8))]
pub struct ChunkBTreeV1 {
    #[br(assert(node_type == 1))]
    node_type: u8,
    node_level: u8,
    entries_used: u16,
    left_address: u64,
    right_address: u64,

    #[br(count = entries_used, args { inner: (dim,) })]
    children: Vec<ChunkPointerV1>,

    // last key delimits the upper bound of values in the last child.
    final_key: ChunkKeyV1,
}

impl BTree for ChunkBTreeV1 {
    type Leaf = ChunkPointerV1;

    type Args = (u8,);

    fn children(&self) -> &[Self::Leaf] {
        &self.children
    }

    fn args(&self) -> Self::Args {
        (self
            .children
            .first()
            .map(|c| c.dimensionality())
            .unwrap_or(1),)
    }

    fn node_level(&self) -> u8 {
        self.node_level
    }
}

#[derive(BinRead, Debug, Clone)]
#[br(import(dim: u8))]
pub struct ChunkPointerV1 {
    #[br(args(dim,))]
    pub key: ChunkKeyV1,
    pub child_pointer: u64,
}

impl HasPointer for ChunkPointerV1 {
    fn child_pointer(&self) -> u64 {
        self.child_pointer
    }
}

#[derive(BinRead, Debug, Clone)]
#[br(import(dim: u8))]
pub struct ChunkKeyV1 {
    pub chunk_size: u32,
    pub filter_mask: u32,
    #[br(count = dim + 1)]
    pub offsets: Vec<u64>,
}

impl ChunkPointerV1 {
    pub fn dimensionality(&self) -> u8 {
        (self.key.offsets.len() - 1) as u8
    }
}

/// Group Symbol Table Node
/// https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_infra_symboltable
#[derive(BinRead, Debug)]
#[br(magic = b"SNOD")]
pub struct GroupSymbolTableNode {
    verion: u8,
    reserved: u8,
    number_of_symbols: u16,

    #[br(count = number_of_symbols)]
    pub(crate) entries: Vec<SymbolTableEntry>,
}

/// Symbol Table Entry
/// https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_infra_symboltableentry
#[derive(BinRead, Debug, Clone)]
pub struct SymbolTableEntry {
    pub link_name_offset: u64,
    pub object_header_address: u64,
    #[br(pad_after = 20)]
    pub cache_type: u32,
    // don't need to read these, so pad them out with pad_after=20
    //reserved: u32,
    //scratch_pad: [u8; 16],
}

/// Local Heap
/// https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_infra_localheap
#[derive(BinRead, Debug, Clone)]
#[br(magic = b"HEAP")]
pub struct LocalHeap {
    #[br(pad_after = 3)]
    pub verion: u8,
    pub data_segment_size: u64,
    pub offest_to_head_of_free_list: u64,
    pub data_segment_address: u64,
}

impl LocalHeap {
    pub fn load<R: Read + Seek>(&self, reader: &mut R) -> BinResult<LoadedLocalHeap> {
        reader.seek(SeekFrom::Start(self.data_segment_address))?;
        let mut data = vec![0u8; self.data_segment_size as usize];
        reader.read_exact(&mut data[..])?;

        Ok(LoadedLocalHeap {
            header: self.clone(),
            data,
        })
    }
}

pub struct LoadedLocalHeap {
    header: LocalHeap,
    data: Vec<u8>,
}

impl LoadedLocalHeap {
    pub fn get_string(&self, offset: u64) -> BinResult<String> {
        let mut c = Cursor::new(&self.data);
        c.set_position(offset);
        let s = NullString::read_le(&mut c)?;
        Ok(s.to_string())
    }
}
