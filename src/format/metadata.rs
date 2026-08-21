use std::io::Cursor;

use binrw::{BinRead, BinResult, NullString};

use crate::error::{H5Error, H5Result};
use crate::format::btree::*;
use crate::object_store::{ObjectStoreFile, fetch_exact};

#[derive(BinRead, Debug)]
#[br(magic = b"\x89HDF\r\n\x1a\n")]
#[allow(dead_code)]
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

/// Superblock versions 2 and 3. Version 3 differs only in that the consistency
/// flags byte is meaningful (file locking / SWMR), which does not affect reads.
///
/// <https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_boot_super>
#[derive(BinRead, Debug)]
#[br(magic = b"\x89HDF\r\n\x1a\n")]
#[allow(dead_code)]
pub struct SuperblockV23 {
    pub superblock_version: u8,
    pub size_of_offsets: u8,
    pub size_of_lengths: u8,
    pub file_consistency_flags: u8,
    pub base_address: u64,
    pub superblock_extension_address: u64,
    pub end_of_file_address: u64,
    pub root_group_object_header_address: u64,
    pub superblock_checksum: u32,
}

/// The parts of the superblock h5rs needs, normalised across every version.
#[derive(Debug, Clone)]
pub struct Superblock {
    /// Superblock version: 0, 1, 2 or 3. Kept for diagnostics.
    #[allow(dead_code)]
    pub version: u8,
    /// Address of the root group's object header.
    pub root_group_address: u64,
}

impl Superblock {
    /// Read and normalise the superblock at the start of the file.
    ///
    /// h5rs requires 8-byte offsets and lengths (the HDF5 default) and a zero
    /// base address; anything else is reported as unsupported rather than
    /// silently mis-parsed.
    pub async fn read(file: &ObjectStoreFile) -> H5Result<Superblock> {
        let bytes = fetch_exact(file, 0, SUPERBLOCK_FETCH_SIZE).await?;
        if bytes.len() < 9 {
            return Err(H5Error::corrupt("file is too short to hold a superblock"));
        }
        let version = bytes[8];

        let (sizes, base_address, root_group_address) = match version {
            0 | 1 => {
                let sb: SuperblockV0 = read_le_from(&bytes)?;
                (
                    (sb.size_of_offsets, sb.size_of_lengths),
                    sb.base_address,
                    sb.root_group_symbol_table_entry.object_header_address,
                )
            }
            2 | 3 => {
                let sb: SuperblockV23 = read_le_from(&bytes)?;
                (
                    (sb.size_of_offsets, sb.size_of_lengths),
                    sb.base_address,
                    sb.root_group_object_header_address,
                )
            }
            v => {
                return Err(H5Error::unsupported(format!("superblock version {v}")));
            }
        };

        if sizes != (8, 8) {
            return Err(H5Error::unsupported(format!(
                "{}-byte offsets / {}-byte lengths (only 8/8 is implemented)",
                sizes.0, sizes.1
            )));
        }
        if base_address != 0 {
            return Err(H5Error::unsupported(
                "non-zero superblock base address (user block)",
            ));
        }

        Ok(Superblock {
            version,
            root_group_address,
        })
    }
}

/// The largest superblock (version 1, with 8-byte addresses) fits well within this.
const SUPERBLOCK_FETCH_SIZE: u64 = 256;

/// Parse a little-endian `binrw` structure from an in-memory buffer.
fn read_le_from<T: for<'a> BinRead<Args<'a> = ()>>(bytes: &[u8]) -> H5Result<T> {
    let mut cursor = Cursor::new(bytes);
    Ok(T::read_le(&mut cursor)?)
}

/// v1 BTrees
/// https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_infra_btrees_v1
#[derive(BinRead, Debug, Clone)]
#[br(magic = b"TREE")]
#[allow(dead_code)]
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

    fn args(&self) -> Self::Args {}

    fn node_level(&self) -> u8 {
        self.node_level
    }
}

#[derive(BinRead, Debug, Clone)]
#[allow(dead_code)]
pub struct GroupPointerV1 {
    pub key: u64,
    pub child_pointer: u64,
}

impl HasPointer for GroupPointerV1 {
    fn child_pointer(&self) -> u64 {
        self.child_pointer
    }
}

#[derive(BinRead, Debug, Clone)]
#[br(magic = b"TREE")]
#[br(import(dim: u8))]
#[allow(dead_code)]
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
    /// Bitmask of filters skipped for this chunk. Parsed for completeness;
    /// h5rs does not yet honour per-chunk filter skipping.
    #[allow(dead_code)]
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
#[allow(dead_code)]
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
#[allow(dead_code)]
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
#[allow(dead_code)]
pub struct LocalHeap {
    #[br(pad_after = 3)]
    pub verion: u8,
    pub data_segment_size: u64,
    pub offest_to_head_of_free_list: u64,
    pub data_segment_address: u64,
}

impl LocalHeap {
    pub async fn load(&self, file: &ObjectStoreFile) -> H5Result<LoadedLocalHeap> {
        let data = fetch_exact(file, self.data_segment_address, self.data_segment_size).await?;
        Ok(LoadedLocalHeap {
            data: data.to_vec(),
        })
    }
}

pub struct LoadedLocalHeap {
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
