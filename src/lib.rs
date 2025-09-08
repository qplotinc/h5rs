#![allow(dead_code)]
use modular_bitfield::prelude::*;
use std::io::{Read, Seek, SeekFrom};

use binrw::{BinRead, BinResult, NullString};

#[derive(BinRead, Debug)]
#[br(magic = b"\x89HDF\r\n\x1a\n")]
struct SuperblockV0 {
    superblock_version: u8,
    free_space_version: u8,
    root_group_table_version: u8,

    #[br(align_before = 12)]
    shared_header_version: u8,
    size_of_offsets: u8,
    size_of_lengths: u8,

    #[br(align_before = 16)]
    group_leaf_node_k: u16,
    group_internal_node_k: u16,
    file_consistency_flags: u32,
    // only in v1 of superblock
    //indexed_storage_internal_node_k: u16,
    #[br(align_before = 24)]
    base_address: u64,
    free_space_info_address: u64,
    end_of_file_address: u64,
    driver_info_address: u64,
    root_group_symbol_table_entry: u32,
}

#[derive(BinRead, Debug)]
#[br(magic = b"\x89HDF\r\n\x1a\n")]
struct SuperblockV2 {
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
#[derive(BinRead, Debug)]
#[br(magic = b"TREE")]
struct GroupBTreeV1 {
    #[br(assert(node_type == 0))]
    node_type: u8,
    node_level: u8,
    entries_used: u16,
    left_address: u64,
    right_address: u64,

    #[br(count = entries_used)]
    children: Vec<GroupPointerV1>,
}

#[derive(BinRead, Debug)]
struct GroupPointerV1 {
    key: u64,
    child_pointer: u64,
}

#[derive(BinRead, Debug)]
#[br(magic = b"TREE")]
#[br(import(dim: u8))]
struct ChunkBTreeV1 {
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

#[derive(BinRead, Debug, Clone)]
#[br(import(dim: u8))]
struct ChunkKeyV1 {
    chunk_size: u32,
    filter_mask: u32,
    #[br(count = dim + 1)]
    offsets: Vec<u64>,
}

#[derive(BinRead, Debug, Clone)]
#[br(import(dim: u8))]
struct ChunkPointerV1 {
    #[br(args(dim,))]
    key: ChunkKeyV1,
    child_pointer: u64,
}

impl ChunkPointerV1 {
    pub fn dimensionality(&self) -> u8 {
        (self.key.offsets.len() - 1) as u8
    }
}

#[derive(Debug)]
struct BTreeIter<'a, T> {
    reader: &'a mut T,
    stack: Vec<(ChunkBTreeV1, usize)>,
    dimensionality: u8,
}

impl<'a, T> BTreeIter<'a, T> {
    pub fn new(reader: &'a mut T, tree: ChunkBTreeV1) -> BTreeIter<'a, T> {
        let dimensionality = tree
            .children
            .first()
            .map(|x| x.dimensionality())
            .unwrap_or(1);

        BTreeIter {
            reader,
            stack: vec![(tree, 0)],
            dimensionality,
        }
    }
}

enum IterState {
    Done,
    NodeDone,
    InnerNode,
    LeafNode,
}
impl<'a, T: Read + Seek> Iterator for BTreeIter<'a, T> {
    type Item = BinResult<ChunkPointerV1>;

    fn next(&mut self) -> Option<Self::Item> {
        use IterState::*;

        loop {
            let state = match self.stack.last() {
                None => IterState::Done,
                Some((node, pos)) if node.children.len() == *pos => NodeDone,
                Some((node, _)) if node.node_level > 0 => InnerNode,
                Some((_, _)) => LeafNode,
            };

            match state {
                IterState::Done => return None,
                IterState::NodeDone => {
                    let _ = self.stack.pop();
                }
                IterState::InnerNode => {
                    // Need to expand down a level.
                    let child_node = {
                        let Some((node, pos)) = self.stack.last_mut() else {
                            unreachable!();
                        };
                        self.reader
                            .seek(SeekFrom::Start(node.children[*pos].child_pointer))
                            .ok()?;
                        *pos += 1;
                        ChunkBTreeV1::read_le_args(self.reader, (self.dimensionality,)).ok()?
                    };

                    self.stack.push((child_node, 0));
                }
                IterState::LeafNode => {
                    let Some((node, pos)) = self.stack.last_mut() else {
                        unreachable!();
                    };
                    let leaf = node.children[*pos].clone();
                    *pos += 1;
                    return Some(Ok(leaf));
                }
            };
        }
    }
}

/// Group Symbol Table Node
/// https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_infra_symboltable
#[derive(BinRead, Debug)]
#[br(magic = b"SNOD")]
struct GroupSymbolTableNode {
    verion: u8,
    reserved: u8,
    number_of_symbols: u16,

    #[br(count = number_of_symbols)]
    entries: Vec<SymbolTableEntry>,
}

/// Symbol Table Entry
/// https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_infra_symboltableentry
#[derive(BinRead, Debug)]
struct SymbolTableEntry {
    link_name_offset: u64,
    object_header_address: u64,
    #[br(pad_after = 20)]
    cache_type: u32,
    // don't need to read these, so pad them out with pad_after=20
    //reserved: u32,
    //scratch_pad: [u8; 16],
}

/// Local Heap
/// https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_infra_localheap
#[derive(BinRead, Debug)]
#[br(magic = b"HEAP")]
struct LocalHeap {
    #[br(pad_after = 3)]
    verion: u8,
    data_segment_size: u64,
    offest_to_head_of_free_list: u64,
    data_segment_address: u64,
}

/// Object Header
/// https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#sec_fmt4_dataobject
#[derive(BinRead, Debug)]
struct DataObjectHeader {
    version: u8,
    #[br(pad_before = 1)]
    total_header_messages: u16,
    object_reference_count: u32,
    #[br(pad_after = 4)]
    object_header_size: u32,

    #[br(parse_with = parse_header_list, args(object_header_size))]
    messages: Vec<HeaderMessage>,
}

/// Need a custom parser for the Header Message list: we
/// don't know how many messages will be in this block,
/// we only know the total size of the header. So this method
/// tracks the total size and stops when the correct size when the
/// size has been reached.
#[binrw::parser(reader, endian)]
fn parse_header_list(object_header_size: u32) -> BinResult<Vec<HeaderMessage>> {
    // header size does not include reserved space after object_header_size
    let mut total_size = 0;
    let mut headers = vec![];
    while total_size < object_header_size {
        let h = HeaderMessage::read_options(reader, endian, ())?;
        // each message has 8 bytes of header, plus the data_size of the message
        total_size += 8 + h.data_size as u32;
        headers.push(h);
    }

    Ok(headers)
}

#[derive(BinRead, Debug)]
struct HeaderMessage {
    message_type: u16,
    data_size: u16,
    #[br(pad_after = 3)]
    flags: u8,
    #[br(args{ ty: message_type, data_size})]
    #[br(pad_size_to = data_size)]
    inner: InnerMessage,
}

#[derive(BinRead, Debug)]
#[br(import { ty: u16, data_size: u16 })]
enum InnerMessage {
    #[br(pre_assert(ty == 0))]
    Nil(NilMessage),
    #[br(pre_assert(ty == 1))]
    Dataspace(DataspaceMessage),
    #[br(pre_assert(ty == 3))]
    Datatype(DatatypeMessage),
    #[br(pre_assert(ty == 5))]
    FillValue(FillValueMessage),
    #[br(pre_assert(ty == 8))]
    DataLayout(DataLaymoutMessage),
    #[br(pre_assert(ty == 12))]
    Attribute(AttributeMessage),
    #[br(pre_assert(ty == 16))]
    ObjectHeaderContinuation(ObjectHeaderContinuationMessage),
    #[br(pre_assert(ty == 18))]
    ModificationTie(ModificationTimMessage),
    #[br(pre_assert(ty != 12 && ty != 3 && ty != 8))]
    Unknown(#[br(args{data_size})] UnknownMessage),
}

#[derive(BinRead, Debug)]
struct NilMessage {}

#[derive(BinRead, Debug)]
struct DataspaceMessage {
    #[br(assert(version == 1))]
    version: u8,
    dimensionality: u8,
    #[br(pad_after = 5)]
    flags: u8,
    //reserved1: u8,
    //reserved2: u32,
    #[br(count = dimensionality)]
    dimension: Vec<u64>,

    #[br(count = dimensionality)]
    dimension_max: Vec<u64>,

    #[br(count = dimensionality, if(flags & 0x2 > 0))]
    permutation_index: Option<Vec<u64>>,
}

#[derive(BinRead, Debug)]
struct DatatypeMessage {
    version_and_class: VersionAndClass,
    #[br(args { class: version_and_class.class() })]
    type_desc: TypeDescriptor,
}

#[bitfield]
#[derive(BinRead, Debug)]
#[br(map = Self::from_bytes)]
pub struct VersionAndClass {
    class: B4,
    version: B4,
}

#[derive(BinRead, Debug)]
#[br(import { class: u8 })]
enum TypeDescriptor {
    #[br(pre_assert(class == 0))]
    FixedPoint(FixedPointDescriptor),
    #[br(pre_assert(class == 1))]
    FloatingPoint(FloatingPointDescriptor),
    #[br(pre_assert(class == 3))]
    String(StringDescriptor),
    #[br(pre_assert(class > 1 && class != 3))]
    UnimplementedTypeClass,
}

#[bitfield(bits = 88)]
#[derive(BinRead, Debug)]
#[br(map = Self::from_bytes)]
pub struct FixedPointDescriptor {
    byte_order: B1,
    low_padding: B1,
    high_padding: B1,
    signed: B1,
    rest: B20,
    size: u32,
    bit_offset: u16,
    bit_precision: u16,
}

#[bitfield(bits = 152)]
#[derive(BinRead, Debug)]
#[br(map = Self::from_bytes)]
pub struct FloatingPointDescriptor {
    byte_order: B1,
    low_padding: B1,
    high_padding: B1,
    internal_padding: B1,
    mantissa_normalization: B2,
    reserved: B2,
    sign_location: u8,
    rest: u8,
    size: u32,
    bit_offset: u16,
    bit_precision: u16,
    exponent_location: u8,
    exponent_size: u8,
    mantissa_location: u8,
    mantissa_size: u8,
    exponent_bias: u32,
}

#[bitfield(bits = 56)]
#[derive(BinRead, Debug)]
#[br(map = Self::from_bytes)]
pub struct StringDescriptor {
    padding: B4,
    character_set: B4,
    rest: B16,
    size: u32,
}

#[derive(BinRead, Debug)]
struct FillValueMessage {
    //#[br(assert(version == 1))]
    version: u8,
    space_allocation_time: u8,
    fill_value_write_time: u8,
    fill_value_defined: u8,

    #[br(if(fill_value_defined == 1))]
    size: Option<u32>,

    #[br(if(fill_value_defined == 1))]
    #[br(count = size.unwrap())]
    fill_value: Option<Vec<u8>>,
}

#[derive(BinRead, Debug)]
struct DataLaymoutMessage {
    version: u8,
    #[br(args { version })]
    inner: DataLayoutInner,
}

#[derive(BinRead, Debug)]
#[br(import { version: u8 })]
enum DataLayoutInner {
    #[br(pre_assert(version < 3))]
    V12(DataLayoutV12),
    #[br(pre_assert(version == 3))]
    V3(DataLayoutV3),
}

#[derive(BinRead, Debug)]
struct DataLayoutV12 {
    dimensionality: u8,
    #[br(pad_after = 5)]
    layout_class: u8,

    #[br(if(layout_class > 0))]
    data_address: Option<u64>,

    #[br(count = dimensionality)]
    dimension: Vec<u64>,
}

#[derive(BinRead, Debug)]
struct DataLayoutV3 {
    layout_class: u8,
    #[br(args { layout_class })]
    layout_inner: LayoutInner,
}

#[derive(BinRead, Debug)]
#[br(import { layout_class: u8 })]
enum LayoutInner {
    #[br(pre_assert(layout_class == 0))]
    Compact(DataLayoutCompact),
    #[br(pre_assert(layout_class == 1))]
    Contiguous(DataLayoutContiguous),
    #[br(pre_assert(layout_class == 2))]
    Chunked(DataLayoutChunked),
}

#[derive(BinRead, Debug)]
struct DataLayoutCompact {
    size: u16,
    #[br(count = size)]
    data: Vec<u8>,
}

#[derive(BinRead, Debug)]
struct DataLayoutContiguous {
    address: u64,
    size: u64,
}

#[derive(BinRead, Debug)]
struct DataLayoutChunked {
    dimensionality: u8,
    /// Pointer to a Version 1 B-Tree of the chunk data.
    address: u64,
    #[br(count = dimensionality)]
    dimension_sizes: Vec<u32>,
    dataset_element_size: u32,
}

#[derive(BinRead, Debug)]
struct ObjectHeaderContinuationMessage {
    offset: u64,
    length: u64,
}

#[derive(BinRead, Debug)]
struct AttributeMessage {
    version: u8,
    flags: u8,
    name_size: u16,
    datatype_size: u16,
    dataspace_size: u16,

    #[br(align_after = 8)]
    name: NullString,
    #[br(align_after = 8)]
    datatype: DatatypeMessage,
    #[br(align_after = 8)]
    dataspace: DataspaceMessage,
    // TODO - need to calculate this based on dataspace and datatype
    //#[br(count = data_size)]
    //message: Vec<u8>,
}

#[derive(BinRead, Debug)]
struct ModificationTimMessage {
    #[br(pad_after = 3)]
    version: u8,
    /// modification time in seconds after epoch
    modification_time: u32,
}

#[derive(BinRead)]
#[br(import { data_size: u16 })]
struct UnknownMessage {
    #[br(count = data_size)]
    message: Vec<u8>,
}

impl std::fmt::Debug for UnknownMessage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UnknownMessage")
            .field("length", &self.message.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use std::io::{Cursor, Read};

    use super::*;

    fn get_file() -> Vec<u8> {
        let mut f = std::fs::File::open("datasets/frozen_pbmc_donor_c_molecule_info.h5").unwrap();

        let mut buf = vec![0; 10_000_000];
        f.read_to_end(&mut buf).unwrap();
        buf
    }

    #[test]
    fn basic() {
        let buf = get_file();

        let mut full_file = Cursor::new(&buf[..]);

        let sb = SuperblockV0::read_le(&mut full_file);
        println!("{:#?}", sb);

        let mut c = Cursor::new(&buf[0x88..]);

        let sb = GroupBTreeV1::read_le(&mut c).unwrap();
        println!("{:#?}", sb);

        for e in sb.children {
            let mut c = Cursor::new(&buf[e.child_pointer as usize..]);
            let symbol = GroupSymbolTableNode::read_le(&mut c);
            println!("{:#?}", symbol);

            let ste = &symbol.unwrap().entries[0];
            let mut c = Cursor::new(&buf[ste.object_header_address as usize..]);
            let oh = DataObjectHeader::read_le(&mut c);
            println!("{:#?}", oh);

            let obj = oh.unwrap();

            let Some(HeaderMessage {
                inner: InnerMessage::DataLayout(layout),
                ..
            }) = obj
                .messages
                .iter()
                .find(|x| matches!(x.inner, InnerMessage::DataLayout(_)))
            else {
                continue;
            };

            let Some(HeaderMessage {
                inner: InnerMessage::Dataspace(space),
                ..
            }) = obj
                .messages
                .iter()
                .find(|x| matches!(x.inner, InnerMessage::Dataspace(_)))
            else {
                continue;
            };

            let dimensionality = space.dimensionality;

            if let DataLaymoutMessage {
                version,
                inner:
                    DataLayoutInner::V3(DataLayoutV3 {
                        layout_class,
                        layout_inner:
                            LayoutInner::Chunked(
                                DataLayoutChunked {
                                    dimensionality: _,
                                    address,
                                    dimension_sizes,
                                    dataset_element_size,
                                },
                                ..,
                            ),
                    }),
            } = layout
            {
                // now load a B-Tree at address
                let mut c = Cursor::new(&buf[*address as usize..]);
                let bt = ChunkBTreeV1::read_le_args(&mut c, (dimensionality,));

                println!("dataset is chunked. reading B-Tree:\n{:#?}", bt);

                let it = BTreeIter::new(&mut full_file, bt.unwrap());

                let mut last = 0;

                for c in it {
                    if c.is_err() {
                        println!("btree err: {:?}", c);
                    }

                    let c = c.unwrap();
                    let delta = c.key.offsets[0] - last;
                    if c.key.offsets[0] > 0 {
                        assert_eq!(delta, dimension_sizes[0] as u64)
                    }

                    last = c.key.offsets[0]
                }

                println!("last pos: {last}");
            }

            //if  obj.messages.iter().find(|x| matches!(x.inner, InnerMessage::DataLayout(DataLaymoutMessage {inner: DataLayoutChunked {  }, ..}))) {
        }
    }
}
