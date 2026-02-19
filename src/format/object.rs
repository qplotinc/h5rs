#![allow(dead_code)]
use modular_bitfield::prelude::*;

use binrw::{BinRead, BinResult, NullString};

use crate::{
    Dataset, Group,
    error::H5Result,
    format::metadata::{GroupBTreeV1, LocalHeap},
    object_store::{ObjectStoreFile, read_and_parse_args, read_metadata},
};

/// Object Header
/// https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#sec_fmt4_dataobject
#[derive(BinRead, Debug)]
pub struct DataObjectHeader {
    version: u8,
    #[br(pad_before = 1)]
    total_header_messages: u16,
    object_reference_count: u32,
    #[br(pad_after = 4)]
    object_header_size: u32,

    #[br(parse_with = parse_header_list, args(object_header_size))]
    messages: Vec<HeaderMessage>,
}

#[derive(BinRead, Debug)]
#[br(import(object_header_size: u32))]
struct MessageList {
    #[br(parse_with = parse_header_list, args(object_header_size))]
    messages: Vec<HeaderMessage>,
}

impl DataObjectHeader {
    pub fn symbol_table_message(&self) -> Option<&SymbolTableMessage> {
        self.messages.iter().find_map(|x| match &x.inner {
            InnerMessage::SymbolTable(s) => Some(s),
            _ => None,
        })
    }

    pub fn dataspace_message(&self) -> Option<&DataspaceMessage> {
        self.messages.iter().find_map(|x| match &x.inner {
            InnerMessage::Dataspace(s) => Some(s),
            _ => None,
        })
    }

    pub fn datatype_message(&self) -> Option<&DatatypeMessage> {
        self.messages.iter().find_map(|x| match &x.inner {
            InnerMessage::Datatype(s) => Some(s),
            _ => None,
        })
    }

    pub fn data_layout_message(&self) -> Option<&DataLayoutMessage> {
        self.messages.iter().find_map(|x| match &x.inner {
            InnerMessage::DataLayout(s) => Some(s),
            _ => None,
        })
    }

    pub fn filter_message(&self) -> Option<&FilterMessage> {
        self.messages.iter().find_map(|x| match &x.inner {
            InnerMessage::Filter(s) => Some(s),
            _ => None,
        })
    }

    pub fn attribute_messages(&self) -> Vec<&AttributeMessage> {
        self.messages
            .iter()
            .filter_map(|x| match &x.inner {
                InnerMessage::Attribute(a) => Some(a),
                _ => None,
            })
            .collect()
    }

    pub async fn load_continuation_messages(
        &mut self,
        file: &ObjectStoreFile,
    ) -> H5Result<()> {
        let mut new_messages = vec![];
        for m in self.messages.iter() {
            let InnerMessage::ObjectHeaderContinuation(m) = &m.inner else {
                continue;
            };

            let msg: MessageList =
                read_and_parse_args(file, m.offset, m.length, (m.length as u32,)).await?;
            new_messages.extend(msg.messages);
        }

        self.messages.extend(new_messages);
        Ok(())
    }

    pub async fn to_group(
        &self,
        name: String,
        file: &ObjectStoreFile,
    ) -> Option<H5Result<Group>> {
        let Some(stm) = self.symbol_table_message() else {
            return None;
        };

        let attributes = self.attribute_messages().into_iter().cloned().collect();

        let r: H5Result<Group> = async {
            let btree: GroupBTreeV1 = read_metadata(file, stm.btree_address).await?;
            let local_heap: LocalHeap = read_metadata(file, stm.local_heap_address).await?;
            let loaded_local_heap = local_heap.load(file).await?;

            Ok(Group {
                name,
                btree,
                loaded_local_heap,
                attributes,
            })
        }
        .await;

        Some(r)
    }

    pub fn to_dataset(&self, name: String) -> Option<Dataset> {
        let Some(dataspace) = self.dataspace_message() else {
            return None;
        };

        let Some(datatype) = self.datatype_message() else {
            return None;
        };

        let Some(layout) = self.data_layout_message() else {
            return None;
        };

        Some(Dataset {
            name,
            dataspace: dataspace.clone(),
            datatype: datatype.clone(),
            layout: layout.clone(),
            filter: self.filter_message().cloned(),
            attributes: self.attribute_messages().into_iter().cloned().collect(),
        })
    }
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
    DataLayout(DataLayoutMessage),
    #[br(pre_assert(ty == 11))]
    Filter(FilterMessage),
    #[br(pre_assert(ty == 12))]
    Attribute(AttributeMessage),
    #[br(pre_assert(ty == 16))]
    ObjectHeaderContinuation(ObjectHeaderContinuationMessage),
    #[br(pre_assert(ty == 17))]
    SymbolTable(SymbolTableMessage),
    #[br(pre_assert(ty == 18))]
    ModificationTie(ModificationTimMessage),
    #[br(pre_assert(ty != 12 && ty != 3 && ty != 8 && ty != 11))]
    Unknown(#[br(args{data_size})] UnknownMessage),
}

#[derive(BinRead, Debug)]
struct NilMessage {}

#[derive(BinRead, Debug, Clone)]
pub struct DataspaceMessage {
    #[br(assert(version == 1))]
    pub version: u8,
    pub dimensionality: u8,
    #[br(pad_after = 5)]
    pub flags: u8,
    //reserved1: u8,
    //reserved2: u32,
    #[br(count = dimensionality)]
    pub dimension: Vec<u64>,

    #[br(count = dimensionality)]
    pub dimension_max: Vec<u64>,

    #[br(count = dimensionality, if(flags & 0x2 > 0))]
    pub permutation_index: Option<Vec<u64>>,
}

#[derive(BinRead, Debug, Clone)]
pub struct DatatypeMessage {
    pub version_and_class: VersionAndClass,
    #[br(args { class: version_and_class.class() })]
    pub type_desc: TypeDescriptor,
}

#[bitfield]
#[derive(BinRead, Debug, Clone)]
#[br(map = Self::from_bytes)]
pub struct VersionAndClass {
    pub class: B4,
    pub version: B4,
}

#[derive(BinRead, Debug, Clone)]
#[br(import { class: u8 })]
pub enum TypeDescriptor {
    #[br(pre_assert(class == 0))]
    FixedPoint(FixedPointDescriptor),
    #[br(pre_assert(class == 1))]
    FloatingPoint(FloatingPointDescriptor),
    #[br(pre_assert(class == 3))]
    String(StringDescriptor),
    #[br(pre_assert(class == 9))]
    Variable(VariableLengthDescriptor),
    #[br(pre_assert(class > 1 && class != 3 && class != 9))]
    UnimplementedTypeClass,
}

#[bitfield(bits = 88)]
#[derive(BinRead, Debug, Clone)]
#[br(map = Self::from_bytes)]
pub struct FixedPointDescriptor {
    byte_order: B1,
    low_padding: B1,
    high_padding: B1,
    pub signed: B1,
    rest: B20,
    pub size: u32,
    bit_offset: u16,
    bit_precision: u16,
}

#[bitfield(bits = 152)]
#[derive(BinRead, Debug, Clone)]
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
    pub size: u32,
    bit_offset: u16,
    bit_precision: u16,
    exponent_location: u8,
    exponent_size: u8,
    mantissa_location: u8,
    mantissa_size: u8,
    exponent_bias: u32,
}

impl DatatypeMessage {
    pub fn element_size(&self) -> usize {
        match &self.type_desc {
            TypeDescriptor::FixedPoint(fp) => fp.size() as usize,
            TypeDescriptor::FloatingPoint(fp) => fp.size() as usize,
            TypeDescriptor::String(s) => s.size() as usize,
            // VL references on disk: uint32 length + uint64 heap addr + uint32 heap index = 16
            TypeDescriptor::Variable(_) => 16,
            TypeDescriptor::UnimplementedTypeClass => {
                panic!("element_size not supported for unimplemented type class")
            }
        }
    }
}

impl DataspaceMessage {
    pub fn num_elements(&self) -> usize {
        if self.dimensionality == 0 {
            1
        } else {
            self.dimension.iter().map(|&d| d as usize).product()
        }
    }
}

#[bitfield(bits = 56)]
#[derive(BinRead, Debug, Clone)]
#[br(map = Self::from_bytes)]
pub struct StringDescriptor {
    pub padding: B4,
    pub character_set: B4,
    rest: B16,
    pub size: u32,
}

#[bitfield(bits = 24)]
#[derive(BinRead, Debug, Clone)]
#[br(map = Self::from_bytes)]
pub struct VariableLengthDescriptorBits {
    variable_type: B4,
    padding: B4,
    character_set: B4,
    rest: B12,
}

#[derive(BinRead, Debug, Clone)]
pub struct VariableLengthDescriptor {
    bits: VariableLengthDescriptorBits,
    parent_type: Box<DatatypeMessage>,
}

#[derive(BinRead, Debug)]
pub struct FillValueMessage {
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

#[derive(BinRead, Debug, Clone)]
pub struct DataLayoutMessage {
    pub version: u8,
    #[br(args { version })]
    pub inner: DataLayoutInner,
}

#[derive(BinRead, Debug, Clone)]
#[br(import { version: u8 })]
pub enum DataLayoutInner {
    #[br(pre_assert(version < 3))]
    V12(DataLayoutV12),
    #[br(pre_assert(version == 3))]
    V3(DataLayoutV3),
}

#[derive(BinRead, Debug, Clone)]
pub struct DataLayoutV12 {
    dimensionality: u8,
    #[br(pad_after = 5)]
    layout_class: u8,

    #[br(if(layout_class > 0))]
    data_address: Option<u64>,

    #[br(count = dimensionality)]
    dimension: Vec<u64>,
}

#[derive(BinRead, Debug, Clone)]
pub struct DataLayoutV3 {
    pub layout_class: u8,
    #[br(args { layout_class })]
    pub layout_inner: LayoutInner,
}

#[derive(BinRead, Debug, Clone)]
#[br(import { layout_class: u8 })]
pub enum LayoutInner {
    #[br(pre_assert(layout_class == 0))]
    Compact(DataLayoutCompact),
    #[br(pre_assert(layout_class == 1))]
    Contiguous(DataLayoutContiguous),
    #[br(pre_assert(layout_class == 2))]
    Chunked(DataLayoutChunked),
}

#[derive(BinRead, Debug, Clone)]
pub struct DataLayoutCompact {
    pub size: u16,
    #[br(count = size)]
    pub data: Vec<u8>,
}

#[derive(BinRead, Debug, Clone)]
pub struct DataLayoutContiguous {
    pub address: u64,
    pub size: u64,
}

#[derive(BinRead, Debug, Clone)]
pub struct DataLayoutChunked {
    pub dimensionality: u8,
    /// Pointer to a Version 1 B-Tree of the chunk data.
    pub address: u64,
    #[br(count = dimensionality)]
    pub dimension_sizes: Vec<u32>,
    pub dataset_element_size: u32,
}

#[derive(BinRead, Debug)]
struct ObjectHeaderContinuationMessage {
    offset: u64,
    length: u64,
}

#[derive(BinRead, Debug)]
pub struct SymbolTableMessage {
    pub btree_address: u64,
    pub local_heap_address: u64,
}

#[derive(BinRead, Debug, Clone)]
pub struct FilterMessage {
    pub version: u8,
    #[br(pad_after = 6)]
    pub num_filters: u8,

    #[br(count = num_filters)]
    pub filters: Vec<FilterDescription>,
}

#[derive(BinRead, Debug, Clone)]
#[br(repr = u16)]
pub enum FilterType {
    None = 0,
    Deflate = 1,
    Shuffle = 2,
    Fletcher32 = 3,
    Szip = 4,
    Nbit = 5,
    ScaleOffset = 6,
}

#[derive(BinRead, Debug, Clone)]
pub struct FilterDescription {
    pub filter_type: FilterType,
    pub name_length: u16,
    pub flags: u16,
    pub number_of_client_values: u16,

    #[br(if(name_length > 0), pad_size_to = name_length)]
    pub name: Option<NullString>,

    // FIXME -- this field needs to be padded so that the length is a multiple of 8.
    // but this field may not be 8-byte aligned in the file. So we'll pad up the count.
    // this will lead to 1 extra value in the array if the true length is odd.
    #[br(count = number_of_client_values.next_multiple_of(2))]
    pub client_data: Vec<u32>,
}

#[derive(BinRead, Debug, Clone)]
pub struct AttributeMessage {
    #[br(assert(version == 1))]
    version: u8,
    flags: u8,
    name_size: u16,
    datatype_size: u16,
    dataspace_size: u16,

    #[br(align_after = 8)]
    pub name: NullString,
    #[br(pad_size_to = datatype_size.next_multiple_of(8))]
    pub datatype: DatatypeMessage,
    #[br(pad_size_to = dataspace_size.next_multiple_of(8))]
    pub dataspace: DataspaceMessage,
    #[br(count = datatype.element_size() * dataspace.num_elements())]
    pub data: Vec<u8>,
}

impl AttributeMessage {
    pub fn name(&self) -> String {
        self.name.to_string()
    }

    pub fn read<T: crate::h5type::H5Type>(&self) -> Vec<T> {
        T::check_dtype(&self.datatype);
        bytemuck::cast_slice(&self.data).to_vec()
    }

    pub fn read_strings(&self) -> Vec<String> {
        let TypeDescriptor::String(ref sd) = self.datatype.type_desc else {
            panic!("read_strings called on non-string type: {:?}", self.datatype.type_desc);
        };
        let elem_size = sd.size() as usize;
        let num = self.dataspace.num_elements();
        self.data
            .chunks(elem_size)
            .take(num)
            .map(|chunk| {
                let trimmed = match sd.padding() {
                    // Null-terminated: data up to first null
                    0 => &chunk[..chunk.iter().position(|&b| b == 0).unwrap_or(chunk.len())],
                    // Null-padded: strip trailing nulls
                    1 => {
                        let end = chunk.iter().rposition(|&b| b != 0).map_or(0, |i| i + 1);
                        &chunk[..end]
                    }
                    // Space-padded: strip trailing spaces
                    2 => {
                        let end = chunk.iter().rposition(|&b| b != b' ').map_or(0, |i| i + 1);
                        &chunk[..end]
                    }
                    p => panic!("unknown string padding type: {p}"),
                };
                String::from_utf8_lossy(trimmed).into_owned()
            })
            .collect()
    }
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
