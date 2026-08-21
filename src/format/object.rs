#![allow(unused_assignments)] // binrw `#[br(import {...})]` triggers this in derived impls

use binrw::{BinRead, BinResult};

use crate::{
    Dataset, Group, GroupLinks,
    error::{H5Error, H5Result},
    format::metadata::{GroupBTreeV1, LocalHeap},
    object_store::{ObjectStoreFile, read_and_parse_args, read_metadata},
};

/// A data object header, in either the version 1 or version 2 (`OHDR`) encoding.
///
/// <https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#sec_fmt4_dataobject>
#[derive(Debug)]
pub struct DataObjectHeader {
    /// Object header version: 1 or 2.
    pub version: u8,
    /// Version 2 only: message prefixes carry a 2-byte creation order field.
    track_order: bool,
    messages: Vec<HeaderMessage>,
}

/// Prefix bytes before each message's data: type, size, flags, and for version 2
/// headers that track creation order, the 2-byte order field.
const MSG_PREFIX_V1: u64 = 8;
const MSG_PREFIX_V2: u64 = 4;
const MSG_PREFIX_V2_ORDERED: u64 = 6;

impl BinRead for DataObjectHeader {
    type Args<'a> = ();

    fn read_options<R: std::io::Read + std::io::Seek>(
        reader: &mut R,
        endian: binrw::Endian,
        _: Self::Args<'_>,
    ) -> BinResult<Self> {
        let start = reader.stream_position()?;
        let mut signature = [0u8; 4];
        reader.read_exact(&mut signature)?;

        if &signature == OHDR_SIGNATURE {
            Self::read_v2_body(reader, endian, start)
        } else {
            reader.seek(std::io::SeekFrom::Start(start))?;
            Self::read_v1_body(reader, endian)
        }
    }
}

const OHDR_SIGNATURE: &[u8; 4] = b"OHDR";
const OCHK_SIGNATURE: &[u8; 4] = b"OCHK";

impl DataObjectHeader {
    /// Version 1 prefix, then a message area whose length the prefix gives.
    fn read_v1_body<R: std::io::Read + std::io::Seek>(
        reader: &mut R,
        endian: binrw::Endian,
    ) -> BinResult<Self> {
        let prefix = ObjectHeaderPrefixV1::read_options(reader, endian, ())?;
        let messages = parse_messages(
            reader,
            endian,
            prefix.object_header_size as u64,
            MessageFraming::V1,
        )?;
        Ok(DataObjectHeader {
            version: 1,
            track_order: false,
            messages,
        })
    }

    /// Version 2 prefix. `start` is the position of the `OHDR` signature, which
    /// has already been consumed.
    fn read_v2_body<R: std::io::Read + std::io::Seek>(
        reader: &mut R,
        endian: binrw::Endian,
        start: u64,
    ) -> BinResult<Self> {
        let version = u8::read_options(reader, endian, ())?;
        if version != 2 {
            return Err(binrw::Error::AssertFail {
                pos: start,
                message: format!("unsupported object header version {version} after OHDR"),
            });
        }
        let flags = u8::read_options(reader, endian, ())?;

        // Bit 5: access / modification / change / birth times.
        if flags & 0x20 != 0 {
            reader.seek(std::io::SeekFrom::Current(16))?;
        }
        // Bit 4: non-default attribute storage phase change values.
        if flags & 0x10 != 0 {
            reader.seek(std::io::SeekFrom::Current(4))?;
        }

        // Bits 0-1 select the width of the chunk-0 size field.
        let chunk_size = match flags & 0x03 {
            0 => u8::read_options(reader, endian, ())? as u64,
            1 => u16::read_options(reader, endian, ())? as u64,
            2 => u32::read_options(reader, endian, ())? as u64,
            _ => u64::read_options(reader, endian, ())?,
        };

        // Bit 2: attribute creation order tracked, which also adds a creation
        // order field to every message prefix in this header.
        let track_order = flags & 0x04 != 0;
        let messages = parse_messages(
            reader,
            endian,
            chunk_size,
            MessageFraming::V2 { track_order },
        )?;

        Ok(DataObjectHeader {
            version: 2,
            track_order,
            messages,
        })
    }
}

/// Bytes the first chunk of an object header occupies on disk, read from its
/// opening bytes alone.
///
/// Object headers have no fixed size — a large compact dataset or a run of
/// attributes can push one well past any default fetch — so the reader has to
/// learn the extent before it can fetch the whole thing.
pub fn header_chunk_extent(bytes: &[u8]) -> Option<u64> {
    if bytes.len() >= 4 && &bytes[..4] == OHDR_SIGNATURE {
        let flags = *bytes.get(5)?;
        let mut pos = 6usize;
        if flags & 0x20 != 0 {
            pos += 16; // access / modification / change / birth times
        }
        if flags & 0x10 != 0 {
            pos += 4; // attribute storage phase change values
        }
        let width = 1usize << (flags & 0x03);
        let mut buf = [0u8; 8];
        buf[..width].copy_from_slice(bytes.get(pos..pos + width)?);
        // prefix, the size field itself, the message area, and the checksum
        Some(pos as u64 + width as u64 + u64::from_le_bytes(buf) + 4)
    } else {
        // Version 1: version, padding, message count, reference count, then the
        // message area size, then four bytes of padding.
        let size = u32::from_le_bytes(bytes.get(12..16)?.try_into().ok()?) as u64;
        Some(16 + size)
    }
}

/// Version 1 object header prefix, before the message area.
#[derive(BinRead, Debug)]
#[allow(dead_code)]
struct ObjectHeaderPrefixV1 {
    #[br(assert(version == 1, "unsupported object header version {}", version))]
    version: u8,
    #[br(pad_before = 1)]
    total_header_messages: u16,
    object_reference_count: u32,
    #[br(pad_after = 4)]
    object_header_size: u32,
}

/// How the messages in a header chunk are framed.
#[derive(Debug, Clone, Copy)]
enum MessageFraming {
    /// Type and size are 16-bit, and each message is padded to 8 bytes.
    V1,
    /// Type is 8-bit, messages are packed with no padding, and an optional
    /// 2-byte creation order sits between the flags and the data.
    V2 { track_order: bool },
}

impl MessageFraming {
    fn prefix_len(&self) -> u64 {
        match self {
            MessageFraming::V1 => MSG_PREFIX_V1,
            MessageFraming::V2 { track_order: false } => MSG_PREFIX_V2,
            MessageFraming::V2 { track_order: true } => MSG_PREFIX_V2_ORDERED,
        }
    }
}

/// Read messages until `area_size` bytes have been consumed.
///
/// The message count is not stored, so the loop is bounded by the byte extent of
/// the chunk. Version 2 chunks may end with a gap too small to hold another
/// message prefix; that gap is simply left unread.
fn parse_messages<R: std::io::Read + std::io::Seek>(
    reader: &mut R,
    endian: binrw::Endian,
    area_size: u64,
    framing: MessageFraming,
) -> BinResult<Vec<HeaderMessage>> {
    let area_start = reader.stream_position()?;
    let area_end = area_start + area_size;
    let prefix_len = framing.prefix_len();
    let mut messages = vec![];

    loop {
        let pos = reader.stream_position()?;
        if pos + prefix_len > area_end {
            break;
        }

        let (message_type, data_size, flags) = match framing {
            MessageFraming::V1 => {
                let ty = u16::read_options(reader, endian, ())?;
                let size = u16::read_options(reader, endian, ())?;
                let flags = u8::read_options(reader, endian, ())?;
                reader.seek(std::io::SeekFrom::Current(3))?; // reserved
                (ty, size, flags)
            }
            MessageFraming::V2 { track_order } => {
                let ty = u8::read_options(reader, endian, ())? as u16;
                let size = u16::read_options(reader, endian, ())?;
                let flags = u8::read_options(reader, endian, ())?;
                if track_order {
                    reader.seek(std::io::SeekFrom::Current(2))?;
                }
                (ty, size, flags)
            }
        };

        let data_start = pos + prefix_len;
        if data_start + data_size as u64 > area_end {
            return Err(binrw::Error::AssertFail {
                pos,
                message: format!(
                    "object header message of {data_size} bytes overruns its {area_size}-byte chunk"
                ),
            });
        }

        // Bit 1 of the message flags means the data is a shared-message
        // reference rather than the message itself.
        let inner = if flags & 0x02 != 0 {
            InnerMessage::Shared { message_type }
        } else {
            parse_inner_message(reader, endian, message_type, data_size)?
        };

        messages.push(HeaderMessage {
            message_type,
            data_size,
            flags,
            inner,
        });

        reader.seek(std::io::SeekFrom::Start(data_start + data_size as u64))?;
    }

    reader.seek(std::io::SeekFrom::Start(area_end))?;
    Ok(messages)
}

/// Dispatch on the message type. Unlike a `binrw` enum this never falls through
/// to `Unknown` when a known message fails to parse — the error propagates.
fn parse_inner_message<R: std::io::Read + std::io::Seek>(
    reader: &mut R,
    endian: binrw::Endian,
    message_type: u16,
    data_size: u16,
) -> BinResult<InnerMessage> {
    macro_rules! msg {
        ($variant:ident) => {
            InnerMessage::$variant(BinRead::read_options(reader, endian, ())?)
        };
    }
    Ok(match message_type {
        0 => InnerMessage::Nil,
        1 => msg!(Dataspace),
        2 => msg!(LinkInfo),
        3 => msg!(Datatype),
        5 => msg!(FillValue),
        6 => msg!(Link),
        8 => msg!(DataLayout),
        11 => msg!(Filter),
        12 => InnerMessage::Attribute(AttributeMessage::read_options(
            reader,
            endian,
            (data_size,),
        )?),
        16 => msg!(ObjectHeaderContinuation),
        17 => msg!(SymbolTable),
        21 => msg!(AttributeInfo),
        _ => InnerMessage::Unknown,
    })
}

#[derive(Debug)]
#[allow(dead_code)]
struct HeaderMessage {
    message_type: u16,
    data_size: u16,
    flags: u8,
    inner: InnerMessage,
}

#[derive(Debug)]
#[allow(dead_code)]
enum InnerMessage {
    Nil,
    Dataspace(DataspaceMessage),
    LinkInfo(LinkInfoMessage),
    Datatype(DatatypeMessage),
    FillValue(FillValueMessage),
    Link(LinkMessage),
    DataLayout(DataLayoutMessage),
    Filter(FilterMessage),
    Attribute(AttributeMessage),
    ObjectHeaderContinuation(ObjectHeaderContinuationMessage),
    SymbolTable(SymbolTableMessage),
    AttributeInfo(AttributeInfoMessage),
    /// The message data is a reference to a message shared between objects.
    Shared {
        message_type: u16,
    },
    /// A message type h5rs does not interpret. Its bytes are skipped.
    Unknown,
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

    pub fn link_info_message(&self) -> Option<&LinkInfoMessage> {
        self.messages.iter().find_map(|x| match &x.inner {
            InnerMessage::LinkInfo(s) => Some(s),
            _ => None,
        })
    }

    pub fn attribute_info_message(&self) -> Option<&AttributeInfoMessage> {
        self.messages.iter().find_map(|x| match &x.inner {
            InnerMessage::AttributeInfo(s) => Some(s),
            _ => None,
        })
    }

    /// Links stored "compactly", as messages in the object header itself.
    pub fn link_messages(&self) -> Vec<&LinkMessage> {
        self.messages
            .iter()
            .filter_map(|x| match &x.inner {
                InnerMessage::Link(l) => Some(l),
                _ => None,
            })
            .collect()
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

    /// Return `Unsupported` if a message h5rs needs is stored as a shared
    /// message, which would otherwise look like the message simply being absent.
    fn reject_shared(&self, wanted: &[(u16, &str)]) -> H5Result<()> {
        for m in &self.messages {
            if let InnerMessage::Shared { message_type } = m.inner {
                if let Some((_, name)) = wanted.iter().find(|(ty, _)| *ty == message_type) {
                    return Err(H5Error::unsupported(format!("shared {name} message")));
                }
            }
        }
        Ok(())
    }

    /// Follow every object header continuation message and append the messages
    /// found there. Continuation blocks may themselves continue, so this loops
    /// until no new continuations appear.
    pub async fn load_continuation_messages(&mut self, file: &ObjectStoreFile) -> H5Result<()> {
        let framing = if self.version == 1 {
            MessageFraming::V1
        } else {
            MessageFraming::V2 {
                track_order: self.track_order,
            }
        };

        let mut pending: Vec<ObjectHeaderContinuationMessage> = self.continuation_blocks();
        let mut seen: Vec<u64> = pending.iter().map(|c| c.offset).collect();

        while let Some(cont) = pending.pop() {
            let block: ContinuationBlock =
                read_and_parse_args(file, cont.offset, cont.length, (cont.length, framing)).await?;

            for next in block.messages.iter().filter_map(|m| match &m.inner {
                InnerMessage::ObjectHeaderContinuation(c) => Some(c),
                _ => None,
            }) {
                if !seen.contains(&next.offset) {
                    seen.push(next.offset);
                    pending.push(ObjectHeaderContinuationMessage {
                        offset: next.offset,
                        length: next.length,
                    });
                }
            }

            self.messages.extend(block.messages);
        }

        Ok(())
    }

    fn continuation_blocks(&self) -> Vec<ObjectHeaderContinuationMessage> {
        self.messages
            .iter()
            .filter_map(|m| match &m.inner {
                InnerMessage::ObjectHeaderContinuation(c) => {
                    Some(ObjectHeaderContinuationMessage {
                        offset: c.offset,
                        length: c.length,
                    })
                }
                _ => None,
            })
            .collect()
    }

    /// Every attribute of this object: those stored as messages in the header,
    /// plus any stored densely in a fractal heap.
    pub async fn all_attributes(&self, file: &ObjectStoreFile) -> H5Result<Vec<AttributeMessage>> {
        let mut attributes: Vec<AttributeMessage> =
            self.attribute_messages().into_iter().cloned().collect();

        if let Some(info) = self.attribute_info_message() {
            if !is_undefined_address(info.fractal_heap_address) {
                attributes.extend(
                    crate::format::dense::read_attributes(
                        file,
                        info.fractal_heap_address,
                        info.name_btree_address,
                    )
                    .await?,
                );
            }
        }
        Ok(attributes)
    }

    /// Interpret this header as a group, if it is one.
    ///
    /// Old-style groups carry a Symbol Table message; new-style groups carry a
    /// Link Info message and store their links either compactly (as Link
    /// messages here) or densely (in a fractal heap indexed by a v2 B-tree).
    pub async fn to_group(&self, file: &ObjectStoreFile) -> H5Result<Option<Group>> {
        let attributes = self.all_attributes(file).await?;

        if let Some(stm) = self.symbol_table_message() {
            let btree: GroupBTreeV1 = read_metadata(file, stm.btree_address).await?;
            let local_heap: LocalHeap = read_metadata(file, stm.local_heap_address).await?;
            let loaded_local_heap = local_heap.load(file).await?;
            return Ok(Some(Group {
                links: GroupLinks::SymbolTable {
                    btree,
                    heap: loaded_local_heap,
                },
                attributes,
            }));
        }

        let Some(info) = self.link_info_message() else {
            return Ok(None);
        };

        let links = if is_undefined_address(info.fractal_heap_address) {
            GroupLinks::Compact(self.link_messages().into_iter().cloned().collect())
        } else {
            GroupLinks::Dense {
                fractal_heap_address: info.fractal_heap_address,
                name_btree_address: info.name_btree_address,
            }
        };

        Ok(Some(Group { links, attributes }))
    }

    pub async fn to_dataset(
        &self,
        name: String,
        file: &ObjectStoreFile,
    ) -> H5Result<Option<Dataset>> {
        self.reject_shared(&[(1, "dataspace"), (3, "datatype"), (8, "data layout")])?;

        let (Some(dataspace), Some(datatype), Some(layout)) = (
            self.dataspace_message(),
            self.datatype_message(),
            self.data_layout_message(),
        ) else {
            return Ok(None);
        };

        Ok(Some(Dataset::new(
            name,
            dataspace.clone(),
            datatype.clone(),
            layout.clone(),
            self.filter_message().cloned(),
            self.all_attributes(file).await?,
        )))
    }
}

/// The HDF5 "undefined address" sentinel: all bits set.
pub fn is_undefined_address(addr: u64) -> bool {
    addr == u64::MAX
}

/// A version 1 or version 2 (`OCHK`) object header continuation block.
#[derive(Debug)]
struct ContinuationBlock {
    messages: Vec<HeaderMessage>,
}

impl BinRead for ContinuationBlock {
    type Args<'a> = (u64, MessageFraming);

    fn read_options<R: std::io::Read + std::io::Seek>(
        reader: &mut R,
        endian: binrw::Endian,
        (length, framing): Self::Args<'_>,
    ) -> BinResult<Self> {
        let area_size = match framing {
            MessageFraming::V1 => length,
            MessageFraming::V2 { .. } => {
                let mut signature = [0u8; 4];
                reader.read_exact(&mut signature)?;
                if &signature != OCHK_SIGNATURE {
                    return Err(binrw::Error::BadMagic {
                        pos: 0,
                        found: Box::new(signature),
                    });
                }
                // Minus the signature just read and the trailing checksum.
                length.saturating_sub(8)
            }
        };
        Ok(ContinuationBlock {
            messages: parse_messages(reader, endian, area_size, framing)?,
        })
    }
}

/// Dataspace message, version 1 or 2.
///
/// Version 2 drops the never-implemented permutation indices and adds an
/// explicit dataspace type (scalar / simple / null).
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct DataspaceMessage {
    pub version: u8,
    pub dimensionality: u8,
    pub flags: u8,
    /// 0 scalar, 1 simple, 2 null. Version 1 messages have no such field, so
    /// it is derived from the dimensionality there.
    pub dataspace_type: u8,
    pub dimension: Vec<u64>,
    pub dimension_max: Vec<u64>,
}

impl BinRead for DataspaceMessage {
    type Args<'a> = ();

    fn read_options<R: std::io::Read + std::io::Seek>(
        reader: &mut R,
        endian: binrw::Endian,
        _: Self::Args<'_>,
    ) -> BinResult<Self> {
        let pos = reader.stream_position()?;
        let version = u8::read_options(reader, endian, ())?;
        let dimensionality = u8::read_options(reader, endian, ())?;
        let flags = u8::read_options(reader, endian, ())?;

        let dataspace_type = match version {
            1 => {
                // Reserved byte, then a reserved 32-bit word. Version 1 has no
                // null dataspace: rank zero means scalar.
                reader.seek(std::io::SeekFrom::Current(5))?;
                if dimensionality == 0 { 0 } else { 1 }
            }
            2 => u8::read_options(reader, endian, ())?,
            v => {
                return Err(binrw::Error::AssertFail {
                    pos,
                    message: format!("unsupported dataspace message version {v}"),
                });
            }
        };

        let mut dimension = Vec::with_capacity(dimensionality as usize);
        for _ in 0..dimensionality {
            dimension.push(u64::read_options(reader, endian, ())?);
        }

        // Bit 0: maximum dimensions are stored. When absent they equal the
        // current dimensions.
        let dimension_max = if flags & 0x01 != 0 {
            let mut max = Vec::with_capacity(dimensionality as usize);
            for _ in 0..dimensionality {
                max.push(u64::read_options(reader, endian, ())?);
            }
            max
        } else {
            dimension.clone()
        };

        Ok(DataspaceMessage {
            version,
            dimensionality,
            flags,
            dataspace_type,
            dimension,
            dimension_max,
        })
    }
}

/// Link Info message: where a "new style" group keeps its links.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct LinkInfoMessage {
    pub version: u8,
    pub flags: u8,
    /// Address of the fractal heap holding densely stored links, or the
    /// undefined address when the group stores its links compactly.
    pub fractal_heap_address: u64,
    /// Address of the v2 B-tree indexing link names.
    pub name_btree_address: u64,
}

impl BinRead for LinkInfoMessage {
    type Args<'a> = ();

    fn read_options<R: std::io::Read + std::io::Seek>(
        reader: &mut R,
        endian: binrw::Endian,
        _: Self::Args<'_>,
    ) -> BinResult<Self> {
        let version = u8::read_options(reader, endian, ())?;
        let flags = u8::read_options(reader, endian, ())?;
        // Bit 0: maximum creation index is stored.
        if flags & 0x01 != 0 {
            reader.seek(std::io::SeekFrom::Current(8))?;
        }
        let fractal_heap_address = u64::read_options(reader, endian, ())?;
        let name_btree_address = u64::read_options(reader, endian, ())?;
        // Bit 1: creation-order index B-tree address follows; unused here.
        Ok(LinkInfoMessage {
            version,
            flags,
            fractal_heap_address,
            name_btree_address,
        })
    }
}

/// Attribute Info message: where an object keeps densely stored attributes.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct AttributeInfoMessage {
    pub version: u8,
    pub flags: u8,
    pub fractal_heap_address: u64,
    pub name_btree_address: u64,
}

impl BinRead for AttributeInfoMessage {
    type Args<'a> = ();

    fn read_options<R: std::io::Read + std::io::Seek>(
        reader: &mut R,
        endian: binrw::Endian,
        _: Self::Args<'_>,
    ) -> BinResult<Self> {
        let version = u8::read_options(reader, endian, ())?;
        let flags = u8::read_options(reader, endian, ())?;
        // Bit 0: maximum creation index (2 bytes, unlike the Link Info message).
        if flags & 0x01 != 0 {
            reader.seek(std::io::SeekFrom::Current(2))?;
        }
        let fractal_heap_address = u64::read_options(reader, endian, ())?;
        let name_btree_address = u64::read_options(reader, endian, ())?;
        Ok(AttributeInfoMessage {
            version,
            flags,
            fractal_heap_address,
            name_btree_address,
        })
    }
}

/// A single link in a "new style" group, either stored compactly as an object
/// header message or densely as an object in the group's fractal heap.
#[derive(Debug, Clone)]
pub struct LinkMessage {
    /// The link's name.
    pub name: String,
    /// What the link points at.
    pub target: LinkTarget,
}

/// The resolved destination of a [`LinkMessage`].
///
/// Only hard links are followed; the other variants carry enough to report what
/// was skipped.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub enum LinkTarget {
    /// A hard link: the address of the target's object header.
    Hard(u64),
    /// A soft link: a path to resolve within this file.
    Soft(String),
    /// A link type h5rs does not follow (external or user-defined).
    Other(u8),
}

impl BinRead for LinkMessage {
    type Args<'a> = ();

    fn read_options<R: std::io::Read + std::io::Seek>(
        reader: &mut R,
        endian: binrw::Endian,
        _: Self::Args<'_>,
    ) -> BinResult<Self> {
        let pos = reader.stream_position()?;
        let version = u8::read_options(reader, endian, ())?;
        if version != 1 {
            return Err(binrw::Error::AssertFail {
                pos,
                message: format!("unsupported link message version {version}"),
            });
        }
        let flags = u8::read_options(reader, endian, ())?;

        // Bit 3: an explicit link type follows; otherwise the link is hard.
        let link_type = if flags & 0x08 != 0 {
            u8::read_options(reader, endian, ())?
        } else {
            0
        };
        // Bit 2: creation order.
        if flags & 0x04 != 0 {
            reader.seek(std::io::SeekFrom::Current(8))?;
        }
        // Bit 4: name character set.
        if flags & 0x10 != 0 {
            reader.seek(std::io::SeekFrom::Current(1))?;
        }

        // Bits 0-1 select the width of the name length field.
        let name_len = match flags & 0x03 {
            0 => u8::read_options(reader, endian, ())? as u64,
            1 => u16::read_options(reader, endian, ())? as u64,
            2 => u32::read_options(reader, endian, ())? as u64,
            _ => u64::read_options(reader, endian, ())?,
        };

        let mut name_bytes = vec![0u8; name_len as usize];
        reader.read_exact(&mut name_bytes)?;
        let name = String::from_utf8_lossy(&name_bytes).into_owned();

        let target = match link_type {
            0 => LinkTarget::Hard(u64::read_options(reader, endian, ())?),
            1 => {
                let len = u16::read_options(reader, endian, ())? as usize;
                let mut value = vec![0u8; len];
                reader.read_exact(&mut value)?;
                LinkTarget::Soft(String::from_utf8_lossy(&value).into_owned())
            }
            other => LinkTarget::Other(other),
        };

        Ok(LinkMessage { name, target })
    }
}

#[derive(BinRead, Debug, Clone)]
pub struct DatatypeMessage {
    pub version_and_class: VersionAndClass,
    #[br(args { class: version_and_class.class() })]
    pub type_desc: TypeDescriptor,
}

#[derive(BinRead, Debug, Clone)]
#[br(map = VersionAndClass)]
pub struct VersionAndClass(u8);

impl VersionAndClass {
    pub fn class(&self) -> u8 {
        self.0 & 0x0F
    }
    pub fn version(&self) -> u8 {
        self.0 >> 4
    }
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

#[derive(BinRead, Debug, Clone)]
#[br(map = FixedPointDescriptor)]
pub struct FixedPointDescriptor([u8; 11]);

impl FixedPointDescriptor {
    pub fn signed(&self) -> u8 {
        (self.0[0] >> 3) & 1
    }
    pub fn size(&self) -> u32 {
        u32::from_le_bytes([self.0[3], self.0[4], self.0[5], self.0[6]])
    }
}

#[derive(BinRead, Debug, Clone)]
#[br(map = FloatingPointDescriptor)]
pub struct FloatingPointDescriptor([u8; 19]);

impl FloatingPointDescriptor {
    pub fn size(&self) -> u32 {
        u32::from_le_bytes([self.0[3], self.0[4], self.0[5], self.0[6]])
    }
}

impl DatatypeMessage {
    pub fn element_size(&self) -> H5Result<usize> {
        Ok(match &self.type_desc {
            TypeDescriptor::FixedPoint(fp) => fp.size() as usize,
            TypeDescriptor::FloatingPoint(fp) => fp.size() as usize,
            TypeDescriptor::String(s) => s.size() as usize,
            // VL references on disk: uint32 length + uint64 heap addr + uint32 heap index = 16
            TypeDescriptor::Variable(_) => 16,
            TypeDescriptor::UnimplementedTypeClass => {
                return Err(H5Error::unsupported(
                    "datatype class (only fixed-point, floating-point, string and \
                     variable-length are implemented)",
                ));
            }
        })
    }
}

impl DataspaceMessage {
    /// A null dataspace holds no elements at all, unlike a scalar one which
    /// holds exactly one.
    pub fn is_null(&self) -> bool {
        self.dataspace_type == 2
    }

    pub fn num_elements(&self) -> usize {
        if self.is_null() {
            0
        } else if self.dimensionality == 0 {
            1
        } else {
            self.dimension.iter().map(|&d| d as usize).product()
        }
    }
}

#[derive(BinRead, Debug, Clone)]
#[br(map = StringDescriptor)]
pub struct StringDescriptor([u8; 7]);

impl StringDescriptor {
    pub fn padding(&self) -> u8 {
        self.0[0] & 0x0F
    }
    pub fn character_set(&self) -> u8 {
        self.0[0] >> 4
    }
    pub fn size(&self) -> u32 {
        u32::from_le_bytes([self.0[3], self.0[4], self.0[5], self.0[6]])
    }
}

#[derive(BinRead, Debug, Clone)]
#[br(map = VariableLengthDescriptorBits)]
#[allow(dead_code)]
pub struct VariableLengthDescriptorBits([u8; 3]);

#[derive(BinRead, Debug, Clone)]
#[allow(dead_code)]
pub struct VariableLengthDescriptor {
    bits: VariableLengthDescriptorBits,
    parent_type: Box<DatatypeMessage>,
}

/// Fill Value message, versions 1 through 3.
///
/// Version 3 packs the allocation and write times into a flags byte and adds an
/// explicit "undefined" state. h5rs parses the message so that object headers
/// stay readable, and keeps the fill value for callers that need it.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct FillValueMessage {
    pub version: u8,
    /// The fill value bytes, when the dataset defines one.
    pub fill_value: Option<Vec<u8>>,
}

impl BinRead for FillValueMessage {
    type Args<'a> = ();

    fn read_options<R: std::io::Read + std::io::Seek>(
        reader: &mut R,
        endian: binrw::Endian,
        _: Self::Args<'_>,
    ) -> BinResult<Self> {
        let pos = reader.stream_position()?;
        let version = u8::read_options(reader, endian, ())?;

        let defined = match version {
            1 | 2 => {
                reader.seek(std::io::SeekFrom::Current(2))?; // allocation and write time
                let defined = u8::read_options(reader, endian, ())?;
                // Version 1 always stores the size and value.
                version == 1 || defined == 1
            }
            3 => {
                let flags = u8::read_options(reader, endian, ())?;
                flags & 0x20 != 0
            }
            v => {
                return Err(binrw::Error::AssertFail {
                    pos,
                    message: format!("unsupported fill value message version {v}"),
                });
            }
        };

        let fill_value = if defined {
            let size = u32::read_options(reader, endian, ())? as usize;
            let mut value = vec![0u8; size];
            reader.read_exact(&mut value)?;
            Some(value)
        } else {
            None
        };

        Ok(FillValueMessage {
            version,
            fill_value,
        })
    }
}

/// Data Layout message, versions 1 through 5.
///
/// Versions 1-3 always index chunks with a version 1 B-tree. Version 4 (HDF5
/// 1.10+) adds a choice of five chunk index structures; version 5 (HDF5 2.0+)
/// differs only in how filtered chunk records encode their size.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct DataLayoutMessage {
    pub version: u8,
    pub layout_class: u8,
    pub inner: LayoutInner,
}

/// Class-specific layout information.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub enum LayoutInner {
    /// Raw data stored inline in the object header.
    Compact { data: Vec<u8> },
    /// Raw data stored as one contiguous run.
    Contiguous { address: u64, size: u64 },
    /// Raw data split into equal-sized chunks.
    Chunked(ChunkedLayout),
    /// A layout class h5rs does not read (currently only virtual datasets).
    Unsupported(u8),
}

/// Chunked layout, normalised across message versions.
#[derive(Debug, Clone)]
pub struct ChunkedLayout {
    /// Chunk extent along each dataset dimension, outermost first.
    pub chunk_dims: Vec<u64>,
    /// Size of one dataset element in bytes.
    pub element_size: u32,
    /// How to find the address of each chunk.
    pub index: ChunkIndex,
}

impl ChunkedLayout {
    /// Number of dataset dimensions.
    pub fn ndim(&self) -> usize {
        self.chunk_dims.len()
    }

    /// Bytes in one whole (non-edge) chunk.
    pub fn chunk_bytes(&self) -> u64 {
        self.chunk_dims.iter().product::<u64>() * self.element_size as u64
    }
}

/// The structure used to map a chunk's coordinates to its address on disk.
///
/// <https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#sec_fmt4_appendixc>
#[derive(Debug, Clone)]
pub enum ChunkIndex {
    /// Version 1 B-tree; used by every pre-1.10 file.
    BTreeV1 { address: u64 },
    /// The dataset holds exactly one chunk, stored at this address.
    SingleChunk {
        address: u64,
        /// Present when the single chunk is filtered.
        filtered: Option<(u64, u32)>,
    },
    /// Chunks laid out contiguously from a base address, addressed
    /// arithmetically. Only used for unfiltered, early-allocated datasets.
    Implicit { address: u64 },
    /// A flat array of chunk records; used when all dimensions are fixed.
    FixedArray { address: u64 },
    /// Used when exactly one dimension is unlimited.
    ExtensibleArray { address: u64 },
    /// Used when more than one dimension is unlimited.
    BTreeV2 { address: u64 },
}

impl ChunkIndex {
    /// Address of the index (or of the data, for single-chunk and implicit
    /// indexes). The undefined address means storage is not yet allocated.
    pub fn address(&self) -> u64 {
        match self {
            ChunkIndex::BTreeV1 { address }
            | ChunkIndex::SingleChunk { address, .. }
            | ChunkIndex::Implicit { address }
            | ChunkIndex::FixedArray { address }
            | ChunkIndex::ExtensibleArray { address }
            | ChunkIndex::BTreeV2 { address } => *address,
        }
    }
}

impl DataLayoutMessage {
    /// The chunked layout, if this dataset is chunked.
    pub fn chunked(&self) -> Option<&ChunkedLayout> {
        match &self.inner {
            LayoutInner::Chunked(c) => Some(c),
            _ => None,
        }
    }
}

impl BinRead for DataLayoutMessage {
    type Args<'a> = ();

    fn read_options<R: std::io::Read + std::io::Seek>(
        reader: &mut R,
        endian: binrw::Endian,
        _: Self::Args<'_>,
    ) -> BinResult<Self> {
        let pos = reader.stream_position()?;
        let version = u8::read_options(reader, endian, ())?;

        // Versions 1 and 2 put the dimensionality before the layout class.
        if version < 3 {
            return read_layout_v12(reader, endian, version);
        }
        if version > 5 {
            return Err(binrw::Error::AssertFail {
                pos,
                message: format!("unsupported data layout message version {version}"),
            });
        }

        let layout_class = u8::read_options(reader, endian, ())?;
        let inner = match layout_class {
            0 => {
                let size = u16::read_options(reader, endian, ())? as usize;
                let mut data = vec![0u8; size];
                reader.read_exact(&mut data)?;
                LayoutInner::Compact { data }
            }
            1 => LayoutInner::Contiguous {
                address: u64::read_options(reader, endian, ())?,
                size: u64::read_options(reader, endian, ())?,
            },
            2 if version == 3 => LayoutInner::Chunked(read_chunked_v3(reader, endian)?),
            2 => LayoutInner::Chunked(read_chunked_v4(reader, endian)?),
            other => LayoutInner::Unsupported(other),
        };

        Ok(DataLayoutMessage {
            version,
            layout_class,
            inner,
        })
    }
}

/// Versions 1 and 2: `dimensionality`, then the layout class.
fn read_layout_v12<R: std::io::Read + std::io::Seek>(
    reader: &mut R,
    endian: binrw::Endian,
    version: u8,
) -> BinResult<DataLayoutMessage> {
    let dimensionality = u8::read_options(reader, endian, ())?;
    let layout_class = u8::read_options(reader, endian, ())?;
    reader.seek(std::io::SeekFrom::Current(5))?; // reserved

    let inner = match layout_class {
        0 => {
            // Compact: the dimension sizes come first, then the data length.
            let mut dims = Vec::new();
            for _ in 0..dimensionality {
                dims.push(u32::read_options(reader, endian, ())?);
            }
            let size = u32::read_options(reader, endian, ())? as usize;
            let mut data = vec![0u8; size];
            reader.read_exact(&mut data)?;
            LayoutInner::Compact { data }
        }
        1 => {
            let address = u64::read_options(reader, endian, ())?;
            let mut dims: Vec<u64> = Vec::new();
            for _ in 0..dimensionality {
                dims.push(u32::read_options(reader, endian, ())? as u64);
            }
            let size = dims.iter().product::<u64>();
            LayoutInner::Contiguous { address, size }
        }
        2 => {
            let address = u64::read_options(reader, endian, ())?;
            let mut dims = Vec::new();
            for _ in 0..dimensionality {
                dims.push(u32::read_options(reader, endian, ())? as u64);
            }
            LayoutInner::Chunked(split_chunk_dims(dims, address)?)
        }
        other => LayoutInner::Unsupported(other),
    };

    Ok(DataLayoutMessage {
        version,
        layout_class,
        inner,
    })
}

/// Version 3 chunked properties: dimensionality, B-tree address, then
/// `dimensionality` 4-byte extents whose last entry is the element size.
fn read_chunked_v3<R: std::io::Read + std::io::Seek>(
    reader: &mut R,
    endian: binrw::Endian,
) -> BinResult<ChunkedLayout> {
    let dimensionality = u8::read_options(reader, endian, ())?;
    let address = u64::read_options(reader, endian, ())?;
    let mut dims = Vec::with_capacity(dimensionality as usize);
    for _ in 0..dimensionality {
        dims.push(u32::read_options(reader, endian, ())? as u64);
    }
    split_chunk_dims(dims, address)
}

/// Split the on-disk extent list, whose final entry is the element size rather
/// than a chunk dimension, into its two parts.
fn split_chunk_dims(mut dims: Vec<u64>, address: u64) -> BinResult<ChunkedLayout> {
    let Some(element_size) = dims.pop() else {
        return Err(binrw::Error::AssertFail {
            pos: 0,
            message: "chunked layout with no dimensions".to_string(),
        });
    };
    Ok(ChunkedLayout {
        chunk_dims: dims,
        element_size: element_size as u32,
        index: ChunkIndex::BTreeV1 { address },
    })
}

/// Versions 4 and 5 chunked properties, which carry the chunk index type.
fn read_chunked_v4<R: std::io::Read + std::io::Seek>(
    reader: &mut R,
    endian: binrw::Endian,
) -> BinResult<ChunkedLayout> {
    let pos = reader.stream_position()?;
    let flags = u8::read_options(reader, endian, ())?;
    let dimensionality = u8::read_options(reader, endian, ())?;
    let dim_encoded_len = u8::read_options(reader, endian, ())?;

    if dim_encoded_len == 0 || dim_encoded_len > 8 {
        return Err(binrw::Error::AssertFail {
            pos,
            message: format!("invalid chunk dimension encoding length {dim_encoded_len}"),
        });
    }

    // As in version 3, the last extent is the element size.
    let mut dims = Vec::with_capacity(dimensionality as usize);
    for _ in 0..dimensionality {
        let mut buf = [0u8; 8];
        reader.read_exact(&mut buf[..dim_encoded_len as usize])?;
        dims.push(u64::from_le_bytes(buf));
    }
    let Some(element_size) = dims.pop() else {
        return Err(binrw::Error::AssertFail {
            pos,
            message: "chunked layout with no dimensions".to_string(),
        });
    };

    let index_type = u8::read_options(reader, endian, ())?;

    // Index-specific information, then the address, which is shared by all types.
    let single_filtered = if index_type == 1 && flags & 0x02 != 0 {
        // SINGLE_INDEX_WITH_FILTER: filtered size and mask precede the address.
        let size = u64::read_options(reader, endian, ())?;
        let mask = u32::read_options(reader, endian, ())?;
        Some((size, mask))
    } else {
        match index_type {
            3 => {
                reader.seek(std::io::SeekFrom::Current(1))?; // page bits
                None
            }
            4 => {
                reader.seek(std::io::SeekFrom::Current(5))?; // max bits .. page bits
                None
            }
            5 => {
                reader.seek(std::io::SeekFrom::Current(6))?; // node size, split %, merge %
                None
            }
            _ => None,
        }
    };

    let address = u64::read_options(reader, endian, ())?;
    let index = match index_type {
        1 => ChunkIndex::SingleChunk {
            address,
            filtered: single_filtered,
        },
        2 => ChunkIndex::Implicit { address },
        3 => ChunkIndex::FixedArray { address },
        4 => ChunkIndex::ExtensibleArray { address },
        5 => ChunkIndex::BTreeV2 { address },
        other => {
            return Err(binrw::Error::AssertFail {
                pos,
                message: format!("unknown chunk indexing type {other}"),
            });
        }
    };

    Ok(ChunkedLayout {
        chunk_dims: dims,
        element_size: element_size as u32,
        index,
    })
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

/// Filter Pipeline message, version 1 or 2.
///
/// Version 2 drops the reserved padding, omits the name for the filters defined
/// by the format specification, and no longer pads the client data to an even
/// number of values.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct FilterMessage {
    pub version: u8,
    pub num_filters: u8,
    pub filters: Vec<FilterDescription>,
}

/// One filter in the pipeline.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct FilterDescription {
    pub filter_type: FilterType,
    pub flags: u16,
    pub client_data: Vec<u32>,
}

impl BinRead for FilterMessage {
    type Args<'a> = ();

    fn read_options<R: std::io::Read + std::io::Seek>(
        reader: &mut R,
        endian: binrw::Endian,
        _: Self::Args<'_>,
    ) -> BinResult<Self> {
        let pos = reader.stream_position()?;
        let version = u8::read_options(reader, endian, ())?;
        let num_filters = u8::read_options(reader, endian, ())?;

        match version {
            1 => reader.seek(std::io::SeekFrom::Current(6))?,
            2 => 0,
            v => {
                return Err(binrw::Error::AssertFail {
                    pos,
                    message: format!("unsupported filter pipeline message version {v}"),
                });
            }
        };

        let mut filters = Vec::with_capacity(num_filters as usize);
        for _ in 0..num_filters {
            let id = u16::read_options(reader, endian, ())?;

            // Version 2 stores a name only for filters outside the range
            // defined by the format specification.
            let name_length = if version == 1 || id >= 256 {
                u16::read_options(reader, endian, ())?
            } else {
                0
            };
            let flags = u16::read_options(reader, endian, ())?;
            let num_values = u16::read_options(reader, endian, ())?;

            if name_length > 0 {
                reader.seek(std::io::SeekFrom::Current(name_length as i64))?;
            }

            let mut client_data = Vec::with_capacity(num_values as usize);
            for _ in 0..num_values {
                client_data.push(u32::read_options(reader, endian, ())?);
            }
            // Version 1 pads the client data to an even number of values.
            if version == 1 && num_values % 2 == 1 {
                reader.seek(std::io::SeekFrom::Current(4))?;
            }

            filters.push(FilterDescription {
                filter_type: FilterType::from(id),
                flags,
                client_data,
            });
        }

        Ok(FilterMessage {
            version,
            num_filters,
            filters,
        })
    }
}

/// Attribute message, versions 1 through 3.
///
/// The raw data length is taken from the enclosing message rather than computed
/// from the datatype, so an attribute whose datatype class h5rs does not
/// interpret still parses and only fails when its value is read.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct AttributeMessage {
    pub version: u8,
    pub name: String,
    pub datatype: DatatypeMessage,
    pub dataspace: DataspaceMessage,
    pub data: Vec<u8>,
}

impl BinRead for AttributeMessage {
    /// The total size of the attribute message, used to bound the raw data.
    type Args<'a> = (u16,);

    fn read_options<R: std::io::Read + std::io::Seek>(
        reader: &mut R,
        endian: binrw::Endian,
        (message_size,): Self::Args<'_>,
    ) -> BinResult<Self> {
        let start = reader.stream_position()?;
        let version = u8::read_options(reader, endian, ())?;
        let flags = u8::read_options(reader, endian, ())?;
        let name_size = u16::read_options(reader, endian, ())?;
        let datatype_size = u16::read_options(reader, endian, ())?;
        let dataspace_size = u16::read_options(reader, endian, ())?;

        match version {
            // Version 1's `flags` byte is reserved, and its name, datatype and
            // dataspace are each padded out to a multiple of eight bytes.
            1 => {}
            2 => {}
            3 => {
                reader.seek(std::io::SeekFrom::Current(1))?; // name character set
            }
            v => {
                return Err(binrw::Error::AssertFail {
                    pos: start,
                    message: format!("unsupported attribute message version {v}"),
                });
            }
        }

        if version > 1 && flags & 0x03 != 0 {
            return Err(binrw::Error::AssertFail {
                pos: start,
                message: "attribute with a shared datatype or dataspace".to_string(),
            });
        }

        let pad = |n: u16| {
            if version == 1 {
                n.next_multiple_of(8)
            } else {
                n
            }
        };

        let field_start = reader.stream_position()?;
        let mut name_bytes = vec![0u8; name_size as usize];
        reader.read_exact(&mut name_bytes)?;
        // The stored length includes the null terminator.
        let name = String::from_utf8_lossy(name_bytes.split(|&b| b == 0).next().unwrap_or(&[]))
            .into_owned();
        reader.seek(std::io::SeekFrom::Start(
            field_start + pad(name_size) as u64,
        ))?;

        let field_start = reader.stream_position()?;
        let datatype = DatatypeMessage::read_options(reader, endian, ())?;
        reader.seek(std::io::SeekFrom::Start(
            field_start + pad(datatype_size) as u64,
        ))?;

        let field_start = reader.stream_position()?;
        let dataspace = DataspaceMessage::read_options(reader, endian, ())?;
        reader.seek(std::io::SeekFrom::Start(
            field_start + pad(dataspace_size) as u64,
        ))?;

        // Prefer the length implied by the datatype and dataspace, and fall back
        // to whatever remains of the message when the datatype class is one h5rs
        // cannot size. Version 1 messages are padded out to a multiple of eight
        // bytes, so the remainder is an upper bound, not the exact length.
        let consumed = reader.stream_position()? - start;
        let remaining = (message_size as u64).saturating_sub(consumed);
        let data_len = match datatype.element_size() {
            Ok(elem) => ((elem * dataspace.num_elements()) as u64).min(remaining),
            Err(_) => remaining,
        };
        let mut data = vec![0u8; data_len as usize];
        reader.read_exact(&mut data)?;

        Ok(AttributeMessage {
            version,
            name,
            datatype,
            dataspace,
            data,
        })
    }
}

/// Filter identifiers defined by the HDF5 format specification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FilterType {
    None,
    Deflate,
    Shuffle,
    Fletcher32,
    Szip,
    Nbit,
    ScaleOffset,
    /// A filter registered outside the format specification.
    Other(u16),
}

impl From<u16> for FilterType {
    fn from(id: u16) -> Self {
        match id {
            0 => FilterType::None,
            1 => FilterType::Deflate,
            2 => FilterType::Shuffle,
            3 => FilterType::Fletcher32,
            4 => FilterType::Szip,
            5 => FilterType::Nbit,
            6 => FilterType::ScaleOffset,
            other => FilterType::Other(other),
        }
    }
}

#[allow(dead_code)]
impl AttributeMessage {
    pub fn name(&self) -> String {
        self.name.clone()
    }

    pub fn read<T: crate::h5type::H5Type>(&self) -> H5Result<Vec<T>> {
        T::check_dtype(&self.datatype)?;
        Ok(bytemuck::cast_slice(&self.data).to_vec())
    }

    pub fn read_strings(&self) -> H5Result<Vec<String>> {
        let TypeDescriptor::String(ref sd) = self.datatype.type_desc else {
            return Err(H5Error::type_mismatch::<String>(format!(
                "{:?}",
                self.datatype.type_desc
            )));
        };
        let elem_size = sd.size() as usize;
        if elem_size == 0 {
            return Err(H5Error::corrupt("string datatype with zero element size"));
        }
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
                    p => {
                        return Err(H5Error::unsupported(format!("string padding type {p}")));
                    }
                };
                Ok(String::from_utf8_lossy(trimmed).into_owned())
            })
            .collect()
    }
}
