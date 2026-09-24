#![allow(missing_docs)] // the item names are the docs here.
//! What a *format* reader needs beyond "read this numeric dataset by path",
//! and which AnnData needs in particular:
//!
//! - **objects with their attributes** ([`open_object`]): a group's children
//!   and every object's attributes, decoded — fixed and variable-length
//!   strings, integers, floats. AnnData's `encoding-type` / `_index` /
//!   `column-order` live here.
//! - **the global heap** ([`read_vl_strings`]): variable-length strings, in
//!   attributes and in datasets, are 16-byte references into global heap
//!   collections; h5py writes every Python `str` this way.
//! - **strings out of a dataset** ([`Dataset::read_strings`]), of either
//!   kind, built on the raw byte view in `dataset.rs`.

use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::ops::Range;

use crate::dataset::Dataset;
use crate::error::{H5Error, H5Result};
use crate::format::object::{AttributeMessage, DatatypeMessage, TypeDescriptor};
use crate::object_store::{ObjectStoreFile, fetch_data};
use crate::{File, Group, read_object_header, read_object_headers};

/// What an object is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObjectKind {
    Group,
    Dataset,
}

/// One child of a group.
#[derive(Debug, Clone)]
pub struct Child {
    pub name: String,
    pub kind: ObjectKind,
}

/// A decoded attribute value. Numbers come back as vectors even when
/// scalar; strings keep the scalar/array distinction because AnnData tells
/// `"csr_matrix"` from `["a", "b"]` by it.
#[derive(Debug, Clone, PartialEq)]
pub enum AttrValue {
    Str(String),
    Strs(Vec<String>),
    Ints(Vec<i64>),
    Floats(Vec<f64>),
    /// A datatype this layer does not decode; the raw element bytes.
    Raw(Vec<u8>),
}

impl AttrValue {
    pub fn as_str(&self) -> Option<&str> {
        match self {
            AttrValue::Str(s) => Some(s),
            AttrValue::Strs(v) if v.len() == 1 => Some(&v[0]),
            _ => None,
        }
    }

    pub fn as_strs(&self) -> Option<&[String]> {
        match self {
            AttrValue::Strs(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_ints(&self) -> Option<&[i64]> {
        match self {
            AttrValue::Ints(v) => Some(v),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Attribute {
    pub name: String,
    pub value: AttrValue,
}

/// An object, its attributes, and — for a group — its children.
#[derive(Debug, Clone)]
pub struct ObjectInfo {
    pub kind: ObjectKind,
    pub attributes: Vec<Attribute>,
    pub children: Vec<Child>,
}

impl ObjectInfo {
    pub fn attr(&self, name: &str) -> Option<&AttrValue> {
        self.attributes
            .iter()
            .find(|a| a.name == name)
            .map(|a| &a.value)
    }

    pub fn child(&self, name: &str) -> Option<&Child> {
        self.children.iter().find(|c| c.name == name)
    }
}

/// How a string datatype stores its values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StringKind {
    /// `width` bytes per element, padded.
    Fixed { width: usize },
    /// 16-byte global-heap references per element.
    Variable,
}

fn string_kind(dt: &DatatypeMessage) -> Option<StringKind> {
    match &dt.type_desc {
        TypeDescriptor::String(s) => Some(StringKind::Fixed {
            width: s.size() as usize,
        }),
        // A variable-length *string* is class 9 with the string type bits;
        // sequences of other types land here too, but nothing this crate is
        // asked to read stores those.
        TypeDescriptor::Variable(_) => Some(StringKind::Variable),
        _ => None,
    }
}

/// Split fixed-width string bytes into strings, trimming NUL/space padding.
pub fn fixed_strings(raw: &[u8], width: usize) -> Vec<String> {
    if width == 0 {
        return Vec::new();
    }
    raw.chunks(width)
        .map(|s| {
            let end = s
                .iter()
                .rposition(|&b| b != 0 && b != b' ')
                .map_or(0, |i| i + 1);
            String::from_utf8_lossy(&s[..end]).into_owned()
        })
        .collect()
}

// --- Navigation --------------------------------------------------------------

/// The object header at `path` (empty for the root).
async fn navigate(
    file: &ObjectStoreFile,
    path: &[&str],
) -> H5Result<Option<crate::format::object::DataObjectHeader>> {
    if path.is_empty() {
        let sb = crate::format::metadata::Superblock::read(file).await?;
        return Ok(Some(read_object_header(file, sb.root_group_address).await?));
    }
    let f = File::open(file).await?;
    let mut group: Group = f.root_group;
    for (i, segment) in path.iter().enumerate() {
        let Some(obj) = group.find_obj(segment, file).await? else {
            return Ok(None);
        };
        if i + 1 == path.len() {
            return Ok(Some(obj.header));
        }
        match obj.to_group(file).await? {
            Some(g) => group = g,
            None => return Ok(None),
        }
    }
    Ok(None)
}

/// The object at `path` (a slice of names; empty for the root), with its
/// attributes decoded and, for a group, its children listed.
pub async fn open_object(file: &ObjectStoreFile, path: &[&str]) -> H5Result<Option<ObjectInfo>> {
    let Some(header) = navigate(file, path).await? else {
        return Ok(None);
    };
    let mut attributes = Vec::new();
    for a in header.all_attributes(file).await? {
        attributes.push(Attribute {
            name: a.name.clone(),
            value: decode_attribute(file, &a).await?,
        });
    }
    let Some(group) = header.to_group(file).await? else {
        return Ok(Some(ObjectInfo {
            kind: ObjectKind::Dataset,
            attributes,
            children: Vec::new(),
        }));
    };
    let refs = group.object_refs(file).await?;
    let addresses: Vec<u64> = refs.iter().map(|(_, a)| *a).collect();
    let headers = read_object_headers(file, &addresses).await?;
    let mut children = Vec::new();
    for ((name, _), child) in refs.into_iter().zip(headers) {
        let kind = if child.to_group(file).await?.is_some() {
            ObjectKind::Group
        } else {
            ObjectKind::Dataset
        };
        children.push(Child { name, kind });
    }
    children.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(Some(ObjectInfo {
        kind: ObjectKind::Group,
        attributes,
        children,
    }))
}

async fn decode_attribute(file: &ObjectStoreFile, a: &AttributeMessage) -> H5Result<AttrValue> {
    let scalar = a.dataspace.dimensionality == 0;
    let n = a.dataspace.num_elements();
    Ok(match &a.datatype.type_desc {
        TypeDescriptor::String(s) => {
            let strs = fixed_strings(&a.data, s.size() as usize);
            if scalar {
                AttrValue::Str(strs.into_iter().next().unwrap_or_default())
            } else {
                AttrValue::Strs(strs)
            }
        }
        TypeDescriptor::Variable(_) => {
            let strs = read_vl_strings(file, &a.data).await?;
            if scalar {
                AttrValue::Str(strs.into_iter().next().unwrap_or_default())
            } else {
                AttrValue::Strs(strs)
            }
        }
        TypeDescriptor::FixedPoint(fp) => {
            let size = fp.size() as usize;
            let signed = fp.signed() != 0;
            AttrValue::Ints(
                a.data
                    .chunks(size.max(1))
                    .take(n)
                    .map(|b| int_from_le(b, signed))
                    .collect(),
            )
        }
        TypeDescriptor::Enumeration(e) => AttrValue::Ints(
            a.data
                .chunks((e.size() as usize).max(1))
                .take(n)
                .map(|b| int_from_le(b, false))
                .collect(),
        ),
        TypeDescriptor::FloatingPoint(fp) => {
            let size = fp.size() as usize;
            AttrValue::Floats(
                a.data
                    .chunks(size.max(1))
                    .take(n)
                    .map(|b| match size {
                        4 => f32::from_le_bytes(b.try_into().unwrap_or([0; 4])) as f64,
                        8 => f64::from_le_bytes(b.try_into().unwrap_or([0; 8])),
                        _ => f64::NAN,
                    })
                    .collect(),
            )
        }
        TypeDescriptor::UnimplementedTypeClass => AttrValue::Raw(a.data.clone()),
    })
}

fn int_from_le(b: &[u8], signed: bool) -> i64 {
    let mut buf = [0u8; 8];
    let n = b.len().min(8);
    buf[..n].copy_from_slice(&b[..n]);
    if signed && n > 0 && n < 8 && (b[n - 1] & 0x80) != 0 {
        for byte in &mut buf[n..] {
            *byte = 0xFF;
        }
    }
    i64::from_le_bytes(buf)
}

// --- The global heap ---------------------------------------------------------

/// One global heap collection: object index → bytes.
struct HeapCollection {
    objects: HashMap<u16, Vec<u8>>,
}

async fn read_heap_collection(file: &ObjectStoreFile, address: u64) -> H5Result<HeapCollection> {
    // Header: "GCOL", version (1), 3 reserved, collection size (u64, the
    // whole collection including this header).
    let head = fetch_data(file, address, 16).await?;
    if head.len() < 16 || &head[..4] != b"GCOL" {
        return Err(H5Error::corrupt(format!(
            "no global heap collection at {address:#x}"
        )));
    }
    let size = u64::from_le_bytes(head[8..16].try_into().expect("8 bytes"));
    let bytes = fetch_data(file, address, size).await?;
    let mut objects = HashMap::new();
    let mut at = 16usize;
    // Each object: index u16, reference count u16, 4 reserved, size u64,
    // data padded to a multiple of 8. Index 0 is the free-space object,
    // which ends the list.
    while at + 16 <= bytes.len() {
        let index = u16::from_le_bytes([bytes[at], bytes[at + 1]]);
        let obj_size =
            u64::from_le_bytes(bytes[at + 8..at + 16].try_into().expect("8 bytes")) as usize;
        if index == 0 {
            break;
        }
        let start = at + 16;
        let end = (start + obj_size).min(bytes.len());
        objects.insert(index, bytes[start..end].to_vec());
        at = start + obj_size.next_multiple_of(8);
    }
    Ok(HeapCollection { objects })
}

/// Decode variable-length string elements: `raw` holds 16 bytes per element
/// — length (u32), heap collection address (u64), object index (u32).
pub async fn read_vl_strings(file: &ObjectStoreFile, raw: &[u8]) -> H5Result<Vec<String>> {
    let mut collections: HashMap<u64, HeapCollection> = HashMap::new();
    let mut out = Vec::with_capacity(raw.len() / 16);
    for e in raw.chunks_exact(16) {
        let len = u32::from_le_bytes(e[0..4].try_into().expect("4 bytes")) as usize;
        let address = u64::from_le_bytes(e[4..12].try_into().expect("8 bytes"));
        let index = u32::from_le_bytes(e[12..16].try_into().expect("4 bytes")) as u16;
        if len == 0 || address == 0 || address == u64::MAX {
            out.push(String::new());
            continue;
        }
        let collection = match collections.entry(address) {
            Entry::Occupied(e) => e.into_mut(),
            Entry::Vacant(e) => e.insert(read_heap_collection(file, address).await?),
        };
        let bytes = collection.objects.get(&index).ok_or_else(|| {
            H5Error::corrupt(format!(
                "global heap object {index} missing from collection at {address:#x}"
            ))
        })?;
        let s = &bytes[..len.min(bytes.len())];
        out.push(String::from_utf8_lossy(s).into_owned());
    }
    Ok(out)
}

// --- Strings out of a dataset ------------------------------------------------

impl Dataset {
    /// How this dataset's strings are stored, if it is a string dataset.
    pub fn string_kind(&self) -> Option<StringKind> {
        string_kind(&self.datatype)
    }

    /// Every element as a string, for a string dataset of either kind;
    /// `None` when the dataset is not strings.
    pub async fn read_strings(&self, file: &ObjectStoreFile) -> H5Result<Option<Vec<String>>> {
        let Some(kind) = self.string_kind() else {
            return Ok(None);
        };
        let sel: Vec<Range<u64>> = self.shape().iter().map(|&d| 0..d).collect();
        let (bytes, _) = self.read_range_bytes(&sel, file).await?;
        Ok(Some(match kind {
            StringKind::Fixed { width } => fixed_strings(&bytes, width),
            StringKind::Variable => read_vl_strings(file, &bytes).await?,
        }))
    }
}
