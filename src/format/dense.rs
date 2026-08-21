//! Densely stored links and attributes.
//!
//! Once a group has more links (or an object more attributes) than fit
//! comfortably in its object header, HDF5 moves them into a fractal heap and
//! indexes them with a version 2 B-tree. Reading them back means walking the
//! B-tree for heap IDs and dereferencing each one.

use std::io::Cursor;

use binrw::BinRead;

use crate::error::{H5Error, H5Result};
use crate::format::btree2::BTreeV2Header;
use crate::format::fractal_heap::FractalHeap;
use crate::format::object::{AttributeMessage, LinkMessage};
use crate::object_store::ObjectStoreFile;

/// Read every link of a group that stores its links densely.
pub async fn read_links(
    file: &ObjectStoreFile,
    fractal_heap_address: u64,
    name_btree_address: u64,
) -> H5Result<Vec<LinkMessage>> {
    // Type 5 records are a 4-byte name hash followed by a 7-byte heap ID.
    let ids = heap_ids(file, name_btree_address, 5, 4, 7).await?;
    let heap = FractalHeap::open(file, fractal_heap_address).await?;

    let mut links = Vec::with_capacity(ids.len());
    for id in ids {
        let bytes = heap.read_object(file, &id).await?;
        let mut cursor = Cursor::new(&bytes);
        links.push(LinkMessage::read_le(&mut cursor)?);
    }
    Ok(links)
}

/// Read every attribute of an object that stores its attributes densely.
pub async fn read_attributes(
    file: &ObjectStoreFile,
    fractal_heap_address: u64,
    name_btree_address: u64,
) -> H5Result<Vec<AttributeMessage>> {
    // Type 8 records start with an 8-byte heap ID.
    let ids = heap_ids(file, name_btree_address, 8, 0, 8).await?;
    let heap = FractalHeap::open(file, fractal_heap_address).await?;

    let mut attributes = Vec::with_capacity(ids.len());
    for id in ids {
        let bytes = heap.read_object(file, &id).await?;
        let mut cursor = Cursor::new(&bytes);
        attributes.push(AttributeMessage::read_le_args(
            &mut cursor,
            (bytes.len() as u16,),
        )?);
    }
    Ok(attributes)
}

/// Collect the fractal heap IDs embedded in a name-index B-tree's records.
async fn heap_ids(
    file: &ObjectStoreFile,
    btree_address: u64,
    expected_type: u8,
    id_offset: usize,
    id_len: usize,
) -> H5Result<Vec<Vec<u8>>> {
    let header = BTreeV2Header::read(file, btree_address).await?;
    if header.btree_type != expected_type {
        return Err(H5Error::corrupt(format!(
            "expected a type {expected_type} v2 B-tree, found type {}",
            header.btree_type
        )));
    }

    header
        .collect_records(file)
        .await?
        .into_iter()
        .map(|record| {
            record
                .get(id_offset..id_offset + id_len)
                .map(<[u8]>::to_vec)
                .ok_or_else(|| H5Error::corrupt("v2 B-tree record is too short for its heap ID"))
        })
        .collect()
}
