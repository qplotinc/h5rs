//! Locating the chunks of a chunked dataset.
//!
//! Files written before HDF5 1.10 always index chunks with a version 1 B-tree.
//! Newer files pick one of five structures depending on the dataset's shape and
//! filters; this module enumerates the chunks of any of them into a common
//! [`ChunkRecord`] list.
//!
//! <https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#sec_fmt4_appendixc>

use std::io::Cursor;

use binrw::BinRead;

use crate::error::{H5Error, H5Result};
use crate::format::btree::collect_btree_leaves_args;
use crate::format::btree2::BTreeV2Header;
use crate::format::metadata::ChunkBTreeV1;
use crate::format::object::{ChunkIndex, ChunkedLayout, is_undefined_address};
use crate::object_store::{ObjectStoreFile, fetch_exact};

/// One chunk of a chunked dataset, as located through the dataset's index.
#[derive(Debug, Clone)]
pub struct ChunkRecord {
    /// Address of the chunk's bytes, as stored (filtered, if filters apply).
    pub address: u64,
    /// Stored size in bytes.
    pub size: u64,
    /// Bitmask of pipeline filters skipped for this chunk.
    pub filter_mask: u32,
    /// Offset of the chunk within the dataset, in elements, outermost
    /// dimension first.
    pub offsets: Vec<u64>,
}

/// Signature-and-prefix overhead common to the array block structures:
/// 4-byte signature, version, client ID, and a trailing 4-byte checksum.
const BLOCK_OVERHEAD: u64 = 10;

/// Enumerate every allocated chunk of a chunked dataset.
///
/// `dataset_dims` is the dataset's current extent, needed to turn a linear
/// chunk index back into per-dimension offsets.
pub async fn enumerate_chunks(
    file: &ObjectStoreFile,
    layout: &ChunkedLayout,
    dataset_dims: &[u64],
) -> H5Result<Vec<ChunkRecord>> {
    if is_undefined_address(layout.index.address()) {
        // Storage has not been allocated; the dataset reads as its fill value.
        return Ok(vec![]);
    }

    match &layout.index {
        ChunkIndex::BTreeV1 { address } => read_btree_v1(file, *address, layout).await,
        ChunkIndex::SingleChunk { address, filtered } => Ok(vec![ChunkRecord {
            address: *address,
            size: filtered.map_or_else(|| layout.chunk_bytes(), |(size, _)| size),
            filter_mask: filtered.map_or(0, |(_, mask)| mask),
            offsets: vec![0; layout.ndim()],
        }]),
        ChunkIndex::Implicit { address } => Ok(read_implicit(*address, layout, dataset_dims)),
        ChunkIndex::FixedArray { address } => {
            read_fixed_array(file, *address, layout, dataset_dims).await
        }
        ChunkIndex::ExtensibleArray { address } => {
            read_extensible_array(file, *address, layout, dataset_dims).await
        }
        ChunkIndex::BTreeV2 { address } => {
            read_btree_v2(file, *address, layout, dataset_dims).await
        }
    }
}

/// Number of chunks along each dimension.
fn chunk_counts(dataset_dims: &[u64], chunk_dims: &[u64]) -> Vec<u64> {
    dataset_dims
        .iter()
        .zip(chunk_dims)
        .map(|(&d, &c)| if c == 0 { 0 } else { d.div_ceil(c) })
        .collect()
}

/// Strides for converting between a linear chunk index and per-dimension chunk
/// coordinates. The fastest-changing dimension is last, as in C order.
fn down_chunks(counts: &[u64]) -> Vec<u64> {
    let mut down = vec![1u64; counts.len()];
    for i in (0..counts.len().saturating_sub(1)).rev() {
        down[i] = down[i + 1].saturating_mul(counts[i + 1]);
    }
    down
}

/// Invert the linear chunk index used by the array and implicit indexes.
fn offsets_for_index(index: u64, counts: &[u64], down: &[u64], chunk_dims: &[u64]) -> Vec<u64> {
    (0..counts.len())
        .map(|d| {
            if down[d] == 0 || counts[d] == 0 {
                0
            } else {
                ((index / down[d]) % counts[d]) * chunk_dims[d]
            }
        })
        .collect()
}

/// A version 1 B-tree stores each chunk's offsets in its keys directly.
async fn read_btree_v1(
    file: &ObjectStoreFile,
    address: u64,
    layout: &ChunkedLayout,
) -> H5Result<Vec<ChunkRecord>> {
    let ndim = layout.ndim();
    let btree: ChunkBTreeV1 =
        crate::object_store::read_metadata_args(file, address, (ndim as u8,)).await?;
    let leaves = collect_btree_leaves_args(file, btree).await?;

    Ok(leaves
        .into_iter()
        .map(|c| ChunkRecord {
            address: c.child_pointer,
            size: c.key.chunk_size as u64,
            filter_mask: c.key.filter_mask,
            // The stored key carries one trailing offset for the element
            // dimension, which is always zero and not a dataset dimension.
            offsets: c.key.offsets[..ndim].to_vec(),
        })
        .collect())
}

/// Implicit indexes store no records at all: chunk *i* lives at
/// `base + i * chunk_bytes`.
fn read_implicit(address: u64, layout: &ChunkedLayout, dataset_dims: &[u64]) -> Vec<ChunkRecord> {
    let counts = chunk_counts(dataset_dims, &layout.chunk_dims);
    let down = down_chunks(&counts);
    let total: u64 = counts.iter().product();
    let chunk_bytes = layout.chunk_bytes();

    (0..total)
        .map(|i| ChunkRecord {
            address: address + i * chunk_bytes,
            size: chunk_bytes,
            filter_mask: 0,
            offsets: offsets_for_index(i, &counts, &down, &layout.chunk_dims),
        })
        .collect()
}

/// How a chunk record is encoded inside an array index element.
#[derive(Debug, Clone, Copy)]
struct ElementFormat {
    /// True when elements carry a stored size and filter mask.
    filtered: bool,
    /// Width in bytes of the stored-size field.
    size_len: usize,
    /// Total element width.
    entry_size: usize,
}

impl ElementFormat {
    /// Derive the element layout from the entry size recorded in the index
    /// header, rather than recomputing the library's sizing rule.
    fn new(client_id: u8, entry_size: usize, what: &str) -> H5Result<ElementFormat> {
        match client_id {
            0 => {
                if entry_size != 8 {
                    return Err(H5Error::corrupt(format!(
                        "{what} unfiltered entry size is {entry_size}, expected 8"
                    )));
                }
                Ok(ElementFormat {
                    filtered: false,
                    size_len: 0,
                    entry_size,
                })
            }
            1 => {
                // address + variable-width size + 4-byte filter mask
                if entry_size <= 12 || entry_size > 20 {
                    return Err(H5Error::corrupt(format!(
                        "{what} filtered entry size {entry_size} is out of range"
                    )));
                }
                Ok(ElementFormat {
                    filtered: true,
                    size_len: entry_size - 12,
                    entry_size,
                })
            }
            other => Err(H5Error::unsupported(format!(
                "{what} client ID {other} (only dataset chunks are implemented)"
            ))),
        }
    }

    /// Decode one element. Returns `None` for a chunk that has no storage.
    fn decode(&self, bytes: &[u8], chunk_bytes: u64) -> Option<(u64, u64, u32)> {
        let address = u64::from_le_bytes(bytes[..8].try_into().ok()?);
        if is_undefined_address(address) || address == 0 {
            return None;
        }
        if !self.filtered {
            return Some((address, chunk_bytes, 0));
        }
        let mut size_buf = [0u8; 8];
        size_buf[..self.size_len].copy_from_slice(&bytes[8..8 + self.size_len]);
        let size = u64::from_le_bytes(size_buf);
        let mask = u32::from_le_bytes(bytes[8 + self.size_len..self.entry_size].try_into().ok()?);
        Some((address, size, mask))
    }
}

/// Fixed Array index header.
#[derive(BinRead, Debug)]
#[br(magic = b"FAHD")]
#[allow(dead_code)]
struct FixedArrayHeader {
    #[br(assert(version == 0, "unsupported fixed array version {}", version))]
    version: u8,
    client_id: u8,
    entry_size: u8,
    page_bits: u8,
    max_num_entries: u64,
    data_block_address: u64,
    checksum: u32,
}

/// A Fixed Array holds one element per chunk of the dataset, in linear chunk
/// index order, optionally split into fixed-size pages.
async fn read_fixed_array(
    file: &ObjectStoreFile,
    address: u64,
    layout: &ChunkedLayout,
    dataset_dims: &[u64],
) -> H5Result<Vec<ChunkRecord>> {
    let header: FixedArrayHeader = crate::object_store::read_metadata(file, address).await?;
    if is_undefined_address(header.data_block_address) {
        return Ok(vec![]);
    }
    let format = ElementFormat::new(header.client_id, header.entry_size as usize, "fixed array")?;

    let nelmts = header.max_num_entries;
    let page_nelmts = 1u64 << header.page_bits;
    let paged = nelmts > page_nelmts;

    // Prefix of the data block: signature, version, client ID, header address.
    const DBLK_PREFIX: u64 = 4 + 1 + 1 + 8;

    let elements = if !paged {
        let block_size = DBLK_PREFIX + nelmts * format.entry_size as u64 + 4;
        let bytes = fetch_exact(file, header.data_block_address, block_size).await?;
        check_signature(&bytes, b"FADB", "fixed array data block")?;
        bytes[DBLK_PREFIX as usize..].to_vec()
    } else {
        // The data block holds a bitmap of initialised pages; the pages
        // themselves follow it contiguously, each with its own checksum.
        let npages = nelmts.div_ceil(page_nelmts);
        let bitmap_bytes = npages.div_ceil(8);
        let dblk_size = DBLK_PREFIX + bitmap_bytes + 4;
        let page_stride = page_nelmts * format.entry_size as u64 + 4;
        let total = dblk_size + npages * page_stride;

        let bytes = fetch_exact(file, header.data_block_address, total).await?;
        check_signature(&bytes, b"FADB", "fixed array data block")?;
        let bitmap = &bytes[DBLK_PREFIX as usize..(DBLK_PREFIX + bitmap_bytes) as usize];

        let mut elements = Vec::with_capacity((nelmts * format.entry_size as u64) as usize);
        for page in 0..npages {
            let in_page = page_nelmts.min(nelmts - page * page_nelmts);
            // The library stores these bits most-significant first within
            // each byte.
            let initialised = bitmap
                .get((page / 8) as usize)
                .is_some_and(|b| b & (0x80 >> (page % 8)) != 0);
            if initialised {
                let start = (dblk_size + page * page_stride) as usize;
                let len = (in_page * format.entry_size as u64) as usize;
                elements.extend_from_slice(&bytes[start..start + len]);
            } else {
                // An uninitialised page has no allocated chunks.
                elements.extend(std::iter::repeat_n(
                    0xffu8,
                    (in_page * format.entry_size as u64) as usize,
                ));
            }
        }
        elements
    };

    Ok(decode_elements(&elements, &format, 0, layout, dataset_dims))
}

/// Turn a run of index elements starting at chunk index `start` into records.
fn decode_elements(
    elements: &[u8],
    format: &ElementFormat,
    start: u64,
    layout: &ChunkedLayout,
    dataset_dims: &[u64],
) -> Vec<ChunkRecord> {
    let counts = chunk_counts(dataset_dims, &layout.chunk_dims);
    let down = down_chunks(&counts);
    let chunk_bytes = layout.chunk_bytes();

    elements
        .chunks_exact(format.entry_size)
        .enumerate()
        .filter_map(|(i, raw)| {
            let (address, size, filter_mask) = format.decode(raw, chunk_bytes)?;
            Some(ChunkRecord {
                address,
                size,
                filter_mask,
                offsets: offsets_for_index(start + i as u64, &counts, &down, &layout.chunk_dims),
            })
        })
        .collect()
}

fn check_signature(bytes: &[u8], expected: &[u8; 4], what: &str) -> H5Result<()> {
    if bytes.len() < 4 || &bytes[..4] != expected {
        return Err(H5Error::corrupt(format!(
            "expected a {what} signature ({}) here",
            String::from_utf8_lossy(expected)
        )));
    }
    Ok(())
}

/// Extensible Array index header.
#[derive(BinRead, Debug)]
#[br(magic = b"EAHD")]
#[allow(dead_code)]
struct ExtensibleArrayHeader {
    #[br(assert(version == 0, "unsupported extensible array version {}", version))]
    version: u8,
    client_id: u8,
    element_size: u8,
    max_nelmts_bits: u8,
    index_blk_elmts: u8,
    data_blk_min_elmts: u8,
    sup_blk_min_data_ptrs: u8,
    max_dblk_page_nelmts_bits: u8,
    num_secondary_blks: u64,
    secondary_blk_size: u64,
    num_data_blks: u64,
    data_blk_size: u64,
    max_index_set: u64,
    num_elements: u64,
    index_block_address: u64,
    checksum: u32,
}

/// Geometry of one "super block" of the extensible array: a run of data blocks
/// that all hold the same number of elements.
#[derive(Debug, Clone, Copy)]
struct SuperBlockInfo {
    /// Number of data blocks in this super block.
    ndblks: u64,
    /// Elements held by each of those data blocks.
    dblk_nelmts: u64,
    /// Array index of the first element covered by this super block.
    start_idx: u64,
}

/// Reproduce the extensible array's pre-computed super block table. Block sizes
/// double every second super block, which is what makes lookups constant-depth.
fn super_block_table(header: &ExtensibleArrayHeader) -> H5Result<Vec<SuperBlockInfo>> {
    let min_elmts = header.data_blk_min_elmts as u64;
    if min_elmts == 0 || !min_elmts.is_power_of_two() {
        return Err(H5Error::corrupt(
            "extensible array minimum data block elements is not a power of two",
        ));
    }
    let nsblks = 1 + header.max_nelmts_bits as u64 - min_elmts.trailing_zeros() as u64;

    let mut table = Vec::with_capacity(nsblks as usize);
    let mut start_idx = 0u64;
    for u in 0..nsblks {
        let info = SuperBlockInfo {
            ndblks: 1u64 << (u / 2),
            dblk_nelmts: (1u64 << u.div_ceil(2)) * min_elmts,
            start_idx,
        };
        start_idx += info.ndblks * info.dblk_nelmts;
        table.push(info);
    }
    Ok(table)
}

/// Walk an Extensible Array: the index block holds the first few elements and
/// pointers to the first data blocks, then secondary blocks fan out from there.
async fn read_extensible_array(
    file: &ObjectStoreFile,
    address: u64,
    layout: &ChunkedLayout,
    dataset_dims: &[u64],
) -> H5Result<Vec<ChunkRecord>> {
    let header: ExtensibleArrayHeader = crate::object_store::read_metadata(file, address).await?;
    if is_undefined_address(header.index_block_address) {
        return Ok(vec![]);
    }
    let format = ElementFormat::new(
        header.client_id,
        header.element_size as usize,
        "extensible array",
    )?;
    let sblks = super_block_table(&header)?;

    let min_ptrs = header.sup_blk_min_data_ptrs as u64;
    if min_ptrs == 0 || !min_ptrs.is_power_of_two() {
        return Err(H5Error::corrupt(
            "extensible array minimum super block data pointers is not a power of two",
        ));
    }
    // Super blocks whose data blocks the index block points at directly.
    let direct_sblks = 2 * min_ptrs.trailing_zeros() as u64;
    let ndblk_addrs = 2 * (min_ptrs - 1);
    let nsblk_addrs = (sblks.len() as u64).saturating_sub(direct_sblks);

    // Index block: prefix, inline elements, direct data block addresses,
    // secondary block addresses, checksum.
    let idx_elmts = header.index_blk_elmts as u64;
    let iblock_size =
        BLOCK_OVERHEAD + 8 + idx_elmts * format.entry_size as u64 + (ndblk_addrs + nsblk_addrs) * 8;
    let bytes = fetch_exact(file, header.index_block_address, iblock_size).await?;
    check_signature(&bytes, b"EAIB", "extensible array index block")?;

    let mut pos = 4 + 1 + 1 + 8; // signature, version, client ID, header address
    let mut records = decode_elements(
        &bytes[pos..pos + (idx_elmts * format.entry_size as u64) as usize],
        &format,
        0,
        layout,
        dataset_dims,
    );
    pos += (idx_elmts * format.entry_size as u64) as usize;

    let direct_addrs = read_addresses(&bytes[pos..], ndblk_addrs as usize)?;
    pos += ndblk_addrs as usize * 8;
    let secondary_addrs = read_addresses(&bytes[pos..], nsblk_addrs as usize)?;

    // The directly-pointed data blocks are the first `ndblk_addrs` blocks of the
    // array, taken in super block order.
    for (dblk_index, &addr) in direct_addrs.iter().enumerate() {
        let (sblk, within) = locate_data_block(&sblks, dblk_index as u64);
        let start_idx = idx_elmts + sblk.start_idx + within * sblk.dblk_nelmts;
        if !is_undefined_address(addr) && addr != 0 {
            records.extend(
                read_ea_data_block(
                    file,
                    addr,
                    &format,
                    sblk.dblk_nelmts,
                    start_idx,
                    &header,
                    layout,
                    dataset_dims,
                )
                .await?,
            );
        }
    }

    for (i, &addr) in secondary_addrs.iter().enumerate() {
        if is_undefined_address(addr) || addr == 0 {
            continue;
        }
        let sblk_idx = direct_sblks as usize + i;
        let Some(sblk) = sblks.get(sblk_idx) else {
            break;
        };
        records.extend(
            read_ea_secondary_block(
                file,
                addr,
                &format,
                sblk,
                idx_elmts,
                &header,
                layout,
                dataset_dims,
            )
            .await?,
        );
    }

    Ok(records)
}

/// Which super block (and which block within it) holds data block `n`.
fn locate_data_block(sblks: &[SuperBlockInfo], n: u64) -> (SuperBlockInfo, u64) {
    let mut remaining = n;
    for sblk in sblks {
        if remaining < sblk.ndblks {
            return (*sblk, remaining);
        }
        remaining -= sblk.ndblks;
    }
    (
        *sblks.last().unwrap_or(&SuperBlockInfo {
            ndblks: 1,
            dblk_nelmts: 1,
            start_idx: 0,
        }),
        0,
    )
}

fn read_addresses(bytes: &[u8], count: usize) -> H5Result<Vec<u64>> {
    if bytes.len() < count * 8 {
        return Err(H5Error::corrupt("truncated extensible array block"));
    }
    Ok((0..count)
        .map(|i| u64::from_le_bytes(bytes[i * 8..i * 8 + 8].try_into().unwrap()))
        .collect())
}

/// A secondary block points at a run of equally-sized data blocks.
#[allow(clippy::too_many_arguments)]
async fn read_ea_secondary_block(
    file: &ObjectStoreFile,
    address: u64,
    format: &ElementFormat,
    sblk: &SuperBlockInfo,
    idx_elmts: u64,
    header: &ExtensibleArrayHeader,
    layout: &ChunkedLayout,
    dataset_dims: &[u64],
) -> H5Result<Vec<ChunkRecord>> {
    let offset_bytes = (header.max_nelmts_bits as u64).div_ceil(8);
    let page_nelmts = 1u64 << header.max_dblk_page_nelmts_bits;
    let paged = sblk.dblk_nelmts > page_nelmts;
    let bitmap_bytes = if paged {
        (sblk.ndblks * sblk.dblk_nelmts.div_ceil(page_nelmts)).div_ceil(8)
    } else {
        0
    };

    let size = BLOCK_OVERHEAD + 8 + offset_bytes + bitmap_bytes + sblk.ndblks * 8;
    let bytes = fetch_exact(file, address, size).await?;
    check_signature(&bytes, b"EASB", "extensible array secondary block")?;

    let addr_start = (4 + 1 + 1 + 8 + offset_bytes + bitmap_bytes) as usize;
    let addrs = read_addresses(&bytes[addr_start..], sblk.ndblks as usize)?;

    let mut records = vec![];
    for (i, &addr) in addrs.iter().enumerate() {
        if is_undefined_address(addr) || addr == 0 {
            continue;
        }
        let start_idx = idx_elmts + sblk.start_idx + i as u64 * sblk.dblk_nelmts;
        records.extend(
            read_ea_data_block(
                file,
                addr,
                format,
                sblk.dblk_nelmts,
                start_idx,
                header,
                layout,
                dataset_dims,
            )
            .await?,
        );
    }
    Ok(records)
}

/// A data block holds `nelmts` consecutive elements, possibly split into pages
/// that follow the block contiguously.
#[allow(clippy::too_many_arguments)]
async fn read_ea_data_block(
    file: &ObjectStoreFile,
    address: u64,
    format: &ElementFormat,
    nelmts: u64,
    start_idx: u64,
    header: &ExtensibleArrayHeader,
    layout: &ChunkedLayout,
    dataset_dims: &[u64],
) -> H5Result<Vec<ChunkRecord>> {
    let offset_bytes = (header.max_nelmts_bits as u64).div_ceil(8);
    let prefix = 4 + 1 + 1 + 8 + offset_bytes;
    let page_nelmts = 1u64 << header.max_dblk_page_nelmts_bits;

    if nelmts <= page_nelmts {
        let size = prefix + nelmts * format.entry_size as u64 + 4;
        let bytes = fetch_exact(file, address, size).await?;
        check_signature(&bytes, b"EADB", "extensible array data block")?;
        let start = prefix as usize;
        let len = (nelmts * format.entry_size as u64) as usize;
        return Ok(decode_elements(
            &bytes[start..start + len],
            format,
            start_idx,
            layout,
            dataset_dims,
        ));
    }

    // Paged: the block itself holds only the prefix; pages follow it.
    let npages = nelmts.div_ceil(page_nelmts);
    let page_stride = page_nelmts * format.entry_size as u64 + 4;
    let size = prefix + npages * page_stride;
    let bytes = fetch_exact(file, address, size).await?;
    check_signature(&bytes, b"EADB", "extensible array data block")?;

    let mut records = vec![];
    for page in 0..npages {
        let in_page = page_nelmts.min(nelmts - page * page_nelmts);
        let start = (prefix + page * page_stride) as usize;
        let len = (in_page * format.entry_size as u64) as usize;
        records.extend(decode_elements(
            &bytes[start..start + len],
            format,
            start_idx + page * page_nelmts,
            layout,
            dataset_dims,
        ));
    }
    Ok(records)
}

/// A version 2 B-tree index stores each chunk's scaled offsets in its records.
async fn read_btree_v2(
    file: &ObjectStoreFile,
    address: u64,
    layout: &ChunkedLayout,
    _dataset_dims: &[u64],
) -> H5Result<Vec<ChunkRecord>> {
    let ndim = layout.ndim();
    let header = BTreeV2Header::read(file, address).await?;
    // Type 10 records are unfiltered; type 11 adds a stored size and filter
    // mask, whose width follows from the record size the header reports.
    let size_len = match header.btree_type {
        10 => None,
        11 => Some(
            (header.record_size as usize)
                .checked_sub(8 + 4 + 8 * ndim)
                .ok_or_else(|| H5Error::corrupt("filtered chunk B-tree record is too small"))?,
        ),
        other => {
            return Err(H5Error::corrupt(format!(
                "chunk index B-tree has record type {other}, expected 10 or 11"
            )));
        }
    };

    let chunk_bytes = layout.chunk_bytes();
    let records = header.collect_records(file).await?;
    records
        .iter()
        .map(|raw| decode_chunk_btree_record(raw, size_len, &layout.chunk_dims, chunk_bytes))
        .collect()
}

fn decode_chunk_btree_record(
    raw: &[u8],
    size_len: Option<usize>,
    chunk_dims: &[u64],
    chunk_bytes: u64,
) -> H5Result<ChunkRecord> {
    let mut cursor = Cursor::new(raw);
    let address = u64::read_le(&mut cursor)?;
    let (size, filter_mask) = match size_len {
        None => (chunk_bytes, 0),
        Some(size_len) => {
            let mut buf = [0u8; 8];
            buf[..size_len].copy_from_slice(
                raw.get(8..8 + size_len)
                    .ok_or_else(|| H5Error::corrupt("truncated chunk B-tree record"))?,
            );
            cursor.set_position((8 + size_len) as u64);
            let mask = u32::read_le(&mut cursor)?;
            (u64::from_le_bytes(buf), mask)
        }
    };

    // Records store scaled offsets: the chunk's offset divided by the chunk size.
    let mut offsets = Vec::with_capacity(chunk_dims.len());
    for &c in chunk_dims {
        offsets.push(u64::read_le(&mut cursor)? * c);
    }

    Ok(ChunkRecord {
        address,
        size,
        filter_mask,
        offsets,
    })
}
