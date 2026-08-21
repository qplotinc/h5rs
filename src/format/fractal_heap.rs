//! Fractal heaps.
//!
//! A fractal heap stores the link messages of a densely-indexed group and the
//! attribute messages of an object with many attributes. Objects are addressed
//! by heap ID; the heap's "doubling table" of direct and indirect blocks turns
//! the ID's linear offset into a location in the file.
//!
//! <https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_infra_fractalheap>

use binrw::BinRead;
use bytes::Bytes;

use crate::error::{H5Error, H5Result};
use crate::object_store::{ObjectStoreFile, fetch_metadata, read_metadata};

#[derive(BinRead, Debug)]
#[br(magic = b"FRHP")]
#[allow(dead_code)]
struct RawHeader {
    #[br(assert(version == 0, "unsupported fractal heap version {}", version))]
    version: u8,
    heap_id_length: u16,
    io_filters_length: u16,
    flags: u8,
    max_managed_object_size: u32,
    next_huge_object_id: u64,
    huge_object_btree_address: u64,
    free_space: u64,
    free_space_manager_address: u64,
    managed_space: u64,
    allocated_managed_space: u64,
    direct_block_iterator_offset: u64,
    managed_object_count: u64,
    huge_object_size: u64,
    huge_object_count: u64,
    tiny_object_size: u64,
    tiny_object_count: u64,
    table_width: u16,
    starting_block_size: u64,
    max_direct_block_size: u64,
    max_heap_size: u16,
    starting_root_rows: u16,
    root_block_address: u64,
    current_root_rows: u16,
}

/// A fractal heap, loaded far enough to dereference heap IDs.
#[derive(Debug)]
pub struct FractalHeap {
    heap_id_length: u16,
    /// Direct blocks carry a checksum, which shifts where their data starts.
    checksummed: bool,
    filtered: bool,
    table_width: u64,
    starting_block_size: u64,
    max_direct_block_size: u64,
    /// Bits of heap address space, which fixes the width of block offsets.
    max_heap_size: u16,
    max_managed_object_size: u32,
    root_block_address: u64,
    current_root_rows: u16,
}

impl FractalHeap {
    /// Read the heap header at `address`.
    pub async fn open(file: &ObjectStoreFile, address: u64) -> H5Result<FractalHeap> {
        let raw: RawHeader = read_metadata(file, address).await?;

        if raw.starting_block_size == 0 || !raw.starting_block_size.is_power_of_two() {
            return Err(H5Error::corrupt(
                "fractal heap starting block size is not a power of two",
            ));
        }
        if raw.table_width == 0 {
            return Err(H5Error::corrupt("fractal heap table width is zero"));
        }

        Ok(FractalHeap {
            heap_id_length: raw.heap_id_length,
            checksummed: raw.flags & 0x02 != 0,
            filtered: raw.io_filters_length > 0,
            table_width: raw.table_width as u64,
            starting_block_size: raw.starting_block_size,
            max_direct_block_size: raw.max_direct_block_size,
            max_heap_size: raw.max_heap_size,
            max_managed_object_size: raw.max_managed_object_size,
            root_block_address: raw.root_block_address,
            current_root_rows: raw.current_root_rows,
        })
    }

    /// Width in bytes of a block offset, and of the offset field of a heap ID.
    fn offset_bytes(&self) -> u64 {
        (self.max_heap_size as u64).div_ceil(8)
    }

    /// Width in bytes of the length field of a managed-object heap ID: enough
    /// for the larger of a direct block's offsets, but never more than the
    /// largest managed object needs.
    fn length_bytes(&self) -> u64 {
        let direct_block_bits = log2(self.max_direct_block_size);
        direct_block_bits
            .div_ceil(8)
            .min(limit_enc_size(self.max_managed_object_size as u64))
    }

    /// Rows of direct blocks any indirect block can have before its entries
    /// start pointing at further indirect blocks.
    fn max_direct_rows(&self) -> u64 {
        log2(self.max_direct_block_size) - log2(self.starting_block_size) + 2
    }

    /// Size of the blocks in row `row` of a doubling table. The first two rows
    /// share the starting size; each row after that doubles.
    fn row_block_size(&self, row: u64) -> u64 {
        if row < 2 {
            self.starting_block_size
        } else {
            self.starting_block_size << (row - 1)
        }
    }

    /// Fetch the object a managed heap ID refers to.
    ///
    /// Tiny objects live in the ID itself; huge objects live outside the heap
    /// and are not used for links or attributes, so they are reported as
    /// unsupported rather than mis-read.
    pub async fn read_object(&self, file: &ObjectStoreFile, id: &[u8]) -> H5Result<Vec<u8>> {
        if id.len() < self.heap_id_length as usize {
            return Err(H5Error::corrupt(
                "fractal heap ID is shorter than the heap's ID length",
            ));
        }
        // Bits 4-5 of the first byte give the ID type.
        match (id[0] >> 4) & 0x03 {
            0 => {}
            1 => return Err(H5Error::unsupported("huge fractal heap objects")),
            2 => {
                // Tiny: the object is stored inline in the ID.
                let len = (id[0] & 0x0f) as usize + 1;
                return Ok(id[1..(1 + len).min(id.len())].to_vec());
            }
            other => {
                return Err(H5Error::corrupt(format!(
                    "unknown fractal heap ID type {other}"
                )));
            }
        }

        let offset_bytes = self.offset_bytes() as usize;
        let length_bytes = self.length_bytes() as usize;
        if id.len() < 1 + offset_bytes + length_bytes {
            return Err(H5Error::corrupt("truncated managed fractal heap ID"));
        }
        let heap_offset = le_uint(&id[1..1 + offset_bytes]);
        let length = le_uint(&id[1 + offset_bytes..1 + offset_bytes + length_bytes]);

        let (block_address, block_offset, block_size) = self.locate(file, heap_offset).await?;

        // Heap offsets are measured from the start of the direct block,
        // including its prefix, so objects never begin before it.
        let start = heap_offset
            .checked_sub(block_offset)
            .ok_or_else(|| H5Error::corrupt("fractal heap offset precedes its block"))?;
        if start < self.direct_block_prefix() || start + length > block_size {
            return Err(H5Error::corrupt(
                "fractal heap object lies outside its direct block",
            ));
        }

        let bytes = fetch_metadata(file, block_address + start, length).await?;
        Ok(bytes.to_vec())
    }

    /// Bytes before the object data in a direct block: signature, version, the
    /// heap header address, the block offset, and an optional checksum.
    fn direct_block_prefix(&self) -> u64 {
        4 + 1 + 8 + self.offset_bytes() + if self.checksummed { 4 } else { 0 }
    }

    /// Find the direct block containing `heap_offset`, returning its address,
    /// its own offset in the heap's address space, and its size.
    async fn locate(&self, file: &ObjectStoreFile, heap_offset: u64) -> H5Result<(u64, u64, u64)> {
        if self.filtered {
            return Err(H5Error::unsupported("filtered fractal heap blocks"));
        }
        if self.root_block_address == u64::MAX {
            return Err(H5Error::corrupt("fractal heap has no root block"));
        }

        // A heap small enough to need only one block points straight at it.
        if self.current_root_rows == 0 {
            return Ok((self.root_block_address, 0, self.starting_block_size));
        }

        self.descend(
            file,
            self.root_block_address,
            self.current_root_rows as u64,
            0,
            heap_offset,
        )
        .await
    }

    /// Walk down from an indirect block covering `[base, base + span)` until a
    /// direct block is reached.
    async fn descend(
        &self,
        file: &ObjectStoreFile,
        address: u64,
        nrows: u64,
        base: u64,
        heap_offset: u64,
    ) -> H5Result<(u64, u64, u64)> {
        let mut address = address;
        let mut nrows = nrows;
        let mut base = base;

        loop {
            let max_direct_rows = self.max_direct_rows();
            let direct_rows = nrows.min(max_direct_rows);
            let ndirect = direct_rows * self.table_width;
            let nindirect = (nrows * self.table_width).saturating_sub(ndirect);

            // Which entry of this indirect block covers the offset?
            let mut row_base = base;
            let mut found: Option<(u64, u64, u64, bool)> = None; // entry, entry base, size, is_direct
            for row in 0..nrows {
                let block_size = self.row_block_size(row);
                let row_span = block_size * self.table_width;
                if heap_offset < row_base + row_span {
                    let col = (heap_offset - row_base) / block_size;
                    let entry = row * self.table_width + col;
                    let entry_base = row_base + col * block_size;
                    found = Some((entry, entry_base, block_size, row < max_direct_rows));
                    break;
                }
                row_base += row_span;
            }

            let Some((entry, entry_base, block_size, is_direct)) = found else {
                return Err(H5Error::corrupt(
                    "fractal heap offset is outside the indexed address space",
                ));
            };

            let block = self
                .read_indirect_block(file, address, ndirect, nindirect)
                .await?;

            if is_direct {
                let child = *block
                    .direct
                    .get(entry as usize)
                    .ok_or_else(|| H5Error::corrupt("fractal heap indirect block is truncated"))?;
                if child == u64::MAX {
                    return Err(H5Error::corrupt(
                        "fractal heap offset points at an unallocated direct block",
                    ));
                }
                return Ok((child, entry_base, block_size));
            }

            let child = *block
                .indirect
                .get((entry - ndirect) as usize)
                .ok_or_else(|| H5Error::corrupt("fractal heap indirect block is truncated"))?;
            if child == u64::MAX {
                return Err(H5Error::corrupt(
                    "fractal heap offset points at an unallocated indirect block",
                ));
            }

            // Descend into the child indirect block, which represents a block of
            // `block_size` bytes in the doubling table.
            address = child;
            nrows = log2(block_size) - log2(self.starting_block_size) + 1;
            base = entry_base;
        }
    }

    async fn read_indirect_block(
        &self,
        file: &ObjectStoreFile,
        address: u64,
        ndirect: u64,
        nindirect: u64,
    ) -> H5Result<IndirectBlock> {
        // Direct entries carry a filtered size and mask only in filtered heaps,
        // which `locate` has already rejected.
        let entry_size = 8;
        let size = 4 + 1 + 8 + self.offset_bytes() + ndirect * entry_size + nindirect * 8 + 4;
        let bytes: Bytes = fetch_metadata(file, address, size).await?;
        if bytes.len() < 5 || &bytes[..4] != b"FHIB" {
            return Err(H5Error::corrupt("expected a fractal heap indirect block"));
        }

        let mut pos = (4 + 1 + 8 + self.offset_bytes()) as usize;
        let mut direct = Vec::with_capacity(ndirect as usize);
        for _ in 0..ndirect {
            direct.push(le_uint(&bytes[pos..pos + 8]));
            pos += entry_size as usize;
        }
        let mut indirect = Vec::with_capacity(nindirect as usize);
        for _ in 0..nindirect {
            indirect.push(le_uint(&bytes[pos..pos + 8]));
            pos += 8;
        }
        Ok(IndirectBlock { direct, indirect })
    }
}

struct IndirectBlock {
    direct: Vec<u64>,
    indirect: Vec<u64>,
}

/// Integer log2 of a power of two.
fn log2(v: u64) -> u64 {
    if v == 0 { 0 } else { v.ilog2() as u64 }
}

/// Bytes needed to encode values up to `limit`, matching the library's
/// `H5VM_limit_enc_size`.
fn limit_enc_size(limit: u64) -> u64 {
    (log2(limit) / 8) + 1
}

/// Read a little-endian unsigned integer from up to 8 bytes.
fn le_uint(bytes: &[u8]) -> u64 {
    let mut buf = [0u8; 8];
    let n = bytes.len().min(8);
    buf[..n].copy_from_slice(&bytes[..n]);
    u64::from_le_bytes(buf)
}
