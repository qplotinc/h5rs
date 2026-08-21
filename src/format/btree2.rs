//! Version 2 B-trees.
//!
//! These index link names in dense groups, attribute names in objects with many
//! attributes, and dataset chunks when more than one dimension is unlimited.
//!
//! h5rs only ever needs to *enumerate* a v2 B-tree, never to search it by key,
//! so this module walks every node and hands back the raw record bytes for the
//! caller to interpret according to the tree's type.
//!
//! <https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_infra_btrees_v2>

use binrw::BinRead;

use crate::error::{H5Error, H5Result};
use crate::object_store::{ObjectStoreFile, read_metadata};

/// The prefix and checksum around every internal and leaf node: a 4-byte
/// signature, a version and type byte, and a trailing 4-byte checksum.
const NODE_OVERHEAD: u64 = 10;

/// A version 2 B-tree header, the entry point to the tree.
#[derive(Debug)]
pub struct BTreeV2Header {
    /// The record type, which decides how records are interpreted.
    pub btree_type: u8,
    /// Size in bytes of every record in the tree.
    pub record_size: u16,
    node_size: u32,
    depth: u16,
    root_address: u64,
    root_nrec: u16,
    /// Per-depth node geometry, index 0 being the leaves.
    node_info: Vec<NodeInfo>,
    /// Width of the "number of records in child" field, the same at every depth.
    max_nrec_size: u8,
}

/// Geometry of the nodes at one depth of the tree.
#[derive(Debug, Clone, Copy)]
struct NodeInfo {
    /// Width of the "total records in child and descendants" field for children
    /// at this depth. Zero for leaves, which is why nodes just above the leaves
    /// omit that field entirely.
    cum_max_nrec_size: u8,
}

#[derive(BinRead, Debug)]
#[br(magic = b"BTHD")]
#[allow(dead_code)]
struct RawHeader {
    #[br(assert(version == 0, "unsupported v2 B-tree header version {}", version))]
    version: u8,
    btree_type: u8,
    node_size: u32,
    record_size: u16,
    depth: u16,
    split_percent: u8,
    merge_percent: u8,
    root_address: u64,
    root_nrec: u16,
    total_nrec: u64,
    checksum: u32,
}

/// Bytes needed to encode values up to `limit`, matching the library's
/// `H5VM_limit_enc_size`.
fn limit_enc_size(limit: u64) -> u8 {
    let log2 = if limit == 0 { 0 } else { limit.ilog2() as u64 };
    ((log2 / 8) + 1) as u8
}

impl BTreeV2Header {
    /// Read the header at `address` and derive the node geometry needed to walk
    /// the tree.
    pub async fn read(file: &ObjectStoreFile, address: u64) -> H5Result<BTreeV2Header> {
        let raw: RawHeader = read_metadata(file, address).await?;

        if raw.record_size == 0 {
            return Err(H5Error::corrupt("v2 B-tree with zero-length records"));
        }
        let record_size = raw.record_size as u64;
        let node_size = raw.node_size as u64;
        if node_size <= NODE_OVERHEAD {
            return Err(H5Error::corrupt("v2 B-tree node size is too small"));
        }

        // Leaves hold nothing but records.
        let max_leaf_nrec = (node_size - NODE_OVERHEAD) / record_size;
        let max_nrec_size = limit_enc_size(max_leaf_nrec);

        // Leaves have no descendants, so their parents store no total.
        let mut node_info = vec![NodeInfo {
            cum_max_nrec_size: 0,
        }];
        let mut cum_max_nrec = max_leaf_nrec;

        for depth in 1..=raw.depth as usize {
            // Each child costs an address, a record count, and — except just
            // above the leaves — a total-records count.
            let pointer_size =
                8 + max_nrec_size as u64 + node_info[depth - 1].cum_max_nrec_size as u64;
            let available = node_size
                .checked_sub(NODE_OVERHEAD + pointer_size)
                .ok_or_else(|| H5Error::corrupt("v2 B-tree node size is too small"))?;
            let max_nrec = available / (record_size + pointer_size);

            cum_max_nrec = (max_nrec + 1) * cum_max_nrec + max_nrec;
            node_info.push(NodeInfo {
                cum_max_nrec_size: limit_enc_size(cum_max_nrec),
            });
        }

        Ok(BTreeV2Header {
            btree_type: raw.btree_type,
            record_size: raw.record_size,
            node_size: raw.node_size,
            depth: raw.depth,
            root_address: raw.root_address,
            root_nrec: raw.root_nrec,
            node_info,
            max_nrec_size,
        })
    }

    /// Every record in the tree, as raw `record_size`-byte slices.
    ///
    /// Walking is breadth-first so that each level of the tree costs a single
    /// batched request rather than one request per node.
    pub async fn collect_records(&self, file: &ObjectStoreFile) -> H5Result<Vec<Vec<u8>>> {
        if self.root_address == u64::MAX || self.root_nrec == 0 {
            return Ok(vec![]);
        }

        let mut records = vec![];
        // (address, number of records) for every node at the current depth.
        let mut level = vec![(self.root_address, self.root_nrec as u64)];
        let mut depth = self.depth;

        while !level.is_empty() {
            let spans: Vec<(u64, u64)> = level
                .iter()
                .map(|&(address, _)| (address, self.node_size as u64))
                .collect();
            let nodes = file.read_metadata_many(&spans).await?;

            let mut next = vec![];
            for ((address, nrec), bytes) in level.iter().zip(&nodes) {
                let expected: &[u8; 4] = if depth == 0 { b"BTLF" } else { b"BTIN" };
                if bytes.len() < 6 || &bytes[..4] != expected {
                    return Err(H5Error::corrupt(format!(
                        "expected a {} node at {address}",
                        String::from_utf8_lossy(expected)
                    )));
                }

                let mut pos = 6usize; // signature, version, type
                let record_len = self.record_size as usize;
                for _ in 0..*nrec {
                    let end = pos + record_len;
                    if end > bytes.len() {
                        return Err(H5Error::corrupt("v2 B-tree node overruns its node size"));
                    }
                    records.push(bytes[pos..end].to_vec());
                    pos = end;
                }

                if depth == 0 {
                    continue;
                }

                // An internal node has one more child pointer than it has records.
                let total_size = self.node_info[depth as usize - 1].cum_max_nrec_size as usize;
                let nrec_size = self.max_nrec_size as usize;
                for _ in 0..=*nrec {
                    let child = read_uint(bytes, pos, 8)?;
                    pos += 8;
                    let child_nrec = read_uint(bytes, pos, nrec_size)?;
                    pos += nrec_size;
                    pos += total_size;
                    next.push((child, child_nrec));
                }
            }

            if depth == 0 {
                break;
            }
            depth -= 1;
            level = next;
        }

        Ok(records)
    }
}

/// Read a little-endian unsigned integer of `len` bytes.
fn read_uint(bytes: &[u8], pos: usize, len: usize) -> H5Result<u64> {
    let slice = bytes
        .get(pos..pos + len)
        .ok_or_else(|| H5Error::corrupt("v2 B-tree node overruns its node size"))?;
    let mut buf = [0u8; 8];
    buf[..len].copy_from_slice(slice);
    Ok(u64::from_le_bytes(buf))
}
