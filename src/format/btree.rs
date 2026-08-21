//! Version 1 B-tree traversal.
//!
//! Walking is breadth-first rather than depth-first: every node at one level is
//! read in a single batched request, so a tree costs one round trip per level
//! instead of one per node. For a chunk index over a large dataset that is the
//! difference between a few requests and hundreds.

use std::io::Cursor;

use binrw::BinRead;

use crate::error::H5Result;
use crate::object_store::ObjectStoreFile;

/// How much of a node to read before its true length is known. B-tree nodes are
/// bounded by the "K" values in the superblock and comfortably fit this.
const NODE_FETCH_SIZE: u64 = 8192;

pub trait HasPointer {
    fn child_pointer(&self) -> u64;
}

pub trait BTree: Sized {
    type Leaf: Clone + HasPointer;
    type Args: Clone;
    fn children(&self) -> &[Self::Leaf];
    fn args(&self) -> Self::Args;
    fn node_level(&self) -> u8;
}

/// Collect all leaf entries from a B-tree (no args variant).
pub async fn collect_btree_leaves<B>(file: &ObjectStoreFile, root: B) -> H5Result<Vec<B::Leaf>>
where
    B: BTree<Args = ()> + Clone + for<'a> BinRead<Args<'a> = ()>,
{
    collect_btree_leaves_args(file, root).await
}

/// Collect all leaf entries from a B-tree, one level at a time.
pub async fn collect_btree_leaves_args<B, A>(
    file: &ObjectStoreFile,
    root: B,
) -> H5Result<Vec<B::Leaf>>
where
    A: Clone,
    B: BTree<Args = A> + Clone + for<'a> BinRead<Args<'a> = A>,
{
    let mut result = Vec::new();
    let mut level = vec![root];

    while !level.is_empty() {
        let mut next_addresses: Vec<(u64, u64)> = vec![];
        let mut next_args: Vec<A> = vec![];

        for node in &level {
            if node.node_level() == 0 {
                result.extend(node.children().iter().cloned());
            } else {
                let args = node.args();
                for child in node.children() {
                    next_addresses.push((child.child_pointer(), NODE_FETCH_SIZE));
                    next_args.push(args.clone());
                }
            }
        }

        if next_addresses.is_empty() {
            break;
        }

        let blocks = file.read_metadata_many(&next_addresses).await?;
        level = blocks
            .into_iter()
            .zip(next_args)
            .map(|(bytes, args)| {
                let mut cursor = Cursor::new(bytes);
                B::read_le_args(&mut cursor, args)
            })
            .collect::<Result<Vec<_>, _>>()?;
    }

    Ok(result)
}
