#![allow(dead_code)]

use binrw::BinRead;

use crate::error::H5Result;
use crate::object_store::{ObjectStoreFile, read_metadata, read_metadata_args};

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

/// Collect all leaf entries from a B-tree via async DFS (no args variant).
pub async fn collect_btree_leaves<B>(
    file: &ObjectStoreFile,
    root: B,
) -> H5Result<Vec<B::Leaf>>
where
    B: BTree<Args = ()> + Clone + for<'a> BinRead<Args<'a> = ()>,
{
    let mut result = Vec::new();
    let mut stack = vec![root];

    while let Some(node) = stack.pop() {
        if node.node_level() == 0 {
            result.extend(node.children().iter().cloned());
        } else {
            // Push children in reverse order to preserve left-to-right traversal
            for child in node.children().iter().rev() {
                let child_node: B = read_metadata(file, child.child_pointer()).await?;
                stack.push(child_node);
            }
        }
    }

    Ok(result)
}

/// Collect all leaf entries from a B-tree via async DFS (with args variant).
pub async fn collect_btree_leaves_args<B, A>(
    file: &ObjectStoreFile,
    root: B,
) -> H5Result<Vec<B::Leaf>>
where
    A: Clone,
    B: BTree<Args = A> + Clone + for<'a> BinRead<Args<'a> = A>,
{
    let mut result = Vec::new();
    let mut stack = vec![root];

    while let Some(node) = stack.pop() {
        if node.node_level() == 0 {
            result.extend(node.children().iter().cloned());
        } else {
            let args = node.args();
            // Push children in reverse order to preserve left-to-right traversal
            for child in node.children().iter().rev() {
                let child_node: B =
                    read_metadata_args(file, child.child_pointer(), args.clone()).await?;
                stack.push(child_node);
            }
        }
    }

    Ok(result)
}
