#![allow(dead_code)]
use std::{
    cell::RefCell,
    io::{Read, Seek, SeekFrom},
    rc::Rc,
};

use binrw::{BinRead, BinResult};

#[derive(Debug)]
pub struct BTreeIter<'a, T, B> {
    reader: &'a mut T,
    stack: Vec<(B, usize)>,
}

impl<'a, T, B> BTreeIter<'a, T, B> {
    pub fn new(reader: &'a mut T, tree: B) -> BTreeIter<'a, T, B> {
        BTreeIter {
            reader,
            stack: vec![(tree, 0)],
        }
    }
}

#[derive(Debug)]
pub struct BTreeIter2<B, R> {
    reader: Rc<RefCell<R>>,
    stack: Vec<(B, usize)>,
}

impl<B, R> BTreeIter2<B, R> {
    pub fn new(reader: Rc<RefCell<R>>, tree: B) -> BTreeIter2<B, R> {
        BTreeIter2 {
            reader,
            stack: vec![(tree, 0)],
        }
    }
}

pub trait HasPointer {
    fn child_pointer(&self) -> u64;
}
pub trait BTree {
    type Leaf: Clone + HasPointer;
    type Args;
    fn children(&self) -> &[Self::Leaf];
    fn args(&self) -> Self::Args;
    fn node_level(&self) -> u8;
}
enum IterState {
    Done,
    NodeDone,
    InnerNode,
    LeafNode,
}
impl<'a, T: Read + Seek, B: BTree<Args = A> + BinRead<Args<'a> = A>, A> Iterator
    for BTreeIter<'a, T, B>
{
    type Item = BinResult<<B as BTree>::Leaf>;

    fn next(&mut self) -> Option<Self::Item> {
        use IterState::*;

        loop {
            let state = match self.stack.last() {
                None => IterState::Done,
                Some((node, pos)) if node.children().len() == *pos => NodeDone,
                Some((node, _)) if node.node_level() > 0 => InnerNode,
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
                            .seek(SeekFrom::Start(node.children()[*pos].child_pointer()))
                            .ok()?;
                        *pos += 1;

                        let args = node.args();
                        B::read_le_args(self.reader, args).ok()?
                    };

                    self.stack.push((child_node, 0));
                }
                IterState::LeafNode => {
                    let Some((node, pos)) = self.stack.last_mut() else {
                        unreachable!();
                    };
                    let leaf = node.children()[*pos].clone();
                    *pos += 1;
                    return Some(Ok(leaf));
                }
            };
        }
    }
}

impl<'a, R: Read + Seek, B: BTree<Args = A> + BinRead<Args<'a> = A>, A> Iterator
    for BTreeIter2<B, R>
{
    type Item = BinResult<<B as BTree>::Leaf>;

    fn next(&mut self) -> Option<Self::Item> {
        use IterState::*;

        loop {
            let state = match self.stack.last() {
                None => IterState::Done,
                Some((node, pos)) if node.children().len() == *pos => NodeDone,
                Some((node, _)) if node.node_level() > 0 => InnerNode,
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
                            .borrow_mut()
                            .seek(SeekFrom::Start(node.children()[*pos].child_pointer()))
                            .ok()?;
                        *pos += 1;

                        let args = node.args();
                        B::read_le_args(&mut *self.reader.borrow_mut(), args).ok()?
                    };

                    self.stack.push((child_node, 0));
                }
                IterState::LeafNode => {
                    let Some((node, pos)) = self.stack.last_mut() else {
                        unreachable!();
                    };
                    let leaf = node.children()[*pos].clone();
                    *pos += 1;
                    return Some(Ok(leaf));
                }
            };
        }
    }
}
