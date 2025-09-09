#![allow(dead_code)]
use std::io::{Read, Seek, SeekFrom};

use binrw::{BinRead, BinResult};

use crate::format::{
    btree::BTreeIter,
    metadata::{
        GroupBTreeV1, GroupPointerV1, GroupSymbolTableNode, LoadedLocalHeap, SymbolTableEntry,
    },
    object::{DataLayoutMessage, DataObjectHeader, DataspaceMessage, DatatypeMessage},
};

pub(crate) mod format;

struct Object {
    name: String,
    header: DataObjectHeader,
}

struct Group {
    name: String,
    btree: GroupBTreeV1,
    loaded_local_heap: LoadedLocalHeap,
}

struct GroupObjectIter<'a, R> {
    group: &'a Group,
    reader: &'a mut R,
    btree_iter: BTreeIter<'a, R, GroupBTreeV1>,
    current_symbol_table: Option<GroupSymbolTableNode>,
    symbol_table_pos: usize,
}

impl<'a, R: Read + Seek> Iterator for GroupObjectIter<'a, R> {
    type Item = BinResult<Object>;

    fn next(&mut self) -> Option<Self::Item> {
        // check if we're done with the current ST
        if let Some(st) = &self.current_symbol_table {
            if self.symbol_table_pos == st.entries.len() {
                self.current_symbol_table = None;
            }
        }

        if self.current_symbol_table.is_none() {
            let ptr = self.btree_iter.next()?;
            match ptr {
                Ok(p) => {
                    let st = self.group.load_symbol_table(&p, self.reader);
                    match st {
                        Ok(st) => {
                            self.current_symbol_table = Some(st);
                            self.symbol_table_pos = 0;
                        }
                        Err(e) => return Some(Err(e)),
                    }
                }
                Err(e) => return Some(Err(e)),
            }
        }

        let Some(st) = &self.current_symbol_table else {
            return None;
        };

        let e = &st.entries[self.symbol_table_pos];
        self.symbol_table_pos += 1;
        Some(self.group.load_object(e, self.reader))
    }
}

impl Group {
    fn iter_objects<'a, R: Read + Seek>(&'a self, reader: &'a mut R) -> GroupObjectIter<'a, R> {
        let it = BTreeIter::new(reader, self.btree.clone());

        GroupObjectIter {
            group: self,
            reader,
            btree_iter: it,
            current_symbol_table: None,
            symbol_table_pos: 0,
        }
    }

    fn load_symbol_table<R: Read + Seek>(
        &self,
        ptr: &GroupPointerV1,
        reader: &mut R,
    ) -> BinResult<GroupSymbolTableNode> {
        reader.seek(SeekFrom::Start(ptr.child_pointer))?;
        GroupSymbolTableNode::read_le(reader)
    }

    fn iter_symbol_table<R: Read + Seek>(
        &self,
        symbol_table: &GroupSymbolTableNode,
        reader: &mut R,
    ) -> impl Iterator<Item = BinResult<Object>> {
        symbol_table
            .entries
            .iter()
            .map(|e| self.load_object(e, reader))
    }

    fn load_object<R: Read + Seek>(
        &self,
        ptr: &SymbolTableEntry,
        reader: &mut R,
    ) -> BinResult<Object> {
        let name = self.loaded_local_heap.get_string(ptr.link_name_offset)?;

        reader.seek(SeekFrom::Start(ptr.object_header_address))?;
        let header = DataObjectHeader::read_le(reader)?;

        Ok(Object { name, header })
    }

    // FIXME: this loads the object header of every object - we can avoid this while
    // looking for an object by name, if needed.
    fn find_obj<R: Read + Seek>(&self, name: impl AsRef<str>, reader: &mut R) -> Object {}
}

struct Dataset {
    dataspace: DataspaceMessage,
    datatype: DatatypeMessage,
    layout: DataLayoutMessage,
}
