pub mod btree;
pub mod metadata;
pub mod object;

#[cfg(test)]
mod tests {
    use std::io::{Cursor, Read, Seek, SeekFrom};

    use binrw::{BinRead, BinResult};

    use crate::{
        Dataset,
        format::{
            btree::BTreeIter,
            metadata::{ChunkBTreeV1, GroupBTreeV1, GroupSymbolTableNode, LocalHeap, SuperblockV0},
            object::{
                DataLayoutChunked, DataLayoutInner, DataLayoutMessage, DataLayoutV3,
                DataObjectHeader, LayoutInner,
            },
        },
    };

    fn get_file() -> Vec<u8> {
        let mut f = std::fs::File::open("datasets/gene_bc_matrix.h5").unwrap();

        let mut buf = vec![];
        f.read_to_end(&mut buf).unwrap();
        buf
    }

    #[test]
    fn basic() {
        let buf = get_file();

        let mut full_file = Cursor::new(&buf[..]);

        let sb = SuperblockV0::read_le(&mut full_file);
        println!("superblock: {:#?}, end of sb: {}", sb, full_file.position());
        let sb = sb.unwrap();

        full_file.set_position(sb.root_group_symbol_table_entry.object_header_address);
        let root_group = DataObjectHeader::read_le(&mut full_file);
        println!("root group orig: {:#?}", root_group);
        let mut root_group = root_group.unwrap();
        root_group
            .load_continuation_messages(&mut full_file)
            .unwrap();

        println!("root group new messages: {:#?}", root_group);

        let root_group_symbol_table = root_group.symbol_table_message().unwrap();

        let rg = root_group
            .to_group("/".to_string(), &mut full_file)
            .unwrap()
            .unwrap();

        full_file.set_position(root_group_symbol_table.btree_address);
        let root_group_btree = GroupBTreeV1::read_le(&mut full_file).unwrap();
        println!("root group btree: {:#?}", root_group_btree);

        full_file.set_position(root_group_symbol_table.local_heap_address);
        let root_group_heap = LocalHeap::read_le(&mut full_file);
        println!("root group heap: {:#?}", root_group_heap);

        // load the group object
        let g0 = &root_group_btree.children[0];
        full_file.set_position(g0.child_pointer);

        for e in root_group_btree.children {
            let mut c = Cursor::new(&buf[e.child_pointer as usize..]);
            let symbol = GroupSymbolTableNode::read_le(&mut c);
            println!("{:#?}", symbol);

            let ste = &symbol.unwrap().entries[0];
            //for ste in &symbol.unwrap().entries {
            let mut c = Cursor::new(&buf[ste.object_header_address as usize..]);
            let oh = DataObjectHeader::read_le(&mut c);
            println!("{:#?}", oh);
            let obj = oh.unwrap();

            if let Some(ds) = obj.to_dataset("asdf".to_string()) {
                test_chunk_iter(&ds, &mut full_file).unwrap();
            }
        }
    }

    fn test_chunk_iter<R: Read + Seek>(dataset: &Dataset, reader: &mut R) -> BinResult<()> {
        let d = dataset.dataspace.dimensionality;

        let DataLayoutMessage {
            inner:
                DataLayoutInner::V3(DataLayoutV3 {
                    layout_inner:
                        LayoutInner::Chunked(
                            DataLayoutChunked {
                                address,
                                dimension_sizes,
                                ..
                            },
                            ..,
                        ),
                    ..
                }),
            ..
        } = &dataset.layout
        else {
            return Ok(());
        };

        // now load a B-Tree at address
        let _ = reader.seek(SeekFrom::Start(*address))?;
        let bt = ChunkBTreeV1::read_le_args(reader, (d,));

        println!("dataset is chunked. reading B-Tree:\n{:#?}", bt);

        let it = BTreeIter::new(reader, bt.unwrap());

        let mut last = 0;
        let mut n = 0;

        for c in it {
            if c.is_err() {
                println!("btree err: {:?}", c);
            }

            let c = c.unwrap();
            let delta = c.key.offsets[0] - last;
            if c.key.offsets[0] > 0 {
                assert_eq!(delta, dimension_sizes[0] as u64)
            }

            last = c.key.offsets[0];
            n += 1;
        }

        println!("last pos: {last}, num_chunks: {n}");
        Ok(())
    }
}
