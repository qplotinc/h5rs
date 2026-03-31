pub mod btree;
pub mod metadata;
pub mod object;

#[cfg(test)]
mod tests {
    use object_store::path::Path;

    use crate::{
        Dataset,
        error::H5Result,
        format::{
            btree::collect_btree_leaves_args,
            metadata::{ChunkBTreeV1, GroupBTreeV1, SuperblockV0},
            object::{
                DataLayoutChunked, DataLayoutInner, DataLayoutMessage, DataLayoutV3,
                DataObjectHeader, LayoutInner,
            },
        },
        object_store::{ObjectStoreFile, read_metadata, read_metadata_args},
    };

    fn test_file(path: &str) -> ObjectStoreFile {
        #[cfg(not(target_arch = "wasm32"))]
        {
            use object_store::local::LocalFileSystem;
            let cwd = std::env::current_dir().unwrap();
            let store = LocalFileSystem::new_with_prefix(&cwd).unwrap();
            ObjectStoreFile::new(Box::new(store), Path::from(path))
        }
        #[cfg(target_arch = "wasm32")]
        {
            let store = crate::node_store::NodeFileSystem::cwd();
            ObjectStoreFile::new(Box::new(store), Path::from(path))
        }
    }

    #[crate::async_test]
    async fn basic() -> H5Result<()> {
        let file = test_file("datasets/gene_bc_matrix.h5");

        let sb: SuperblockV0 = read_metadata(&file, 0).await?;
        println!("superblock: {:#?}", sb);

        let mut root_group: DataObjectHeader = read_metadata(
            &file,
            sb.root_group_symbol_table_entry.object_header_address,
        )
        .await?;
        println!("root group orig: {:#?}", root_group);

        root_group.load_continuation_messages(&file).await?;
        println!("root group new messages: {:#?}", root_group);

        let root_group_symbol_table = root_group.symbol_table_message().unwrap();

        let rg = root_group
            .to_group("/".to_string(), &file)
            .await
            .unwrap()?;

        let root_group_btree: GroupBTreeV1 =
            read_metadata(&file, root_group_symbol_table.btree_address).await?;
        println!("root group btree: {:#?}", root_group_btree);

        let root_group_heap: crate::format::metadata::LocalHeap =
            read_metadata(&file, root_group_symbol_table.local_heap_address).await?;
        println!("root group heap: {:#?}", root_group_heap);

        // load a child object to exercise symbol table reading
        let g0 = &root_group_btree.children[0];
        let symbol: crate::format::metadata::GroupSymbolTableNode =
            read_metadata(&file, g0.child_pointer).await?;
        println!("{:#?}", symbol);

        let ste = &symbol.entries[0];
        let obj: DataObjectHeader =
            read_metadata(&file, ste.object_header_address).await?;
        println!("{:#?}", obj);

        if let Some(ds) = obj.to_dataset("asdf".to_string()) {
            test_chunk_iter(&ds, &file).await?;
        }

        Ok(())
    }

    async fn test_chunk_iter(dataset: &Dataset, file: &ObjectStoreFile) -> H5Result<()> {
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

        let bt: ChunkBTreeV1 = read_metadata_args(file, *address, (d,)).await?;
        println!("dataset is chunked. reading B-Tree:\n{:#?}", bt);

        let leaves = collect_btree_leaves_args(file, bt).await?;

        let mut last = 0;
        let mut n = 0;

        for c in &leaves {
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
