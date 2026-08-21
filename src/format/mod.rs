pub mod btree;
pub mod btree2;
pub mod chunk_index;
pub mod dense;
pub mod fractal_heap;
pub mod metadata;
pub mod object;

#[cfg(test)]
mod tests {
    use object_store::path::Path;

    use crate::{error::H5Result, format::metadata::Superblock, object_store::ObjectStoreFile};

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

    const MATRIX_FILE: &str = "datasets/gene_bc_matrix.h5";

    /// Walk a real file from the superblock down to a dataset's chunks, so a
    /// regression anywhere in that path shows up here rather than only in the
    /// higher-level read tests.
    #[crate::async_test]
    async fn walk_matrix_file() -> H5Result<()> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            if !std::path::Path::new(MATRIX_FILE).exists() {
                println!("SKIP: {MATRIX_FILE} not present (see README: Test data)");
                return Ok(());
            }
        }
        let file = test_file(MATRIX_FILE);

        let sb = Superblock::read(&file).await?;
        println!("superblock version {}", sb.version);

        let datasets = crate::list_datasets(&file).await?;
        assert!(!datasets.is_empty(), "expected datasets in the matrix file");

        let chunked = datasets
            .iter()
            .find(|d| d.chunk_shape.is_some())
            .expect("expected at least one chunked dataset");
        println!(
            "{} shape={:?} chunks={:?}",
            chunked.path, chunked.shape, chunked.chunk_shape
        );

        let segments: Vec<&str> = chunked.path.trim_start_matches('/').split('/').collect();
        let ds = crate::open_chunked_dataset(&file, &segments)
            .await?
            .expect("chunked dataset should open");
        let chunks = ds.collect_chunks(&file).await?;
        assert!(!chunks.is_empty(), "chunked dataset should have chunks");
        println!("{} chunks", chunks.len());

        Ok(())
    }
}
