#![allow(dead_code)]

use crate::error::H5Result;
use crate::format::{
    btree::collect_btree_leaves,
    metadata::{
        GroupBTreeV1, GroupPointerV1, GroupSymbolTableNode, LoadedLocalHeap, SuperblockV0,
        SymbolTableEntry,
    },
    object::{
        AttributeMessage, DataLayoutInner, DataLayoutMessage, DataLayoutV3, DataObjectHeader,
        DataspaceMessage, DatatypeMessage, FilterMessage, LayoutInner,
    },
};
use crate::object_store::{ObjectStoreFile, read_metadata};

pub(crate) mod chunked;
pub mod error;
pub(crate) mod format;
pub mod h5type;
pub(crate) mod object_store;

pub use chunked::{ChunkedDataset, NdArray};

struct File {
    superblock: SuperblockV0,
    pub root_group: Group,
}

impl File {
    pub async fn open(file: &ObjectStoreFile) -> H5Result<File> {
        let sb: SuperblockV0 = read_metadata(file, 0).await?;

        let mut root_group: DataObjectHeader =
            read_metadata(file, sb.root_group_symbol_table_entry.object_header_address).await?;
        root_group.load_continuation_messages(file).await?;

        let rg = root_group.to_group("/".to_string(), file).await.unwrap()?;

        Ok(File {
            superblock: sb,
            root_group: rg,
        })
    }
}

struct Object {
    name: String,
    header: DataObjectHeader,
}

impl Object {
    pub async fn to_group(&self, file: &ObjectStoreFile) -> H5Result<Option<Group>> {
        match self.header.to_group(self.name.clone(), file).await {
            Some(r) => Ok(Some(r?)),
            None => Ok(None),
        }
    }
}

struct Group {
    name: String,
    btree: GroupBTreeV1,
    loaded_local_heap: LoadedLocalHeap,
    pub attributes: Vec<AttributeMessage>,
}

impl Group {
    async fn object_refs(
        &self,
        file: &ObjectStoreFile,
    ) -> H5Result<Vec<(String, SymbolTableEntry)>> {
        let ptrs: Vec<GroupPointerV1> = collect_btree_leaves(file, self.btree.clone()).await?;

        let mut res = vec![];

        for p in ptrs {
            let st = self.load_symbol_table(&p, file).await?;
            for e in &st.entries {
                let name = self.loaded_local_heap.get_string(e.link_name_offset)?;
                let e = e.clone();
                res.push((name, e));
            }
        }

        Ok(res)
    }

    async fn load_symbol_table(
        &self,
        ptr: &GroupPointerV1,
        file: &ObjectStoreFile,
    ) -> H5Result<GroupSymbolTableNode> {
        read_metadata(file, ptr.child_pointer).await
    }

    async fn load_object(
        &self,
        ptr: &SymbolTableEntry,
        file: &ObjectStoreFile,
    ) -> H5Result<Object> {
        let name = self.loaded_local_heap.get_string(ptr.link_name_offset)?;
        let header: DataObjectHeader = read_metadata(file, ptr.object_header_address).await?;

        Ok(Object { name, header })
    }

    async fn find_obj(
        &self,
        name: impl AsRef<str>,
        file: &ObjectStoreFile,
    ) -> H5Result<Option<Object>> {
        let r = self.object_refs(file).await?;
        let Some((name, ste)) = r.iter().find(|(n, _)| n == name.as_ref()) else {
            println!("didn't find object: {}", name.as_ref());
            return Ok(None);
        };

        let mut header: DataObjectHeader = read_metadata(file, ste.object_header_address).await?;
        header.load_continuation_messages(file).await?;

        Ok(Some(Object {
            name: name.clone(),
            header,
        }))
    }
}

struct Dataset {
    name: String,
    dataspace: DataspaceMessage,
    datatype: DatatypeMessage,
    layout: DataLayoutMessage,
    filter: Option<FilterMessage>,
    pub attributes: Vec<AttributeMessage>,
}

impl Dataset {
    pub fn chunked(&self) -> Option<chunked::ChunkedDataset> {
        let DataLayoutInner::V3(DataLayoutV3 {
            layout_inner: LayoutInner::Chunked(chunk),
            ..
        }) = self.layout.inner.clone()
        else {
            return None;
        };

        Some(chunked::ChunkedDataset {
            name: self.name.clone(),
            dataspace: self.dataspace.clone(),
            datatype: self.datatype.clone(),
            layout: self.layout.clone(),
            filter: self.filter.clone(),
            chunks_layout: chunk,
        })
    }
}

#[cfg(test)]
mod test {

    use std::fmt::Debug;

    use object_store::{local::LocalFileSystem, path::Path};

    use crate::error::H5Result;
    use crate::format::object::{AttributeMessage, DataObjectHeader, TypeDescriptor};
    use crate::h5type::H5Type;
    use crate::object_store::{ObjectStoreFile, read_metadata};

    const MOL_INFO_FILE: &str = "datasets/frozen_pbmc_donor_c_molecule_info.h5";
    const MATRIX_FILE: &str = "datasets/gene_bc_matrix.h5";

    fn test_file(path: &str) -> ObjectStoreFile {
        let cwd = std::env::current_dir().unwrap();
        let store = LocalFileSystem::new_with_prefix(&cwd).unwrap();
        ObjectStoreFile::new(Box::new(store), Path::from(path))
    }

    /// Read all chunks of a dataset via h5rs and compare byte-for-byte
    /// against the hdf5 C library (gold standard).
    async fn compare_typed<T>(
        cds: &super::ChunkedDataset,
        hdf5_ds: &hdf5::Dataset,
        path: &str,
        file: &ObjectStoreFile,
    ) -> H5Result<()>
    where
        T: H5Type + hdf5::H5Type + PartialEq + Debug,
    {
        let expected: Vec<T> = hdf5_ds.read_raw::<T>().unwrap();
        let result = cds.read_full::<T>(file).await?;

        assert_eq!(result.data.len(), expected.len(), "{path}: length mismatch");
        assert_eq!(&result.data[..], &expected[..], "{path}: data mismatch");
        println!(
            "  OK {path} ({} values, shape {:?})",
            expected.len(),
            result.shape
        );

        Ok(())
    }

    /// Dispatch to the correct typed comparison based on the HDF5 datatype.
    async fn compare_dataset(
        cds: &super::ChunkedDataset,
        hdf5_ds: &hdf5::Dataset,
        path: &str,
        file: &ObjectStoreFile,
    ) -> H5Result<()> {
        match &cds.datatype.type_desc {
            TypeDescriptor::FixedPoint(fp) => match (fp.signed(), fp.size()) {
                (0, 1) => compare_typed::<u8>(cds, hdf5_ds, path, file).await,
                (0, 2) => compare_typed::<u16>(cds, hdf5_ds, path, file).await,
                (0, 4) => compare_typed::<u32>(cds, hdf5_ds, path, file).await,
                (0, 8) => compare_typed::<u64>(cds, hdf5_ds, path, file).await,
                (1, 1) => compare_typed::<i8>(cds, hdf5_ds, path, file).await,
                (1, 2) => compare_typed::<i16>(cds, hdf5_ds, path, file).await,
                (1, 4) => compare_typed::<i32>(cds, hdf5_ds, path, file).await,
                (1, 8) => compare_typed::<i64>(cds, hdf5_ds, path, file).await,
                (s, sz) => panic!("{path}: unsupported FixedPoint signed={s} size={sz}"),
            },
            TypeDescriptor::FloatingPoint(fp) => match fp.size() {
                4 => compare_typed::<f32>(cds, hdf5_ds, path, file).await,
                8 => compare_typed::<f64>(cds, hdf5_ds, path, file).await,
                sz => panic!("{path}: unsupported FloatingPoint size={sz}"),
            },
            other => {
                println!("  SKIP {path}: unsupported type {other:?}");
                Ok(())
            }
        }
    }

    /// Walk a group's children, collecting chunked datasets and sub-groups.
    async fn collect_from_group(
        group: &super::Group,
        path: &str,
        file: &ObjectStoreFile,
        hf: &hdf5::File,
        datasets: &mut Vec<(String, super::ChunkedDataset)>,
        group_stack: &mut Vec<(super::Group, String)>,
    ) -> H5Result<()> {
        let refs = group.object_refs(file).await?;
        for (name, ste) in &refs {
            let child_path = if path == "/" {
                format!("/{name}")
            } else {
                format!("{path}/{name}")
            };

            let mut header: DataObjectHeader =
                read_metadata(file, ste.object_header_address).await?;
            header.load_continuation_messages(file).await?;

            if let Some(result) = header.to_group(name.clone(), file).await {
                let g = result?;
                let hdf5_group = hf.group(&child_path).unwrap();
                compare_attrs(&g.attributes, &hdf5_group, &child_path);
                group_stack.push((g, child_path));
            } else if let Some(ds) = header.to_dataset(name.clone()) {
                let hdf5_ds = hf.dataset(&child_path).unwrap();
                compare_attrs(&ds.attributes, &hdf5_ds, &child_path);
                if let Some(cds) = ds.chunked() {
                    datasets.push((child_path, cds));
                } else {
                    println!("  SKIP {child_path}: non-chunked layout");
                }
            }
        }
        Ok(())
    }

    fn compare_attr_typed<T>(attr: &AttributeMessage, hdf5_attr: &hdf5::Attribute, path: &str)
    where
        T: H5Type + hdf5::H5Type + PartialEq + Debug,
    {
        let ours: Vec<T> = attr.read::<T>();
        let expected: Vec<T> = hdf5_attr.read_raw::<T>().unwrap();
        assert_eq!(ours, expected, "{path}@{}: data mismatch", attr.name());
        println!("  OK {path}@{} ({} values)", attr.name(), ours.len());
    }

    /// FixedAscii<N> requires a compile-time size, so we dispatch via macro.
    macro_rules! compare_fixed_strings {
        ($attr:expr, $hdf5_attr:expr, $path:expr, $( $n:literal ),*) => {
            match $attr.datatype.element_size() {
                $( $n => {
                    let ours = $attr.read_strings();
                    let expected: Vec<hdf5::types::FixedAscii<$n>> =
                        $hdf5_attr.read_raw().unwrap();
                    assert_eq!(
                        ours.len(), expected.len(),
                        "{}@{}: string count mismatch", $path, $attr.name()
                    );
                    for (a, b) in ours.iter().zip(expected.iter()) {
                        assert_eq!(
                            a.as_str(), b.as_str(),
                            "{}@{}: string mismatch", $path, $attr.name()
                        );
                    }
                    println!(
                        "  OK {}@{} ({} string{})", $path, $attr.name(),
                        ours.len(), if ours.len() == 1 { "" } else { "s" }
                    );
                }, )*
                sz => println!(
                    "  SKIP {}@{}: unhandled fixed string size {sz}",
                    $path, $attr.name()
                ),
            }
        };
    }

    fn compare_attr_strings(attr: &AttributeMessage, hdf5_attr: &hdf5::Attribute, path: &str) {
        compare_fixed_strings!(
            attr, hdf5_attr, path, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
            19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 48, 64, 128, 256
        );
    }

    fn compare_attr_untyped(attr: &AttributeMessage, hdf5_attr: &hdf5::Attribute, path: &str) {
        match &attr.datatype.type_desc {
            TypeDescriptor::FixedPoint(fp) => match (fp.signed(), fp.size()) {
                (0, 1) => compare_attr_typed::<u8>(attr, hdf5_attr, path),
                (0, 2) => compare_attr_typed::<u16>(attr, hdf5_attr, path),
                (0, 4) => compare_attr_typed::<u32>(attr, hdf5_attr, path),
                (0, 8) => compare_attr_typed::<u64>(attr, hdf5_attr, path),
                (1, 1) => compare_attr_typed::<i8>(attr, hdf5_attr, path),
                (1, 2) => compare_attr_typed::<i16>(attr, hdf5_attr, path),
                (1, 4) => compare_attr_typed::<i32>(attr, hdf5_attr, path),
                (1, 8) => compare_attr_typed::<i64>(attr, hdf5_attr, path),
                (s, sz) => println!(
                    "  SKIP {path}@{}: unsupported FixedPoint signed={s} size={sz}",
                    attr.name()
                ),
            },
            TypeDescriptor::FloatingPoint(fp) => match fp.size() {
                4 => compare_attr_typed::<f32>(attr, hdf5_attr, path),
                8 => compare_attr_typed::<f64>(attr, hdf5_attr, path),
                sz => println!(
                    "  SKIP {path}@{}: unsupported FloatingPoint size={sz}",
                    attr.name()
                ),
            },
            TypeDescriptor::String(_) => compare_attr_strings(attr, hdf5_attr, path),
            other => {
                println!(
                    "  SKIP {path}@{}: unsupported attr type {other:?}",
                    attr.name()
                );
            }
        }
    }

    fn compare_attrs(attrs: &[AttributeMessage], hdf5_loc: &hdf5::Location, path: &str) {
        for attr in attrs {
            let name = attr.name();
            let hdf5_attr = hdf5_loc.attr(&name).unwrap();
            compare_attr_untyped(attr, &hdf5_attr, path);
        }
    }

    /// Open an HDF5 file, walk all groups, and compare every chunked
    /// dataset against the hdf5 C library.
    async fn compare_file(path: &str) -> H5Result<()> {
        let file = test_file(path);
        let f = super::File::open(&file).await?;
        let hf = hdf5::File::open(path).unwrap();

        // Compare root group attributes
        compare_attrs(&f.root_group.attributes, &hf, "/");

        let mut datasets: Vec<(String, super::ChunkedDataset)> = vec![];
        let mut group_stack: Vec<(super::Group, String)> = vec![];

        collect_from_group(
            &f.root_group,
            "/",
            &file,
            &hf,
            &mut datasets,
            &mut group_stack,
        )
        .await?;
        while let Some((group, gpath)) = group_stack.pop() {
            collect_from_group(&group, &gpath, &file, &hf, &mut datasets, &mut group_stack).await?;
        }

        println!("{path}: found {} chunked datasets", datasets.len());

        for (ds_path, cds) in &datasets {
            let hdf5_ds = hf.dataset(ds_path).unwrap();
            compare_dataset(cds, &hdf5_ds, ds_path, &file).await?;
        }

        Ok(())
    }

    #[tokio::test]
    async fn mol_info_file() -> H5Result<()> {
        compare_file(MOL_INFO_FILE).await
    }

    #[tokio::test]
    async fn matrix_file() -> H5Result<()> {
        compare_file(MATRIX_FILE).await
    }

    /// Helper: do a full read of a 1D chunked dataset, returning the flat data.
    async fn full_read_1d<T: H5Type>(
        cds: &super::ChunkedDataset,
        file: &ObjectStoreFile,
    ) -> H5Result<Vec<T>> {
        Ok(cds.read_full::<T>(file).await?.data)
    }

    /// Open a specific 1D dataset from MOL_INFO_FILE and return it with
    /// the ObjectStoreFile and full reference data.
    async fn setup_range_test(
        dataset_name: &str,
    ) -> H5Result<(ObjectStoreFile, super::ChunkedDataset, Vec<u8>)> {
        let file = test_file(MOL_INFO_FILE);
        let f = super::File::open(&file).await?;
        let obj = f.root_group.find_obj(dataset_name, &file).await?.unwrap();
        let ds = obj.header.to_dataset(dataset_name.to_string()).unwrap();
        let cds = ds.chunked().unwrap();
        let full_data = full_read_1d::<u8>(&cds, &file).await?;
        Ok((file, cds, full_data))
    }

    /// Verify that read_range for a 1D range matches the corresponding
    /// slice of the full exhaustive read.
    fn assert_range_eq(
        result: &super::NdArray<u8>,
        full_data: &[u8],
        range: &std::ops::Range<u64>,
        total: u64,
    ) {
        let start = (range.start.min(total)) as usize;
        let end = (range.end.min(total)) as usize;
        let expected = &full_data[start..end];
        assert_eq!(
            result.shape,
            vec![end - start],
            "range {range:?}: shape mismatch"
        );
        assert_eq!(
            result.data.len(),
            expected.len(),
            "range {range:?}: length mismatch"
        );
        assert_eq!(&result.data[..], expected, "range {range:?}: data mismatch");
    }

    #[tokio::test]
    async fn read_range_basic() -> H5Result<()> {
        let (file, cds, full_data) = setup_range_test("gem_group").await?;
        let total = full_data.len() as u64;
        let chunk_size = cds.chunks_layout.dimension_sizes[0] as u64;
        println!(
            "gem_group: {total} elements, chunk_size={chunk_size}, {} chunks",
            (total + chunk_size - 1) / chunk_size
        );

        let ranges: Vec<std::ops::Range<u64>> = vec![
            // Basic slices
            0..100,
            0..1,
            total - 100..total,
            total - 1..total,
            1000..2000,
            // Full dataset
            0..total,
            // Empty ranges
            0..0,
            100..100,
            total..total,
        ];

        for range in &ranges {
            let result = cds.read_range::<u8>(&[range.clone()], &file).await?;
            assert_range_eq(&result, &full_data, range, total);
            println!("  OK read_range {range:?} ({} values)", result.data.len());
        }

        Ok(())
    }

    #[tokio::test]
    async fn read_range_chunk_boundaries() -> H5Result<()> {
        let (file, cds, full_data) = setup_range_test("gem_group").await?;
        let total = full_data.len() as u64;
        let cs = cds.chunks_layout.dimension_sizes[0] as u64;
        println!("gem_group: {total} elements, chunk_size={cs}");

        let ranges: Vec<std::ops::Range<u64>> = vec![
            // Exactly one chunk
            0..cs,
            cs..2 * cs,
            // Straddle one chunk boundary
            cs - 1..cs + 1,
            cs - 10..cs + 10,
            // Within a single chunk (no boundary crossing)
            10..cs - 10,
            cs + 10..2 * cs - 10,
            // Exactly two chunks
            0..2 * cs,
            // Exactly three chunks
            cs..4 * cs,
            // Start at chunk boundary
            cs..cs + 50,
            2 * cs..2 * cs + 1,
            // End at chunk boundary
            50..cs,
            cs + 50..2 * cs,
            // Range near end of dataset (edge chunk handling)
            total - cs..total,
            total - 1..total,
            total - cs / 2..total,
        ];

        for range in &ranges {
            let result = cds.read_range::<u8>(&[range.clone()], &file).await?;
            assert_range_eq(&result, &full_data, range, total);
            println!("  OK read_range {range:?} ({} values)", result.data.len());
        }

        Ok(())
    }

    #[tokio::test]
    async fn read_range_various_sizes() -> H5Result<()> {
        let (file, cds, full_data) = setup_range_test("gem_group").await?;
        let total = full_data.len() as u64;
        let cs = cds.chunks_layout.dimension_sizes[0] as u64;

        // Test power-of-two sizes and offsets
        let sizes = [
            1,
            2,
            7,
            63,
            64,
            65,
            1023,
            1024,
            1025,
            cs - 1,
            cs,
            cs + 1,
            cs * 3 + 7,
        ];
        let offsets = [0, 1, cs - 1, cs, cs + 1, total / 2];

        for &offset in &offsets {
            for &size in &sizes {
                let end = (offset + size).min(total);
                if offset >= total {
                    continue;
                }
                let range = offset..end;
                let result = cds.read_range::<u8>(&[range.clone()], &file).await?;
                assert_range_eq(&result, &full_data, &range, total);
            }
        }

        println!(
            "  OK read_range_various_sizes ({} combinations)",
            offsets.len() * sizes.len()
        );
        Ok(())
    }

    #[tokio::test]
    async fn read_range_clamping() -> H5Result<()> {
        let (file, cds, full_data) = setup_range_test("gem_group").await?;
        let total = full_data.len() as u64;

        // Range extends past dataset extent — should be clamped
        let range = total - 10..total + 100;
        let result = cds.read_range::<u8>(&[range.clone()], &file).await?;
        assert_range_eq(&result, &full_data, &range, total);
        println!(
            "  OK clamped range {range:?} → {} values (expected 10)",
            result.data.len()
        );

        // Completely past the end
        let range = total..total + 100;
        let result = cds.read_range::<u8>(&[range.clone()], &file).await?;
        assert_eq!(result.data.len(), 0);
        assert_eq!(result.shape, vec![0]);
        println!("  OK fully out-of-bounds range → empty");

        Ok(())
    }

    #[tokio::test]
    async fn read_range_u32_dataset() -> H5Result<()> {
        // Test read_range on a u32 dataset (barcode_corrected_reads)
        let file = test_file(MOL_INFO_FILE);
        let f = super::File::open(&file).await?;
        let obj = f
            .root_group
            .find_obj("barcode_corrected_reads", &file)
            .await?
            .unwrap();
        let ds = obj
            .header
            .to_dataset("barcode_corrected_reads".to_string())
            .unwrap();
        let cds = ds.chunked().unwrap();

        let full_data = full_read_1d::<u32>(&cds, &file).await?;
        let total = full_data.len() as u64;
        let cs = cds.chunks_layout.dimension_sizes[0] as u64;
        println!(
            "barcode_corrected_reads: {total} u32 values, chunk_size={cs}, filtered={}",
            cds.filter.is_some()
        );

        let ranges: Vec<std::ops::Range<u64>> = vec![
            0..100,
            cs - 5..cs + 5,
            total - 50..total,
            cs * 2..cs * 2 + 1000,
        ];

        for range in &ranges {
            let result = cds.read_range::<u32>(&[range.clone()], &file).await?;
            let start = range.start as usize;
            let end = range.end.min(total) as usize;
            let expected = &full_data[start..end];
            assert_eq!(
                result.shape,
                vec![end - start],
                "range {range:?}: shape mismatch"
            );
            assert_eq!(&result.data[..], expected, "range {range:?}: data mismatch");
            println!("  OK read_range {range:?} ({} values)", result.data.len());
        }

        Ok(())
    }

    #[tokio::test]
    async fn read_range_u64_dataset() -> H5Result<()> {
        // Test read_range on a u64 dataset (barcode) to cover larger element types
        let file = test_file(MOL_INFO_FILE);
        let f = super::File::open(&file).await?;
        let obj = f.root_group.find_obj("barcode", &file).await?.unwrap();
        let ds = obj.header.to_dataset("barcode".to_string()).unwrap();
        let cds = ds.chunked().unwrap();

        let full_data = full_read_1d::<u64>(&cds, &file).await?;
        let total = full_data.len() as u64;
        let cs = cds.chunks_layout.dimension_sizes[0] as u64;
        println!("barcode: {total} u64 values, chunk_size={cs}");

        let ranges: Vec<std::ops::Range<u64>> =
            vec![0..50, cs - 1..cs + 1, total - 10..total, cs..cs * 2 + 7];

        for range in &ranges {
            let result = cds.read_range::<u64>(&[range.clone()], &file).await?;
            let start = range.start as usize;
            let end = range.end.min(total) as usize;
            let expected = &full_data[start..end];
            assert_eq!(
                result.shape,
                vec![end - start],
                "range {range:?}: shape mismatch"
            );
            assert_eq!(&result.data[..], expected, "range {range:?}: data mismatch");
            println!("  OK read_range {range:?} ({} values)", result.data.len());
        }

        Ok(())
    }

    #[tokio::test]
    async fn read_range_chunk_skip_count() -> H5Result<()> {
        // Verify that read_range skips chunks outside the selection.
        // We do this by comparing the number of chunks that overlap with
        // the selection vs the total chunk count.
        let (file, cds, _full_data) = setup_range_test("gem_group").await?;
        let total = cds.dataspace.dimension[0];
        let cs = cds.chunks_layout.dimension_sizes[0] as u64;
        let all_chunks = cds.collect_chunks(&file).await?;
        let total_chunks = all_chunks.len();

        // A range covering ~3 chunks in the middle
        let start = cs * 10;
        let end = cs * 13;
        let range = start..end;

        let overlapping = all_chunks
            .iter()
            .filter(|c| {
                let co = c.key.offsets[0];
                let ce = (co + cs).min(total);
                co < end && ce > start
            })
            .count();

        let result = cds.read_range::<u8>(&[range.clone()], &file).await?;
        assert_eq!(result.data.len(), (end - start) as usize);

        println!("  range {range:?}: {overlapping} chunks needed out of {total_chunks} total");
        assert!(
            overlapping < total_chunks,
            "expected fewer chunks than total"
        );
        assert_eq!(overlapping, 3, "expected exactly 3 chunks for this range");

        Ok(())
    }

    // ---- Performance comparison tests ----

    fn median(times: &[std::time::Duration]) -> std::time::Duration {
        let mut sorted: Vec<_> = times.to_vec();
        sorted.sort();
        sorted[sorted.len() / 2]
    }

    fn fmt_duration(d: std::time::Duration) -> String {
        let ms = d.as_secs_f64() * 1000.0;
        if ms >= 1000.0 {
            format!("{:.2}s", ms / 1000.0)
        } else {
            format!("{:.1}ms", ms)
        }
    }

    struct BenchResult {
        name: String,
        h5rs_times: Vec<std::time::Duration>,
        hdf5_times: Vec<std::time::Duration>,
    }

    impl BenchResult {
        fn print(&self) {
            let h = median(&self.h5rs_times);
            let c = median(&self.hdf5_times);
            let ratio = h.as_secs_f64() / c.as_secs_f64();
            let all_h: Vec<_> = self.h5rs_times.iter().map(|t| fmt_duration(*t)).collect();
            let all_c: Vec<_> = self.hdf5_times.iter().map(|t| fmt_duration(*t)).collect();
            println!("  {}", self.name);
            println!("    h5rs: {:>9}  [{}]", fmt_duration(h), all_h.join(", "));
            println!("    hdf5: {:>9}  [{}]", fmt_duration(c), all_c.join(", "));
            println!("    ratio: {ratio:.2}x");
        }
    }

    #[tokio::test]
    #[ignore] // Run with: cargo test perf -- --ignored --nocapture
    #[allow(unused)]
    async fn perf() -> H5Result<()> {
        use std::time::Instant;

        const ITERS: usize = 5;

        // --- warm page cache ---
        println!("Warming page cache...");
        {
            let buf = std::fs::read(MOL_INFO_FILE).unwrap();
            println!("  {} bytes read", buf.len());
            std::hint::black_box(&buf);
        }

        // --- open files once ---
        let osf = test_file(MOL_INFO_FILE);
        let f = super::File::open(&osf).await?;
        let hf = hdf5::File::open(MOL_INFO_FILE).unwrap();

        // --- pre-open datasets ---
        // gem_group: u8, ~34.7M values, unfiltered
        let gem_obj = f.root_group.find_obj("gem_group", &osf).await?.unwrap();
        let gem_ds = gem_obj.header.to_dataset("gem_group".to_string()).unwrap();
        let gem_cds = gem_ds.chunked().unwrap();
        let gem_hdf5 = hf.dataset("/gem_group").unwrap();
        let gem_total = gem_cds.dataspace.dimension[0];

        // barcode_corrected_reads: u32, ~34.7M values, gzip+shuffle
        let bcr_obj = f
            .root_group
            .find_obj("barcode_corrected_reads", &osf)
            .await?
            .unwrap();
        let bcr_ds = bcr_obj
            .header
            .to_dataset("barcode_corrected_reads".to_string())
            .unwrap();
        let bcr_cds = bcr_ds.chunked().unwrap();
        let bcr_hdf5 = hf.dataset("/barcode_corrected_reads").unwrap();

        // barcode: u64, ~34.7M values
        let bc_obj = f.root_group.find_obj("barcode", &osf).await?.unwrap();
        let bc_ds = bc_obj.header.to_dataset("barcode".to_string()).unwrap();
        let bc_cds = bc_ds.chunked().unwrap();
        let bc_hdf5 = hf.dataset("/barcode").unwrap();

        let mut results: Vec<BenchResult> = Vec::new();

        // ========================================================
        // Full array reads
        // ========================================================

        // 1. Full read: gem_group (u8, unfiltered)
        if false {
            let mut h5rs_t = Vec::new();
            let mut hdf5_t = Vec::new();
            for _ in 0..ITERS {
                let t = Instant::now();
                let d = full_read_1d::<u8>(&gem_cds, &osf).await?;
                h5rs_t.push(t.elapsed());
                std::hint::black_box(&d);

                let t = Instant::now();
                let d: Vec<u8> = gem_hdf5.read_raw().unwrap();
                hdf5_t.push(t.elapsed());
                std::hint::black_box(&d);
            }
            results.push(BenchResult {
                name: format!(
                    "Full read gem_group (u8, {}M, {})",
                    gem_total / 1_000_000,
                    if gem_cds.filter.is_some() {
                        "filtered"
                    } else {
                        "no filter"
                    }
                ),
                h5rs_times: h5rs_t,
                hdf5_times: hdf5_t,
            });
        }

        // 2. Full read: barcode_corrected_reads (u32, gzip+shuffle)
        if false {
            let mut h5rs_t = Vec::new();
            let mut hdf5_t = Vec::new();
            for _ in 0..ITERS {
                let t = Instant::now();
                let d = full_read_1d::<u32>(&bcr_cds, &osf).await?;
                h5rs_t.push(t.elapsed());
                std::hint::black_box(&d);

                let t = Instant::now();
                let d: Vec<u32> = bcr_hdf5.read_raw().unwrap();
                hdf5_t.push(t.elapsed());
                std::hint::black_box(&d);
            }
            results.push(BenchResult {
                name: format!(
                    "Full read barcode_corrected_reads (u32, {}M, {})",
                    bcr_cds.dataspace.dimension[0] / 1_000_000,
                    if bcr_cds.filter.is_some() {
                        "filtered"
                    } else {
                        "no filter"
                    }
                ),
                h5rs_times: h5rs_t,
                hdf5_times: hdf5_t,
            });
        }

        // 3. Full read: barcode (u64, large elements)
        {
            let mut h5rs_t = Vec::new();
            let mut hdf5_t = Vec::new();
            for _ in 0..30 {
                let t = Instant::now();
                let d = full_read_1d::<u64>(&bc_cds, &osf).await?;
                h5rs_t.push(t.elapsed());
                std::hint::black_box(&d);

                let t = Instant::now();
                let d: Vec<u64> = bc_hdf5.read_raw().unwrap();
                hdf5_t.push(t.elapsed());
                std::hint::black_box(&d);
            }
            results.push(BenchResult {
                name: format!(
                    "Full read barcode (u64, {}M, {})",
                    bc_cds.dataspace.dimension[0] / 1_000_000,
                    if bc_cds.filter.is_some() {
                        "filtered"
                    } else {
                        "no filter"
                    }
                ),
                h5rs_times: h5rs_t,
                hdf5_times: hdf5_t,
            });
        }

        // ========================================================
        // Range reads
        // ========================================================

        // 4. Small range: 1M elements from middle of gem_group
        if false {
            let start = gem_total / 2;
            let end = start + 1_000_000;
            let mut h5rs_t = Vec::new();
            let mut hdf5_t = Vec::new();
            for _ in 0..ITERS {
                let t = Instant::now();
                let d = gem_cds.read_range::<u8>(&[start..end], &osf).await?;
                h5rs_t.push(t.elapsed());
                std::hint::black_box(&d);

                let t = Instant::now();
                let d: ndarray::Array1<u8> = gem_hdf5
                    .read_slice_1d(start as usize..end as usize)
                    .unwrap();
                hdf5_t.push(t.elapsed());
                std::hint::black_box(&d);
            }
            results.push(BenchResult {
                name: "Range read gem_group 1M elements (u8, no filter)".to_string(),
                h5rs_times: h5rs_t,
                hdf5_times: hdf5_t,
            });
        }

        // 5. Large range: 10M elements from gem_group
        if false {
            let start = gem_total / 4;
            let end = start + 10_000_000;
            let mut h5rs_t = Vec::new();
            let mut hdf5_t = Vec::new();
            for _ in 0..ITERS {
                let t = Instant::now();
                let d = gem_cds.read_range::<u8>(&[start..end], &osf).await?;
                h5rs_t.push(t.elapsed());
                std::hint::black_box(&d);

                let t = Instant::now();
                let d: ndarray::Array1<u8> = gem_hdf5
                    .read_slice_1d(start as usize..end as usize)
                    .unwrap();
                hdf5_t.push(t.elapsed());
                std::hint::black_box(&d);
            }
            results.push(BenchResult {
                name: "Range read gem_group 10M elements (u8, no filter)".to_string(),
                h5rs_times: h5rs_t,
                hdf5_times: hdf5_t,
            });
        }

        // 6. Range read: 1M elements from barcode_corrected_reads (filtered)
        if false {
            let start = bcr_cds.dataspace.dimension[0] / 2;
            let end = start + 1_000_000;
            let mut h5rs_t = Vec::new();
            let mut hdf5_t = Vec::new();
            for _ in 0..ITERS {
                let t = Instant::now();
                let d = bcr_cds.read_range::<u32>(&[start..end], &osf).await?;
                h5rs_t.push(t.elapsed());
                std::hint::black_box(&d);

                let t = Instant::now();
                let d: ndarray::Array1<u32> = bcr_hdf5
                    .read_slice_1d(start as usize..end as usize)
                    .unwrap();
                hdf5_t.push(t.elapsed());
                std::hint::black_box(&d);
            }
            results.push(BenchResult {
                name: "Range read barcode_corrected_reads 1M (u32, gzip+shuffle)".to_string(),
                h5rs_times: h5rs_t,
                hdf5_times: hdf5_t,
            });
        }

        // ========================================================
        // Attribute reads
        // ========================================================

        // Helper macro: read an attribute with correct h5rs type dispatch
        macro_rules! bench_read_attr_h5rs {
            ($attr:expr) => {
                match &$attr.datatype.type_desc {
                    TypeDescriptor::FixedPoint(fp) => match (fp.signed(), fp.size()) {
                        (0, 1) => {
                            std::hint::black_box($attr.read::<u8>());
                        }
                        (0, 2) => {
                            std::hint::black_box($attr.read::<u16>());
                        }
                        (0, 4) => {
                            std::hint::black_box($attr.read::<u32>());
                        }
                        (0, 8) => {
                            std::hint::black_box($attr.read::<u64>());
                        }
                        (1, 1) => {
                            std::hint::black_box($attr.read::<i8>());
                        }
                        (1, 2) => {
                            std::hint::black_box($attr.read::<i16>());
                        }
                        (1, 4) => {
                            std::hint::black_box($attr.read::<i32>());
                        }
                        (1, 8) => {
                            std::hint::black_box($attr.read::<i64>());
                        }
                        _ => {}
                    },
                    TypeDescriptor::FloatingPoint(fp) => match fp.size() {
                        4 => {
                            std::hint::black_box($attr.read::<f32>());
                        }
                        8 => {
                            std::hint::black_box($attr.read::<f64>());
                        }
                        _ => {}
                    },
                    TypeDescriptor::String(_) => {
                        std::hint::black_box($attr.read_strings());
                    }
                    _ => {}
                }
            };
        }

        // Helper macro: read an attribute with correct hdf5 type dispatch
        macro_rules! bench_read_attr_hdf5 {
            ($attr:expr, $ha:expr) => {
                match &$attr.datatype.type_desc {
                    TypeDescriptor::FixedPoint(fp) => match (fp.signed(), fp.size()) {
                        (0, 1) => {
                            std::hint::black_box($ha.read_raw::<u8>().unwrap());
                        }
                        (0, 2) => {
                            std::hint::black_box($ha.read_raw::<u16>().unwrap());
                        }
                        (0, 4) => {
                            std::hint::black_box($ha.read_raw::<u32>().unwrap());
                        }
                        (0, 8) => {
                            std::hint::black_box($ha.read_raw::<u64>().unwrap());
                        }
                        (1, 1) => {
                            std::hint::black_box($ha.read_raw::<i8>().unwrap());
                        }
                        (1, 2) => {
                            std::hint::black_box($ha.read_raw::<i16>().unwrap());
                        }
                        (1, 4) => {
                            std::hint::black_box($ha.read_raw::<i32>().unwrap());
                        }
                        (1, 8) => {
                            std::hint::black_box($ha.read_raw::<i64>().unwrap());
                        }
                        _ => {}
                    },
                    TypeDescriptor::FloatingPoint(fp) => match fp.size() {
                        4 => {
                            std::hint::black_box($ha.read_raw::<f32>().unwrap());
                        }
                        8 => {
                            std::hint::black_box($ha.read_raw::<f64>().unwrap());
                        }
                        _ => {}
                    },
                    TypeDescriptor::String(_) => {
                        // skip strings for hdf5 C lib (FixedAscii size dispatch too complex)
                    }
                    _ => {}
                }
            };
        }

        // 7. Read all numeric/string attributes from root group (100x)
        {
            const ATTR_ITERS: usize = 100;
            let root_attrs: Vec<_> = f.root_group.attributes.iter().collect();
            let mut h5rs_t = Vec::new();
            let mut hdf5_t = Vec::new();
            for _ in 0..ITERS {
                let t = Instant::now();
                for _ in 0..ATTR_ITERS {
                    for attr in &root_attrs {
                        bench_read_attr_h5rs!(attr);
                    }
                }
                h5rs_t.push(t.elapsed());

                let t = Instant::now();
                for _ in 0..ATTR_ITERS {
                    for attr in &root_attrs {
                        let ha = hf.attr(&attr.name()).unwrap();
                        bench_read_attr_hdf5!(attr, ha);
                    }
                }
                hdf5_t.push(t.elapsed());
            }
            results.push(BenchResult {
                name: format!(
                    "Attribute reads, root group ({} attrs x {ATTR_ITERS} iters)",
                    root_attrs.len()
                ),
                h5rs_times: h5rs_t,
                hdf5_times: hdf5_t,
            });
        }

        // 8. Read all attributes from all datasets (100x)
        {
            const ATTR_ITERS: usize = 100;
            let all_ds_attrs: Vec<(&str, &[AttributeMessage])> = vec![
                ("gem_group", &gem_ds.attributes),
                ("barcode_corrected_reads", &bcr_ds.attributes),
                ("barcode", &bc_ds.attributes),
            ];
            let total_attrs: usize = all_ds_attrs.iter().map(|(_, a)| a.len()).sum();
            if total_attrs > 0 {
                let mut h5rs_t = Vec::new();
                let mut hdf5_t = Vec::new();
                for _ in 0..ITERS {
                    let t = Instant::now();
                    for _ in 0..ATTR_ITERS {
                        for (_, attrs) in &all_ds_attrs {
                            for attr in *attrs {
                                bench_read_attr_h5rs!(attr);
                            }
                        }
                    }
                    h5rs_t.push(t.elapsed());

                    let t = Instant::now();
                    for _ in 0..ATTR_ITERS {
                        for (ds_name, attrs) in &all_ds_attrs {
                            let hds = hf.dataset(&format!("/{ds_name}")).unwrap();
                            for attr in *attrs {
                                let ha = hds.attr(&attr.name()).unwrap();
                                bench_read_attr_hdf5!(attr, ha);
                            }
                        }
                    }
                    hdf5_t.push(t.elapsed());
                }
                results.push(BenchResult {
                    name: format!(
                        "Attribute reads, datasets ({total_attrs} attrs x {ATTR_ITERS} iters)"
                    ),
                    h5rs_times: h5rs_t,
                    hdf5_times: hdf5_t,
                });
            }
        }

        // ========================================================
        // Print results
        // ========================================================
        println!("\n{:=<72}", "");
        println!("PERF RESULTS  ({ITERS} iterations, alternating h5rs/hdf5)");
        println!("{:=<72}", "");
        for r in &results {
            r.print();
            println!();
        }

        Ok(())
    }
}

#[cfg(test)]
mod roundtrip {
    use std::fmt::Debug;

    use object_store::{local::LocalFileSystem, path::Path};

    use crate::error::H5Result;
    use crate::h5type::H5Type;
    use crate::object_store::ObjectStoreFile;

    fn test_file_abs(path: &std::path::Path) -> ObjectStoreFile {
        let parent = path.parent().unwrap();
        let filename = path.file_name().unwrap().to_str().unwrap();
        let store = LocalFileSystem::new_with_prefix(parent).unwrap();
        ObjectStoreFile::new(Box::new(store), Path::from(filename))
    }

    /// Trait for generating deterministic test values from a flat index.
    trait TestValue: Sized {
        fn from_index(i: usize) -> Self;
    }

    macro_rules! impl_test_value {
        ($($ty:ty),*) => {
            $(impl TestValue for $ty {
                fn from_index(i: usize) -> Self { (i % 251) as $ty }
            })*
        };
    }

    impl_test_value!(u8, u16, u32, u64, i8, i16, i32, i64, f32, f64);

    /// Extract elements from a flat row-major array at the given N-dimensional sub-range.
    fn collect_subrange<T: Copy>(
        data: &[T],
        shape: &[usize],
        ranges: &[std::ops::Range<u64>],
    ) -> Vec<T> {
        let ndim = shape.len();
        let range_shape: Vec<usize> = ranges.iter().map(|r| (r.end - r.start) as usize).collect();
        let total: usize = range_shape.iter().product();

        (0..total)
            .map(|linear| {
                let mut remaining = linear;
                let mut src_linear = 0usize;
                for d in 0..ndim {
                    let range_stride: usize = range_shape[d + 1..].iter().product();
                    let idx_in_range = remaining / range_stride;
                    remaining %= range_stride;
                    let src_dim_idx = ranges[d].start as usize + idx_in_range;
                    let src_stride: usize = shape[d + 1..].iter().product();
                    src_linear += src_dim_idx * src_stride;
                }
                data[src_linear]
            })
            .collect()
    }

    struct RoundtripTest {
        shape: Vec<usize>,
        chunk: Option<Vec<usize>>,
        deflate: Option<u8>,
        shuffle: bool,
    }

    impl RoundtripTest {
        fn new() -> Self {
            Self {
                shape: vec![],
                chunk: None,
                deflate: None,
                shuffle: false,
            }
        }

        fn shape(mut self, s: &[usize]) -> Self {
            self.shape = s.to_vec();
            self
        }

        fn chunk(mut self, c: &[usize]) -> Self {
            self.chunk = Some(c.to_vec());
            self
        }

        fn deflate(mut self, level: u8) -> Self {
            self.deflate = Some(level);
            self
        }

        fn shuffle(mut self) -> Self {
            self.shuffle = true;
            self
        }

        async fn run<T>(&self) -> H5Result<()>
        where
            T: H5Type + hdf5::H5Type + TestValue + PartialEq + Debug,
        {
            let tmp = tempfile::NamedTempFile::new().unwrap();
            let path = tmp.path();

            let total: usize = self.shape.iter().product();
            let data: Vec<T> = (0..total).map(T::from_index).collect();

            // Write with hdf5-rust
            {
                let hf = hdf5::File::create(path).unwrap();
                let mut builder = hf.new_dataset::<T>().shape(&self.shape[..]);
                if let Some(ref c) = self.chunk {
                    builder = builder.chunk(&c[..]);
                }
                if self.shuffle {
                    builder = builder.shuffle();
                }
                if let Some(level) = self.deflate {
                    builder = builder.deflate(level);
                }
                let ds = builder.create("data").unwrap();
                ds.write_raw(&data).unwrap();
            }

            // Read with h5rs
            let file = test_file_abs(path);
            let f = super::File::open(&file).await?;
            let obj = f.root_group.find_obj("data", &file).await?.unwrap();
            let ds = obj.header.to_dataset("data".to_string()).unwrap();
            let cds = ds.chunked().unwrap();

            // Full read
            let result = cds.read_full::<T>(&file).await?;
            assert_eq!(result.shape, self.shape, "shape mismatch");
            assert_eq!(&result.data[..], &data[..], "data mismatch");

            // Sub-range read (middle quarter in each dimension)
            if total > 0 {
                let sel: Vec<std::ops::Range<u64>> = self
                    .shape
                    .iter()
                    .map(|&d| {
                        let start = (d / 4) as u64;
                        let end = (3 * d / 4).max(d / 4 + 1) as u64;
                        start..end
                    })
                    .collect();

                let range_result = cds.read_range::<T>(&sel, &file).await?;
                let expected_shape: Vec<usize> =
                    sel.iter().map(|r| (r.end - r.start) as usize).collect();
                assert_eq!(range_result.shape, expected_shape, "range shape mismatch");

                let expected_data = collect_subrange(&data, &self.shape, &sel);
                assert_eq!(
                    &range_result.data[..],
                    &expected_data[..],
                    "range data mismatch"
                );
            }

            Ok(())
        }
    }

    macro_rules! roundtrip {
        ($name:ident, $T:ty, $builder:expr) => {
            #[tokio::test]
            async fn $name() -> H5Result<()> {
                $builder.run::<$T>().await
            }
        };
    }

    // -- Dimensions --
    roundtrip!(
        dim_1d,
        u32,
        RoundtripTest::new().shape(&[10000]).chunk(&[1000])
    );
    roundtrip!(
        dim_2d,
        f64,
        RoundtripTest::new().shape(&[100, 200]).chunk(&[32, 64])
    );
    roundtrip!(
        dim_3d,
        u8,
        RoundtripTest::new().shape(&[10, 20, 30]).chunk(&[4, 8, 16])
    );

    // -- Filters --
    roundtrip!(
        filter_none,
        u32,
        RoundtripTest::new().shape(&[10000]).chunk(&[1000])
    );
    roundtrip!(
        filter_deflate,
        u32,
        RoundtripTest::new()
            .shape(&[10000])
            .chunk(&[1000])
            .deflate(1)
    );
    roundtrip!(
        filter_shuffle_deflate,
        u32,
        RoundtripTest::new()
            .shape(&[10000])
            .chunk(&[1000])
            .shuffle()
            .deflate(1)
    );

    // -- Data types --
    roundtrip!(
        type_u8,
        u8,
        RoundtripTest::new().shape(&[5000]).chunk(&[500]).deflate(1)
    );
    roundtrip!(
        type_u32,
        u32,
        RoundtripTest::new().shape(&[5000]).chunk(&[500]).deflate(1)
    );
    roundtrip!(
        type_u64,
        u64,
        RoundtripTest::new().shape(&[5000]).chunk(&[500]).deflate(1)
    );
    roundtrip!(
        type_i32,
        i32,
        RoundtripTest::new().shape(&[5000]).chunk(&[500]).deflate(1)
    );
    roundtrip!(
        type_f32,
        f32,
        RoundtripTest::new().shape(&[5000]).chunk(&[500]).deflate(1)
    );
    roundtrip!(
        type_f64,
        f64,
        RoundtripTest::new().shape(&[5000]).chunk(&[500]).deflate(1)
    );

    // -- Edge cases --
    roundtrip!(
        edge_partial_chunks,
        u32,
        RoundtripTest::new().shape(&[1003]).chunk(&[100])
    );
    roundtrip!(
        edge_single_element_chunks,
        u32,
        RoundtripTest::new().shape(&[100]).chunk(&[1])
    );
    roundtrip!(
        edge_chunk_equals_dim,
        u32,
        RoundtripTest::new().shape(&[50]).chunk(&[50])
    );
    roundtrip!(
        edge_single_element,
        u32,
        RoundtripTest::new().shape(&[1]).chunk(&[1])
    );
}
