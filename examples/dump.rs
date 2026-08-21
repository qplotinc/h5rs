//! List the datasets in an HDF5 file, and optionally read one.
//!
//! ```bash
//! cargo run --example dump -- path/to/file.h5
//! cargo run --example dump -- path/to/file.h5 /group/dataset
//! ```

use h5rs::object_store::ObjectStoreFile;
use object_store::{local::LocalFileSystem, path::Path};

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let Some(path) = args.next() else {
        eprintln!("usage: dump <file.h5> [/internal/path]");
        std::process::exit(2);
    };
    let wanted = args.next();

    let path = std::path::Path::new(&path).canonicalize()?;
    let dir = path.parent().ok_or("file has no parent directory")?;
    let name = path.file_name().ok_or("not a file")?.to_string_lossy();
    let store = LocalFileSystem::new_with_prefix(dir)?;
    let file = ObjectStoreFile::new(Box::new(store), Path::from(name.as_ref()));

    for info in h5rs::list_datasets(&file).await? {
        let (is_float, is_signed, bytes) = info.dtype_info;
        let kind = match (is_float, is_signed) {
            (true, _) => "float",
            (false, true) => "int",
            (false, false) => "uint",
        };
        println!(
            "{:40} shape={:?} {} chunks={:?} type={kind}{} filters={:?}",
            info.path,
            info.shape,
            info.layout,
            info.chunk_shape,
            bytes * 8,
            info.filters
        );
    }

    report_io(&file, "listing");

    let Some(wanted) = wanted else {
        return Ok(());
    };
    let segments: Vec<&str> = wanted.trim_start_matches('/').split('/').collect();
    let ds = h5rs::open_dataset(&file, &segments)
        .await?
        .ok_or("no such dataset")?;

    let (is_float, is_signed, bytes) = ds.dtype_info();
    println!(
        "\n{wanted}: shape {:?}, {} layout, chunks {:?}{}",
        ds.shape(),
        ds.layout_name(),
        ds.chunk_shape(),
        match ds.chunk_index_name() {
            Some(index) => format!(", {index} index"),
            None => String::new(),
        }
    );
    match (is_float, is_signed, bytes) {
        (false, false, 1) => print_head(ds.read_full::<u8>(&file).await?.data),
        (false, false, 4) => print_head(ds.read_full::<u32>(&file).await?.data),
        (false, false, 8) => print_head(ds.read_full::<u64>(&file).await?.data),
        (false, true, 4) => print_head(ds.read_full::<i32>(&file).await?.data),
        (false, true, 8) => print_head(ds.read_full::<i64>(&file).await?.data),
        (true, _, 4) => print_head(ds.read_full::<f32>(&file).await?.data),
        (true, _, 8) => print_head(ds.read_full::<f64>(&file).await?.data),
        _ => println!("(no printer for this element type)"),
    }
    report_io(&file, "total");
    Ok(())
}

/// Show what the read cost in requests and bytes, which is what matters when
/// the file lives in object storage rather than on local disk.
fn report_io(file: &ObjectStoreFile, label: &str) {
    let s = file.stats();
    println!(
        "[io] {label}: {} round trips, {} requests, {:.2} MiB read, {} cache hits",
        s.batches,
        s.requests,
        s.bytes_fetched as f64 / (1024.0 * 1024.0),
        s.cache_hits
    );
}

fn print_head<T: std::fmt::Debug>(data: Vec<T>) {
    println!(
        "{} values, first 10: {:?}",
        data.len(),
        &data[..data.len().min(10)]
    );
}
