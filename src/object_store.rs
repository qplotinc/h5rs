//! Byte-range access to an HDF5 file living in an [`object_store`] backend.
//!
//! [`ObjectStoreFile`] pairs an [`ObjectStore`] implementation with the path of
//! a single object, and exposes the small set of ranged-read operations the
//! rest of the crate needs. Everything h5rs does — parsing the superblock,
//! walking B-trees, fetching chunks — is expressed as ranged GETs against this
//! type, which is what makes the reader usable over HTTP and object storage.
//!
//! # Round trips
//!
//! Against object storage a request costs far more than the bytes it carries: a
//! round trip is tens of milliseconds, while a few hundred kilobytes is a small
//! fraction of one. So this module trades bytes for requests in two ways.
//!
//! Metadata — superblocks, object headers, B-tree nodes, heaps, chunk indexes —
//! is read through a cache of aligned blocks of
//! [`ReadOptions::metadata_block_size`]. HDF5 writes these structures in
//! clusters, so one block typically satisfies many subsequent reads.
//!
//! Bulk data is read through [`ObjectStoreFile::read_ranges`], which merges
//! ranges that lie close together into single requests and lets the store issue
//! what remains in parallel.
//!
//! [`ObjectStoreFile::stats`] reports what this actually cost.

use std::collections::HashMap;
use std::io::Cursor;
use std::ops::Range;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use binrw::BinRead;
use bytes::Bytes;
use futures_core::stream::BoxStream;
use object_store::{GetOptions, GetRange, ObjectMeta};
use object_store::{ObjectStore, path::Path};

use crate::error::H5Result;

/// How much to read when the size of a metadata structure is not yet known.
/// This is a parse window, not a request size: it is served out of the block
/// cache, and costs a request only when it falls outside every cached block.
pub(crate) const METADATA_FETCH_SIZE: u64 = 8192;

/// Tunables for how aggressively h5rs reads ahead and batches requests.
///
/// The defaults are chosen for object storage, where latency dominates. Reading
/// from a local file, where a request is nearly free, is not harmed by them but
/// gains little either.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReadOptions {
    /// Metadata is fetched in aligned blocks of this size, so that following a
    /// pointer to a nearby structure costs no further request.
    ///
    /// The default of 512 KiB was chosen by measuring round trips and bytes
    /// across several file shapes; past it the round-trip count stops improving
    /// while the bytes keep growing.
    pub metadata_block_size: u64,
    /// Upper bound on the metadata block cache, in bytes. Once reached, the
    /// least recently used blocks are dropped.
    pub cache_capacity: u64,
    /// How large a merged request is allowed to grow.
    ///
    /// This is a ceiling on *coalescing*, not on request size: it decides when
    /// neighbouring ranges stop being worth combining. A single range larger
    /// than this is still fetched whole in one request — a chunk is never split
    /// across requests, since none of it could be decoded until every piece
    /// arrived.
    ///
    /// Merging saves round trips, but nothing in a request can be decoded until
    /// all of it has landed, so merging too eagerly starves the decoder at the
    /// start of a read.
    pub max_coalesced_bytes: u64,
    /// How many bulk requests to keep in flight at once.
    ///
    /// This is what saturates a high-latency link: with a round trip of `t` and
    /// a request that transfers in `d`, roughly `1 + t/d` requests are needed
    /// to keep the pipe full.
    pub io_concurrency: usize,
    /// Ceiling on the bytes a read holds: requests in flight, plus data that
    /// has arrived but not yet been decoded and copied out.
    ///
    /// Concurrency alone does not bound memory, because a single chunk can be
    /// arbitrarily large. This does — but only down to a floor of two requests,
    /// which are always allowed through so that there is something downloading
    /// while something else decodes. Reading a dataset whose chunks are each
    /// larger than this therefore holds about two chunks, not this many bytes.
    ///
    /// With large chunks this, rather than
    /// [`io_concurrency`](Self::io_concurrency), is usually what decides how
    /// many requests are outstanding — and a modern SSD needs several to reach
    /// its rated throughput. Raise it if reads are shallow.
    pub max_inflight_bytes: u64,
    /// Two wanted ranges no further apart than this are fetched as one request,
    /// paying for the bytes in between to save a round trip. Bounded by
    /// [`max_coalesced_bytes`](Self::max_coalesced_bytes).
    ///
    /// Adjacent chunks have no gap at all, so this only decides how much
    /// unwanted data is worth pulling to avoid a request. Set it near
    /// `round_trip_time * bandwidth`.
    pub coalesce_gap: u64,
}

impl Default for ReadOptions {
    fn default() -> Self {
        ReadOptions {
            metadata_block_size: 512 * 1024,
            cache_capacity: 32 * 1024 * 1024,
            max_coalesced_bytes: 512 * 1024,
            io_concurrency: 16,
            max_inflight_bytes: 32 * 1024 * 1024,
            coalesce_gap: 64 * 1024,
        }
    }
}

/// A snapshot of what reading a file has cost so far.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct IoStats {
    /// Times h5rs waited on the store. Requests inside one wait are issued
    /// together, so this is the number of sequential round trips — the figure
    /// that decides how long a read takes over a high-latency link.
    pub batches: u64,
    /// Requests issued to the object store.
    pub requests: u64,
    /// Bytes those requests transferred.
    pub bytes_fetched: u64,
    /// Metadata reads served entirely from the block cache.
    pub cache_hits: u64,
    /// Metadata reads that needed at least one request.
    pub cache_misses: u64,
}

#[derive(Debug, Default)]
struct Counters {
    batches: AtomicU64,
    requests: AtomicU64,
    bytes_fetched: AtomicU64,
    cache_hits: AtomicU64,
    cache_misses: AtomicU64,
}

impl Counters {
    fn snapshot(&self) -> IoStats {
        IoStats {
            batches: self.batches.load(Ordering::Relaxed),
            requests: self.requests.load(Ordering::Relaxed),
            bytes_fetched: self.bytes_fetched.load(Ordering::Relaxed),
            cache_hits: self.cache_hits.load(Ordering::Relaxed),
            cache_misses: self.cache_misses.load(Ordering::Relaxed),
        }
    }

    /// Record one wait on the store, carrying `requests` parallel requests.
    fn record_fetch(&self, requests: u64, bytes: u64) {
        self.batches.fetch_add(1, Ordering::Relaxed);
        self.requests.fetch_add(requests, Ordering::Relaxed);
        self.bytes_fetched.fetch_add(bytes, Ordering::Relaxed);
    }
}

/// A cache of aligned metadata blocks, keyed by block index.
#[derive(Debug, Default)]
struct BlockCache {
    blocks: HashMap<u64, Bytes>,
    /// Monotonic tick per access, used to pick a victim on eviction.
    used: HashMap<u64, u64>,
    clock: u64,
    bytes: u64,
}

impl BlockCache {
    fn get(&mut self, index: u64) -> Option<Bytes> {
        let block = self.blocks.get(&index)?.clone();
        self.clock += 1;
        self.used.insert(index, self.clock);
        Some(block)
    }

    fn insert(&mut self, index: u64, block: Bytes, capacity: u64) {
        if let Some(old) = self.blocks.insert(index, block.clone()) {
            self.bytes -= old.len() as u64;
        }
        self.bytes += block.len() as u64;
        self.clock += 1;
        self.used.insert(index, self.clock);

        while self.bytes > capacity && self.blocks.len() > 1 {
            let Some(victim) = self
                .used
                .iter()
                .min_by_key(|entry| *entry.1)
                .map(|entry| *entry.0)
            else {
                break;
            };
            if let Some(old) = self.blocks.remove(&victim) {
                self.bytes -= old.len() as u64;
            }
            self.used.remove(&victim);
        }
    }
}

/// A single HDF5 file addressed within an [`ObjectStore`].
///
/// Cloning is cheap, and clones share the same block cache and statistics: the
/// store, cache and counters all sit behind an [`Arc`].
///
/// ```no_run
/// use h5rs::object_store::ObjectStoreFile;
/// use object_store::{local::LocalFileSystem, path::Path};
///
/// let store = LocalFileSystem::new_with_prefix("/data").unwrap();
/// let file = ObjectStoreFile::new(Box::new(store), Path::from("example.h5"));
/// ```
///
/// [`Arc`]: std::sync::Arc
#[derive(Clone)]
pub struct ObjectStoreFile {
    path: Path,
    os: std::sync::Arc<Box<dyn ObjectStore>>,
    options: ReadOptions,
    cache: std::sync::Arc<Mutex<BlockCache>>,
    counters: std::sync::Arc<Counters>,
    compute: std::sync::Arc<dyn crate::compute::ComputePool>,
    /// The object's size, learned from the first response. Zero until then.
    /// Read-ahead rounds requests up to a block boundary, which can run past
    /// the end of the file; knowing the size lets those be trimmed instead of
    /// rejected by the store.
    known_size: std::sync::Arc<AtomicU64>,
}

impl ObjectStoreFile {
    /// Address the object at `path` within `os`, with the default
    /// [`ReadOptions`].
    pub fn new(os: Box<dyn ObjectStore>, path: Path) -> ObjectStoreFile {
        Self::with_options(os, path, ReadOptions::default())
    }

    /// Address the object at `path` within `os`, tuning how much h5rs reads
    /// ahead and how large a batch it will hold in memory.
    pub fn with_options(
        os: Box<dyn ObjectStore>,
        path: Path,
        options: ReadOptions,
    ) -> ObjectStoreFile {
        ObjectStoreFile {
            path,
            os: std::sync::Arc::new(os),
            options,
            cache: Default::default(),
            counters: Default::default(),
            compute: crate::compute::default_pool(),
            known_size: Default::default(),
        }
    }

    /// Use `pool` for the CPU-bound part of reads — decompressing and
    /// un-shuffling chunks.
    ///
    /// Without this, decoding happens inline on the async task, which still
    /// interleaves with I/O but uses one core. See [`crate::compute`].
    pub fn with_compute(mut self, pool: std::sync::Arc<dyn crate::compute::ComputePool>) -> Self {
        self.compute = pool;
        self
    }

    /// The compute pool in force.
    pub(crate) fn compute(&self) -> &std::sync::Arc<dyn crate::compute::ComputePool> {
        &self.compute
    }

    /// The read-ahead and batching settings in force.
    pub fn options(&self) -> ReadOptions {
        self.options
    }

    /// What reading this file has cost so far: requests issued, bytes
    /// transferred, and how often read-ahead paid off.
    pub fn stats(&self) -> IoStats {
        self.counters.snapshot()
    }

    /// The underlying object store.
    pub fn store(&self) -> &dyn ObjectStore {
        self.os.as_ref()
    }

    /// The object's path within the store.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The last path segment, or `"default"` if the path has none.
    pub fn filename(&self) -> String {
        self.path.filename().unwrap_or("default").to_string()
    }

    /// Fetch the object's metadata (size, etag, last-modified).
    pub async fn metadata(&self) -> Result<ObjectMeta, object_store::Error> {
        // We make a GET request with a minimal range
        // because S3 pre-signed URLs can't be used with HEAD.
        let opts = GetOptions {
            range: Some(GetRange::Bounded(0..1)),
            ..Default::default()
        };
        let r = self.os.get_opts(&self.path, opts).await?;
        self.known_size.store(r.meta.size, Ordering::Relaxed);
        self.counters.record_fetch(1, 1);
        Ok(r.meta)
    }

    /// The object's size in bytes.
    pub async fn size(&self) -> Result<u64, object_store::Error> {
        Ok(self.metadata().await?.size)
    }

    /// Fetch a byte range as a stream, without buffering the whole range.
    pub async fn get_range_stream(
        &self,
        range: Range<u64>,
    ) -> Result<BoxStream<'static, Result<Bytes, object_store::Error>>, object_store::Error> {
        let opts = GetOptions {
            range: Some(GetRange::Bounded(range.clone())),
            ..Default::default()
        };
        let r = self.os.get_opts(&self.path, opts).await?;
        self.counters
            .record_fetch(1, range.end.saturating_sub(range.start));
        Ok(r.into_stream())
    }

    /// Fetch a byte range into memory, bypassing the metadata cache.
    ///
    /// A range starting past the end of the file yields no bytes rather than an
    /// error, so that speculative read-ahead near the end of a file is safe.
    pub async fn get_range(&self, range: Range<u64>) -> Result<Bytes, object_store::Error> {
        let Some(range) = self.clamp(range) else {
            return Ok(Bytes::new());
        };
        let opts = GetOptions {
            range: Some(GetRange::Bounded(range)),
            ..Default::default()
        };
        let r = self.os.get_opts(&self.path, opts).await?;
        self.known_size.store(r.meta.size, Ordering::Relaxed);
        let bytes = r.bytes().await?;
        self.counters.record_fetch(1, bytes.len() as u64);
        Ok(bytes)
    }

    /// Trim a range to the part of the file that exists, or `None` if it starts
    /// beyond the end. Before the size is known, ranges pass through unchanged.
    fn clamp(&self, range: Range<u64>) -> Option<Range<u64>> {
        let size = self.known_size.load(Ordering::Relaxed);
        if size == 0 {
            return Some(range);
        }
        if range.start >= size {
            return None;
        }
        Some(range.start..range.end.min(size))
    }

    /// Fetch several byte ranges at once.
    ///
    /// Ranges lying close together are merged into one request, and the store
    /// issues what remains in parallel. Results come back in the order the
    /// ranges were given.
    pub async fn read_ranges(&self, ranges: &[Range<u64>]) -> H5Result<Vec<Bytes>> {
        if ranges.is_empty() {
            return Ok(vec![]);
        }

        // Merge here rather than leaving it to the store, so that the request
        // count `stats` reports is the count actually issued.
        let merged = merge_ranges(
            ranges,
            self.options.coalesce_gap,
            self.options.max_coalesced_bytes,
        );
        let clamped: Vec<Option<Range<u64>>> =
            merged.iter().map(|r| self.clamp(r.clone())).collect();
        let to_fetch: Vec<Range<u64>> = clamped.iter().flatten().cloned().collect();

        let fetched = if to_fetch.is_empty() {
            vec![]
        } else {
            let fetched_bytes: u64 = to_fetch.iter().map(|r| r.end - r.start).sum();
            let fetched = self.os.get_ranges(&self.path, &to_fetch).await?;
            self.counters
                .record_fetch(to_fetch.len() as u64, fetched_bytes);
            fetched
        };

        // Ranges that lay entirely past the end of the file contribute nothing.
        let mut supplied = fetched.into_iter();
        let blocks: Vec<Bytes> = clamped
            .iter()
            .map(|c| match c {
                Some(_) => supplied.next().unwrap_or_default(),
                None => Bytes::new(),
            })
            .collect();

        Ok(ranges
            .iter()
            .map(|range| {
                // `merged` is sorted and covers every requested range.
                let idx = merged.partition_point(|m| m.start <= range.start) - 1;
                let block = &blocks[idx];
                let start = ((range.start - merged[idx].start) as usize).min(block.len());
                let end = ((range.end - merged[idx].start) as usize).min(block.len());
                block.slice(start..end.max(start))
            })
            .collect())
    }

    /// Read `len` bytes of metadata at `offset` through the block cache.
    ///
    /// Returns fewer bytes only at end of file; the caller's parser decides
    /// whether that is enough.
    pub(crate) async fn read_metadata_bytes(&self, offset: u64, len: u64) -> H5Result<Bytes> {
        if len == 0 {
            return Ok(Bytes::new());
        }
        let block_size = self.options.metadata_block_size;
        let first = offset / block_size;
        let last = (offset + len - 1) / block_size;

        let mut blocks: Vec<Option<Bytes>> = Vec::with_capacity((last - first + 1) as usize);
        {
            let mut cache = self.cache.lock().expect("read cache poisoned");
            for index in first..=last {
                blocks.push(cache.get(index));
            }
        }

        if blocks.iter().all(Option::is_some) {
            self.counters.cache_hits.fetch_add(1, Ordering::Relaxed);
        } else {
            self.counters.cache_misses.fetch_add(1, Ordering::Relaxed);
            self.fill_blocks(first, last, &mut blocks).await?;
        }

        Ok(assemble(
            |index| blocks[(index - first) as usize].clone().unwrap_or_default(),
            block_size,
            offset,
            len,
        ))
    }

    /// Read many metadata spans at once.
    ///
    /// This is what turns a B-tree level, or a run of sibling nodes, into a
    /// single round trip.
    ///
    /// Unlike the single-span path this does *not* read ahead. Read-ahead pays
    /// off when the next address is only discovered by reading the current
    /// structure; here every address is already known, so rounding each one up
    /// to a whole block would cost bytes and save nothing. Spans that are close
    /// together are still merged into one request.
    pub(crate) async fn read_metadata_many(&self, spans: &[(u64, u64)]) -> H5Result<Vec<Bytes>> {
        if spans.is_empty() {
            return Ok(vec![]);
        }
        if spans.len() == 1 {
            return Ok(vec![
                self.read_metadata_bytes(spans[0].0, spans[0].1).await?,
            ]);
        }

        // Anything the block cache already holds costs nothing.
        let mut out: Vec<Bytes> = Vec::with_capacity(spans.len());
        let mut pending: Vec<usize> = vec![];
        {
            let block_size = self.options.metadata_block_size;
            let mut cache = self.cache.lock().expect("read cache poisoned");
            for (i, &(offset, len)) in spans.iter().enumerate() {
                match cached_span(&mut cache, block_size, offset, len) {
                    Some(bytes) => out.push(bytes),
                    None => {
                        out.push(Bytes::new());
                        pending.push(i);
                    }
                }
            }
        }

        self.counters
            .cache_hits
            .fetch_add((spans.len() - pending.len()) as u64, Ordering::Relaxed);
        if pending.is_empty() {
            return Ok(out);
        }
        self.counters
            .cache_misses
            .fetch_add(pending.len() as u64, Ordering::Relaxed);

        let ranges: Vec<Range<u64>> = pending
            .iter()
            .map(|&i| spans[i].0..spans[i].0 + spans[i].1)
            .collect();
        for (&i, bytes) in pending.iter().zip(self.read_ranges(&ranges).await?) {
            out[i] = bytes;
        }
        Ok(out)
    }

    /// Fetch every block in `first..=last` that is not already present, in one
    /// request covering the missing span.
    async fn fill_blocks(
        &self,
        first: u64,
        last: u64,
        blocks: &mut [Option<Bytes>],
    ) -> H5Result<()> {
        let block_size = self.options.metadata_block_size;
        let missing: Vec<u64> = (first..=last)
            .filter(|i| blocks[(i - first) as usize].is_none())
            .collect();
        let (Some(&lo), Some(&hi)) = (missing.first(), missing.last()) else {
            return Ok(());
        };

        let fetched = self
            .get_range(lo * block_size..(hi + 1) * block_size)
            .await?;

        let mut cache = self.cache.lock().expect("read cache poisoned");
        for index in lo..=hi {
            let from = ((index - lo) * block_size) as usize;
            // Past end of file: cache an empty block so that repeating the read
            // does not issue another request.
            let block = if from >= fetched.len() {
                Bytes::new()
            } else {
                fetched.slice(from..(from + block_size as usize).min(fetched.len()))
            };
            cache.insert(index, block.clone(), self.options.cache_capacity);
            blocks[(index - first) as usize] = Some(block);
        }
        Ok(())
    }
}

/// Serve a span from the block cache, if every block it needs is present.
fn cached_span(cache: &mut BlockCache, block_size: u64, offset: u64, len: u64) -> Option<Bytes> {
    if len == 0 {
        return Some(Bytes::new());
    }
    let first = offset / block_size;
    let last = (offset + len - 1) / block_size;
    let blocks: Option<Vec<Bytes>> = (first..=last).map(|i| cache.get(i)).collect();
    let blocks = blocks?;
    Some(assemble(
        |index| blocks[(index - first) as usize].clone(),
        block_size,
        offset,
        len,
    ))
}

/// Cut `len` bytes at `offset` out of the blocks `block_of` supplies.
///
/// A span that lies inside a single block is returned as a zero-copy slice of
/// it; one straddling a boundary is stitched together.
fn assemble(block_of: impl Fn(u64) -> Bytes, block_size: u64, offset: u64, len: u64) -> Bytes {
    let first = offset / block_size;
    let last = (offset + len - 1) / block_size;

    if first == last {
        let block = block_of(first);
        let start = ((offset - first * block_size) as usize).min(block.len());
        let end = (start + len as usize).min(block.len());
        return block.slice(start..end);
    }

    let mut out = Vec::with_capacity(len as usize);
    for index in first..=last {
        let block = block_of(index);
        let block_start = index * block_size;
        let from = (offset.saturating_sub(block_start) as usize).min(block.len());
        let to = (((offset + len).saturating_sub(block_start)) as usize).min(block.len());
        if from < to {
            out.extend_from_slice(&block[from..to]);
        }
    }
    Bytes::from(out)
}

/// Merge ranges that lie close enough together to be worth a single request.
///
/// The result is sorted by start offset and covers every input range. Merging
/// stops at `max_span` so that one request can never balloon into holding an
/// unbounded amount of unwanted data.
fn merge_ranges(ranges: &[Range<u64>], gap: u64, max_span: u64) -> Vec<Range<u64>> {
    let mut sorted: Vec<Range<u64>> = ranges.to_vec();
    sorted.sort_unstable_by_key(|r| r.start);

    let mut merged: Vec<Range<u64>> = Vec::with_capacity(sorted.len());
    for range in sorted {
        match merged.last_mut() {
            Some(last)
                if range.start <= last.end.saturating_add(gap)
                    && range.end.saturating_sub(last.start) <= max_span =>
            {
                last.end = last.end.max(range.end);
            }
            _ => merged.push(range),
        }
    }
    merged
}

/// Fetch `len` bytes of metadata at `offset` and parse with binrw (no args).
pub(crate) async fn read_and_parse<T: for<'a> BinRead<Args<'a> = ()>>(
    file: &ObjectStoreFile,
    offset: u64,
    len: u64,
) -> H5Result<T> {
    let bytes = file.read_metadata_bytes(offset, len).await?;
    let mut cursor = Cursor::new(bytes);
    Ok(T::read_le(&mut cursor)?)
}

/// Fetch `len` bytes of metadata at `offset` and parse with binrw args.
pub(crate) async fn read_and_parse_args<T: for<'a> BinRead<Args<'a> = A>, A: Clone>(
    file: &ObjectStoreFile,
    offset: u64,
    len: u64,
    args: A,
) -> H5Result<T> {
    let bytes = file.read_metadata_bytes(offset, len).await?;
    let mut cursor = Cursor::new(bytes);
    Ok(T::read_le_args(&mut cursor, args)?)
}

/// Fetch a metadata structure of unknown size at `offset`.
pub(crate) async fn read_metadata<T: for<'a> BinRead<Args<'a> = ()>>(
    file: &ObjectStoreFile,
    offset: u64,
) -> H5Result<T> {
    read_and_parse(file, offset, METADATA_FETCH_SIZE).await
}

/// Fetch a metadata structure of unknown size at `offset` with args.
pub(crate) async fn read_metadata_args<T: for<'a> BinRead<Args<'a> = A>, A: Clone>(
    file: &ObjectStoreFile,
    offset: u64,
    args: A,
) -> H5Result<T> {
    read_and_parse_args(file, offset, METADATA_FETCH_SIZE, args).await
}

/// Fetch `len` bytes of metadata at `offset`, through the block cache.
pub(crate) async fn fetch_metadata(
    file: &ObjectStoreFile,
    offset: u64,
    len: u64,
) -> H5Result<Bytes> {
    file.read_metadata_bytes(offset, len).await
}

/// Fetch `len` bytes of bulk data at `offset`, bypassing the metadata cache.
pub(crate) async fn fetch_data(file: &ObjectStoreFile, offset: u64, len: u64) -> H5Result<Bytes> {
    Ok(file.get_range(offset..offset + len).await?)
}
