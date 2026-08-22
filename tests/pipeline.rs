//! How well reading overlaps I/O and decompression.
//!
//! A chunked read is two kinds of work at once: waiting on the store and
//! decompressing what comes back. Done well, whichever is scarcer sets the
//! pace and the other disappears behind it. These tests measure whether that is
//! actually happening, against a store whose latency and bandwidth can be
//! dialled to match the situation being modelled.
//!
//! The measurements are printed by `#[ignore]`d tests; run them with:
//!
//! ```bash
//! cargo test --test pipeline -- --ignored --nocapture
//! ```
//!
//! The one test that is not ignored asserts the property they measure: that
//! decompression happens *during* the download rather than after it.

// Writing the fixtures needs the HDF5 C library.
#![cfg(all(not(target_arch = "wasm32"), feature = "hdf5-compare"))]

use std::fmt::{Debug, Display, Formatter};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use futures_core::future::BoxFuture;
use futures_core::stream::BoxStream;
use h5rs::compute::{ComputeJob, ComputePool, InlineCompute, ThreadPoolCompute};
use h5rs::error::H5Result;
use h5rs::object_store::{ObjectStoreFile, ReadOptions};
use object_store::local::LocalFileSystem;
use object_store::path::Path;
use object_store::{
    CopyOptions, GetOptions, GetResult, ListResult, MultipartUpload, ObjectMeta, ObjectStore,
    PutMultipartOptions, PutOptions, PutPayload, PutResult, Result as OsResult,
};

/// A generated dataset: `values` u32 elements in chunks of `chunk`, gzip and
/// shuffle applied, written once and reused across scenarios.
///
/// Generating rather than using a sample file keeps these measurements
/// self-contained, and lets each scenario pick the chunk size it is about.
struct Fixture {
    _dir: tempfile::TempDir,
    path: std::path::PathBuf,
    values: usize,
}

impl Fixture {
    fn build(values: usize, chunk: usize) -> Fixture {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("fixture.h5");
        {
            let hf = hdf5::File::create(&path).unwrap();
            let ds = hf
                .new_dataset::<u32>()
                .shape(&[values][..])
                .chunk(&[chunk][..])
                .shuffle()
                .deflate(1)
                .create("data")
                .unwrap();
            ds.write_raw(&sample_data(values)).unwrap();
        }
        Fixture {
            _dir: dir,
            path,
            values,
        }
    }

    fn stored_bytes(&self) -> u64 {
        std::fs::metadata(&self.path).unwrap().len()
    }
}

/// Values with about sixteen bits of entropy each.
///
/// After shuffling, two of every four byte planes are near-constant and two are
/// noise, which deflate compresses roughly two to one — close to what real
/// instrument data does, and far from the 100:1 that a tidy arithmetic sequence
/// would give. Compression ratio is what decides whether a read is limited by
/// the link or by the CPU, so it has to be realistic for these numbers to mean
/// anything.
fn sample_data(values: usize) -> Vec<u32> {
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    (0..values)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state as u32) & 0xFFFF
        })
        .collect()
}

// ---------------------------------------------------------------- telemetry

/// Intervals during which each resource was busy, relative to the read's start.
#[derive(Default)]
struct Trace {
    /// One entry per request, from issue to last byte.
    requests: Vec<(Duration, Duration)>,
    /// One entry per request, covering only the time it held the link. These do
    /// not overlap, which is what makes the link a shared, finite resource.
    transfers: Vec<(Duration, Duration)>,
    /// One entry per decode job, covering only the time it was running.
    decodes: Vec<(Duration, Duration)>,
    bytes: u64,
}

/// What a run cost, and how well the two resources overlapped.
struct Report {
    label: String,
    wall: Duration,
    /// Time the link spent actually transferring.
    link_busy: Duration,
    /// Total CPU time spent decompressing, across all workers. This is not all
    /// the CPU a read uses — copying decoded chunks into the output array
    /// happens on the driving task and is not counted here — but it is the part
    /// a compute pool can spread over cores.
    cpu_busy: Duration,
    /// Wall time during which decoding and downloading were both happening.
    overlap: Duration,
    parallelism: usize,
    /// Wall time during which at least one decode was running.
    decode_span: Duration,
    bytes: u64,
    requests: usize,
    decodes: usize,
    /// The most requests that were ever outstanding at the same moment. This is
    /// what an SSD sees as queue depth, and what decides whether it can keep its
    /// channels busy.
    peak_concurrency: usize,
    /// Average number outstanding over the time any were.
    mean_concurrency: f64,
}

impl Report {
    /// How much of the run the link was transferring for. At 1.0 the link is
    /// the limit and is fully used.
    fn link_utilisation(&self) -> f64 {
        ratio(self.link_busy, self.wall)
    }

    /// How much of the available compute the decoder used. At 1.0 the CPU is
    /// the limit and is fully used.
    fn cpu_utilisation(&self) -> f64 {
        ratio(self.cpu_busy, self.wall * self.parallelism as u32)
    }

    /// The best wall time this run could have had: the slower resource alone.
    ///
    /// Reaching it exactly is not possible — the first request has nothing to
    /// overlap with, nor the last decode — and on a single core the copy into
    /// the output array shares the thread with decompression.
    fn ideal(&self) -> Duration {
        self.link_busy.max(self.cpu_busy / self.parallelism as u32)
    }

    /// How close the run came to that ideal. At 1.0 the faster resource is
    /// entirely hidden behind the slower one.
    fn efficiency(&self) -> f64 {
        ratio(self.ideal(), self.wall)
    }

    /// The share of decoding that happened while a request was in flight.
    fn overlapped(&self) -> f64 {
        ratio(self.overlap, self.decode_span)
    }
}

impl Display for Report {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:<40} wall {:>7.0}ms  link {:>7.0}ms ({:>3.0}%)  decode {:>7.0}ms/{} ({:>3.0}%)  \
             ideal {:>7.0}ms  eff {:>3.0}%  overlap {:>3.0}%  concurrency {:>2}/{:.1}  \
             [{} reqs, {} chunks, {:.0} MiB]",
            self.label,
            self.wall.as_secs_f64() * 1e3,
            self.link_busy.as_secs_f64() * 1e3,
            self.link_utilisation() * 100.0,
            self.cpu_busy.as_secs_f64() * 1e3,
            self.parallelism,
            self.cpu_utilisation() * 100.0,
            self.ideal().as_secs_f64() * 1e3,
            self.efficiency() * 100.0,
            self.overlapped() * 100.0,
            self.peak_concurrency,
            self.mean_concurrency,
            self.requests,
            self.decodes,
            self.bytes as f64 / (1024.0 * 1024.0),
        )
    }
}

fn ratio(a: Duration, b: Duration) -> f64 {
    if b.is_zero() {
        0.0
    } else {
        a.as_secs_f64() / b.as_secs_f64()
    }
}

/// Total length of the union of a set of intervals.
fn union(mut intervals: Vec<(Duration, Duration)>) -> Duration {
    intervals.sort_unstable();
    let mut total = Duration::ZERO;
    let mut cursor: Option<(Duration, Duration)> = None;
    for (start, end) in intervals {
        match &mut cursor {
            Some((_, open_end)) if start <= *open_end => *open_end = (*open_end).max(end),
            Some((open_start, open_end)) => {
                total += *open_end - *open_start;
                cursor = Some((start, end));
            }
            None => cursor = Some((start, end)),
        }
    }
    if let Some((start, end)) = cursor {
        total += end - start;
    }
    total
}

fn total(intervals: &[(Duration, Duration)]) -> Duration {
    intervals.iter().map(|(s, e)| *e - *s).sum()
}

/// How deeply a set of intervals overlaps: the most that were ever open at
/// once, and the average over the time any were open.
fn concurrency(intervals: &[(Duration, Duration)]) -> (usize, f64) {
    let mut events: Vec<(Duration, i32)> = Vec::with_capacity(intervals.len() * 2);
    for (start, end) in intervals {
        events.push((*start, 1));
        events.push((*end, -1));
    }
    // Close before open at the same instant, so touching intervals do not read
    // as overlapping.
    events.sort_unstable_by_key(|(at, delta)| (*at, -*delta));

    let (mut open, mut peak) = (0i32, 0i32);
    let mut weighted = Duration::ZERO.as_secs_f64();
    let mut span = 0.0f64;
    let mut previous: Option<Duration> = None;
    for (at, delta) in events {
        if let Some(prev) = previous {
            if open > 0 {
                let dt = (at - prev).as_secs_f64();
                weighted += dt * open as f64;
                span += dt;
            }
        }
        open += delta;
        peak = peak.max(open);
        previous = Some(at);
    }
    (
        peak as usize,
        if span > 0.0 { weighted / span } else { 0.0 },
    )
}

/// Total length of the overlap between two sets of intervals.
fn intersection(a: &[(Duration, Duration)], b: &[(Duration, Duration)]) -> Duration {
    let (a, b) = (merged(a), merged(b));
    let mut total = Duration::ZERO;
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        let start = a[i].0.max(b[j].0);
        let end = a[i].1.min(b[j].1);
        if start < end {
            total += end - start;
        }
        if a[i].1 < b[j].1 {
            i += 1;
        } else {
            j += 1;
        }
    }
    total
}

/// Collapse overlapping intervals into a sorted, disjoint set.
fn merged(intervals: &[(Duration, Duration)]) -> Vec<(Duration, Duration)> {
    let mut sorted = intervals.to_vec();
    sorted.sort_unstable();
    let mut out: Vec<(Duration, Duration)> = vec![];
    for (start, end) in sorted {
        match out.last_mut() {
            Some(last) if start <= last.1 => last.1 = last.1.max(end),
            _ => out.push((start, end)),
        }
    }
    out
}

// ------------------------------------------------------------- mock storage

/// A store that serves a real file over a simulated link.
///
/// Latency is paid per request and overlaps freely, which is what makes
/// concurrency worth having. Bandwidth is a single shared pipe: each request
/// *reserves* a slot on it, so issuing more requests cannot conjure more
/// throughput.
///
/// The reservation is computed up front rather than by holding a lock across an
/// await, so a caller that blocks its thread — a browser decoding on the main
/// thread, say — does not slow the simulated network down. The bytes keep
/// arriving; the caller just observes them late, which is exactly what happens
/// in a browser.
struct LinkStore {
    inner: LocalFileSystem,
    latency: Duration,
    bytes_per_sec: Option<f64>,
    /// When the link next becomes free, relative to `origin`.
    link_free_at: Mutex<Duration>,
    origin: Instant,
    trace: Arc<Mutex<Trace>>,
}

impl LinkStore {
    fn new(
        dir: &std::path::Path,
        latency: Duration,
        bytes_per_sec: Option<f64>,
        trace: Arc<Mutex<Trace>>,
        origin: Instant,
    ) -> LinkStore {
        LinkStore {
            inner: LocalFileSystem::new_with_prefix(dir).unwrap(),
            latency,
            bytes_per_sec,
            link_free_at: Mutex::new(Duration::ZERO),
            origin,
            trace,
        }
    }
}

impl Debug for LinkStore {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str("LinkStore")
    }
}

impl Display for LinkStore {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str("LinkStore")
    }
}

#[async_trait::async_trait]
impl ObjectStore for LinkStore {
    async fn get_opts(&self, location: &Path, options: GetOptions) -> OsResult<GetResult> {
        let started = self.origin.elapsed();
        if !self.latency.is_zero() {
            tokio::time::sleep(self.latency).await;
        }

        let result = self.inner.get_opts(location, options).await?;
        let meta = result.meta.clone();
        let range = result.range.clone();
        let bytes = result.bytes().await?;

        if let Some(rate) = self.bytes_per_sec {
            // Reserve the next free slot on the shared pipe. Reservations never
            // overlap, so the sum of their lengths is the time the link spent
            // busy no matter how the caller schedules itself.
            let (start, end) = {
                let mut free_at = self.link_free_at.lock().unwrap();
                let start = self.origin.elapsed().max(*free_at);
                let end = start + Duration::from_secs_f64(bytes.len() as f64 / rate);
                *free_at = end;
                (start, end)
            };
            tokio::time::sleep_until((self.origin + end).into()).await;
            self.trace.lock().unwrap().transfers.push((start, end));
        }

        let mut trace = self.trace.lock().unwrap();
        trace.requests.push((started, self.origin.elapsed()));
        trace.bytes += bytes.len() as u64;
        drop(trace);

        Ok(GetResult {
            payload: object_store::GetResultPayload::Stream(Box::pin(futures_util::stream::once(
                async move { Ok(bytes) },
            ))),
            meta,
            range,
            attributes: Default::default(),
            extensions: Default::default(),
        })
    }

    async fn put_opts(&self, _l: &Path, _p: PutPayload, _o: PutOptions) -> OsResult<PutResult> {
        unimplemented!("read-only store")
    }
    async fn put_multipart_opts(
        &self,
        _l: &Path,
        _o: PutMultipartOptions,
    ) -> OsResult<Box<dyn MultipartUpload>> {
        unimplemented!("read-only store")
    }
    fn delete_stream(
        &self,
        _l: BoxStream<'static, OsResult<Path>>,
    ) -> BoxStream<'static, OsResult<Path>> {
        unimplemented!("read-only store")
    }
    fn list(&self, _p: Option<&Path>) -> BoxStream<'static, OsResult<ObjectMeta>> {
        unimplemented!("read-only store")
    }
    async fn list_with_delimiter(&self, _p: Option<&Path>) -> OsResult<ListResult> {
        unimplemented!("read-only store")
    }
    async fn copy_opts(&self, _f: &Path, _t: &Path, _o: CopyOptions) -> OsResult<()> {
        unimplemented!("read-only store")
    }
}

// -------------------------------------------------------------- mock compute

/// Wraps a pool and records when each job actually ran.
struct TracingCompute {
    inner: Arc<dyn ComputePool>,
    origin: Instant,
    trace: Arc<Mutex<Trace>>,
}

impl Debug for TracingCompute {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "TracingCompute({:?})", self.inner)
    }
}

impl ComputePool for TracingCompute {
    fn run(&self, job: ComputeJob) -> BoxFuture<'static, H5Result<Vec<u8>>> {
        let origin = self.origin;
        let trace = self.trace.clone();
        // Time the closure itself, so queueing time is not counted as CPU busy.
        let timed: ComputeJob = Box::new(move || {
            let started = origin.elapsed();
            let result = job();
            trace
                .lock()
                .unwrap()
                .decodes
                .push((started, origin.elapsed()));
            result
        });
        self.inner.run(timed)
    }

    fn parallelism(&self) -> usize {
        self.inner.parallelism()
    }
}

// ------------------------------------------------------------------ harness

struct Scenario<'a> {
    label: String,
    fixture: &'a Fixture,
    latency: Duration,
    bytes_per_sec: Option<f64>,
    compute: Arc<dyn ComputePool>,
    options: ReadOptions,
}

async fn measure(scenario: Scenario<'_>) -> Report {
    let path = scenario.fixture.path.canonicalize().unwrap();
    let trace = Arc::new(Mutex::new(Trace::default()));
    let origin = Instant::now();

    let store = LinkStore::new(
        path.parent().unwrap(),
        scenario.latency,
        scenario.bytes_per_sec,
        trace.clone(),
        origin,
    );
    let parallelism = scenario.compute.parallelism();
    let pool = Arc::new(TracingCompute {
        inner: scenario.compute,
        origin,
        trace: trace.clone(),
    });

    let file = ObjectStoreFile::with_options(
        Box::new(store),
        Path::from(path.file_name().unwrap().to_str().unwrap()),
        scenario.options,
    )
    .with_compute(pool);

    // Open the dataset before timing, so the measurement covers the bulk read
    // rather than the metadata walk that precedes it.
    let ds = h5rs::open_dataset(&file, &["data"]).await.unwrap().unwrap();
    trace.lock().unwrap().requests.clear();
    trace.lock().unwrap().transfers.clear();
    trace.lock().unwrap().bytes = 0;

    let start = Instant::now();
    let values = ds.read_full::<u32>(&file).await.unwrap();
    let wall = start.elapsed();
    assert_eq!(values.data.len(), scenario.fixture.values);
    // Spot-check the decoded data, so a pipeline that drops or mixes up chunks
    // cannot quietly post good numbers.
    let expected = sample_data(values.data.len());
    for i in [0, 1, values.data.len() / 2, values.data.len() - 1] {
        assert_eq!(values.data[i], expected[i], "value {i}");
    }

    let trace = trace.lock().unwrap();
    Report {
        label: scenario.label,
        wall,
        link_busy: total(&trace.transfers),
        cpu_busy: total(&trace.decodes),
        overlap: intersection(&trace.decodes, &trace.requests),
        parallelism,
        decode_span: union(trace.decodes.clone()),
        bytes: trace.bytes,
        requests: trace.requests.len(),
        decodes: trace.decodes.len(),
        peak_concurrency: concurrency(&trace.requests).0,
        mean_concurrency: concurrency(&trace.requests).1,
    }
}

// ---------------------------------------------------------------- scenarios

/// Native: 1 GB of compressed data in ~100 chunks off local flash, where
/// decompression is the scarce resource and the cores should saturate.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn native_flash_and_cores() {
    // 128 Mi u32 = 512 MiB uncompressed, in 100 chunks.
    let fixture = Fixture::build(134_217_728, 1_342_178);
    println!(
        "\nnative: 512 MiB in 100 gzip+shuffle chunks ({:.0} MiB stored), flash at 2 GB/s\n",
        fixture.stored_bytes() as f64 / (1024.0 * 1024.0)
    );

    for threads in [1usize, 2, 4, 8] {
        let compute: Arc<dyn ComputePool> = if threads == 1 {
            Arc::new(InlineCompute)
        } else {
            Arc::new(ThreadPoolCompute::new(threads))
        };
        let report = measure(Scenario {
            label: if threads == 1 {
                "inline (1 core)".to_string()
            } else {
                format!("thread pool ({threads} cores)")
            },
            fixture: &fixture,
            latency: Duration::from_micros(100),
            // Model flash rather than the page cache, so the numbers do not
            // depend on whether the file happens to be resident.
            bytes_per_sec: Some(2.0e9),
            compute,
            options: ReadOptions::default(),
        })
        .await;
        println!("{report}");
    }
}

/// Web: 100 MB in 50 chunks at 30 MB/s with 50 ms of latency, decoding on the
/// one thread a browser gives you. The link is the scarce resource and
/// decompression should vanish behind it.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn web_object_storage() {
    // 50 Mi u32 = 200 MiB uncompressed, in 50 chunks, storing about 100 MiB.
    let fixture = Fixture::build(52_428_800, 1_048_576);
    println!(
        "\nweb: 200 MiB in 50 gzip+shuffle chunks ({:.0} MiB stored) over 30 MB/s, 50 ms latency\n",
        fixture.stored_bytes() as f64 / (1024.0 * 1024.0)
    );

    for concurrency in [1usize, 2, 4, 8, 16] {
        let report = measure(Scenario {
            label: format!("io_concurrency = {concurrency}"),
            fixture: &fixture,
            latency: Duration::from_millis(50),
            bytes_per_sec: Some(30.0e6),
            compute: Arc::new(InlineCompute),
            options: ReadOptions {
                io_concurrency: concurrency,
                ..ReadOptions::default()
            },
        })
        .await;
        println!("{report}");
    }
}

/// Native, on a real filesystem with stock Tokio: how many reads are actually
/// outstanding at once.
///
/// A modern SSD needs several requests in flight to reach its rated
/// throughput — one at a time leaves most of its parallelism unused. Under
/// Tokio, `LocalFileSystem` dispatches each read to the blocking pool, so the
/// depth h5rs reaches is decided by how many requests its pipeline keeps open,
/// not by the filesystem.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn native_request_concurrency() {
    let fixture = Fixture::build(134_217_728, 1_342_178);
    println!(
        "\nnative: 512 MiB in 100 chunks ({:.0} MiB stored), real filesystem, stock Tokio\n",
        fixture.stored_bytes() as f64 / (1024.0 * 1024.0)
    );

    // Chunks here store about 2.6 MiB each, so with the default 32 MiB ceiling
    // it is the memory bound — not `io_concurrency` — that decides how deep the
    // queue gets.
    for max_inflight_bytes in [2u64 << 20, 8 << 20, 32 << 20] {
        for (label, compute) in [
            (
                "inline (1 core)",
                Arc::new(InlineCompute) as Arc<dyn ComputePool>,
            ),
            ("thread pool (8 cores)", Arc::new(ThreadPoolCompute::new(8))),
        ] {
            let report = measure(Scenario {
                label: format!("{label}, {} MiB held", max_inflight_bytes >> 20),
                fixture: &fixture,
                latency: Duration::ZERO,
                // No simulated link: this is the real SSD, so `link` is zero
                // and only the concurrency columns mean anything.
                bytes_per_sec: None,
                compute,
                options: ReadOptions {
                    max_inflight_bytes,
                    ..ReadOptions::default()
                },
            })
            .await;
            println!("{report}");
        }
    }
}

/// How request size trades round trips against pipeline granularity and peak
/// memory. Nothing in a request can be decoded until all of it has arrived, so
/// an oversized request stalls the decoder at the start of a read — and
/// `io_concurrency` of them in flight is what the read costs in memory.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn request_size() {
    let web = Fixture::build(52_428_800, 1_048_576);
    let native = Fixture::build(134_217_728, 1_342_178);

    for (name, fixture, latency, rate, compute) in [
        (
            "web  30 MB/s, 50ms",
            &web,
            Duration::from_millis(50),
            30.0e6,
            Arc::new(InlineCompute) as Arc<dyn ComputePool>,
        ),
        (
            "native 2 GB/s, 8 cores",
            &native,
            Duration::from_micros(100),
            2.0e9,
            Arc::new(ThreadPoolCompute::new(8)) as Arc<dyn ComputePool>,
        ),
    ] {
        println!("\n{name}\n");
        for max_coalesced_bytes in [
            512 * 1024,
            2 * 1024 * 1024,
            8 * 1024 * 1024,
            32 * 1024 * 1024,
        ] {
            let options = ReadOptions {
                max_coalesced_bytes,
                ..ReadOptions::default()
            };
            let report = measure(Scenario {
                label: format!(
                    "request <= {:>4} KiB (<= {:>4} MiB in flight)",
                    max_coalesced_bytes / 1024,
                    max_coalesced_bytes as usize * options.io_concurrency / (1024 * 1024)
                ),
                fixture,
                latency,
                bytes_per_sec: Some(rate),
                compute: compute.clone(),
                options,
            })
            .await;
            println!("{report}");
        }
    }
}

/// `max_coalesced_bytes` decides when neighbouring chunks are worth combining —
/// it never splits one.
///
/// A chunk always arrives in a single request at its full stored size, however
/// large it is. Turning this cap into something that splits chunks would be a
/// serious regression: a partially-arrived chunk cannot be decoded, so the
/// pieces would have to be reassembled before any progress could be made.
#[tokio::test(flavor = "multi_thread")]
async fn a_chunk_is_always_one_request() {
    // 8 chunks of 4 MiB uncompressed, each storing far more than the 512 KiB
    // default cap.
    let fixture = Fixture::build(8 * 1024 * 1024, 1024 * 1024);
    let path = fixture.path.canonicalize().unwrap();
    let stored = fixture.stored_bytes();

    for max_coalesced_bytes in [512 * 1024, 8 * 1024 * 1024] {
        let store = LocalFileSystem::new_with_prefix(path.parent().unwrap()).unwrap();
        let file = ObjectStoreFile::with_options(
            Box::new(store),
            Path::from(path.file_name().unwrap().to_str().unwrap()),
            ReadOptions {
                max_coalesced_bytes,
                ..ReadOptions::default()
            },
        );

        let ds = h5rs::open_dataset(&file, &["data"]).await.unwrap().unwrap();
        let before = file.stats();
        let values = ds.read_full::<u32>(&file).await.unwrap();
        assert_eq!(values.data.len(), fixture.values);

        let after = file.stats();
        let requests = after.requests - before.requests;
        let bytes = after.bytes_fetched - before.bytes_fetched;

        // Eight chunks, each bigger than the smaller cap: one request each when
        // the cap forbids merging, fewer once it allows it. Never more than one
        // request per chunk, and never more bytes than the file holds.
        assert!(
            requests <= 8,
            "cap {max_coalesced_bytes}: {requests} requests for 8 chunks — a chunk was split"
        );
        assert!(
            bytes <= stored,
            "cap {max_coalesced_bytes}: fetched {bytes} bytes for a {stored}-byte file"
        );
        println!(
            "cap {:>5} KiB: {requests} requests, {bytes} bytes",
            max_coalesced_bytes / 1024
        );
    }
}

/// A single-threaded host gets control back periodically during a long read.
///
/// Decompression has no await points inside it, so the only place a yield can
/// happen is between chunks. What matters is that it happens often enough that
/// no single stretch of held thread is long in terms of bytes decoded.
#[tokio::test(flavor = "current_thread")]
async fn yields_to_the_host_while_decoding() {
    use h5rs::compute::{YieldingCompute, yield_now};

    // 64 MiB of output in 32 chunks of 2 MiB.
    let fixture = Fixture::build(16 * 1024 * 1024, 524_288);
    let path = fixture.path.canonicalize().unwrap();

    let yields = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let counted = yields.clone();
    let pool = YieldingCompute::with_yield(Arc::new(InlineCompute), 8 << 20, move || {
        counted.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Box::pin(yield_now())
    });

    let store = LocalFileSystem::new_with_prefix(path.parent().unwrap()).unwrap();
    let file = ObjectStoreFile::new(
        Box::new(store),
        Path::from(path.file_name().unwrap().to_str().unwrap()),
    )
    .with_compute(Arc::new(pool));

    let ds = h5rs::open_dataset(&file, &["data"]).await.unwrap().unwrap();
    let values = ds.read_full::<u32>(&file).await.unwrap();
    assert_eq!(values.data.len(), fixture.values);

    // 64 MiB decoded, yielding every 8 MiB: about seven breaks, and certainly
    // neither none nor one per chunk.
    let count = yields.load(std::sync::atomic::Ordering::Relaxed);
    assert!(
        (4..=12).contains(&count),
        "expected roughly 7 yields over 64 MiB at 8 MiB apart, got {count}"
    );

    // The data still has to be right: yielding must not reorder or drop chunks.
    let expected = sample_data(values.data.len());
    assert_eq!(values.data, expected);
}

/// A dataset whose chunks are each larger than the memory ceiling still
/// pipelines.
///
/// The ceiling bounds what a read holds, but taken literally it would allow
/// only one request at a time here — nothing downloading while a chunk decodes,
/// nothing decoding while one downloads. A floor of two requests keeps the
/// pipeline alive at the cost of holding about two chunks.
#[tokio::test(flavor = "multi_thread")]
async fn pipelines_when_a_chunk_exceeds_the_memory_ceiling() {
    let fixture = Fixture::build(4 * 1024 * 1024, 1024 * 1024);
    let path = fixture.path.canonicalize().unwrap();

    let trace = Arc::new(Mutex::new(Trace::default()));
    let origin = Instant::now();
    let store = LinkStore::new(
        path.parent().unwrap(),
        Duration::from_millis(1),
        Some(200.0e6),
        trace.clone(),
        origin,
    );
    let pool = Arc::new(TracingCompute {
        inner: Arc::new(InlineCompute),
        origin,
        trace: trace.clone(),
    });
    let file = ObjectStoreFile::with_options(
        Box::new(store),
        Path::from(path.file_name().unwrap().to_str().unwrap()),
        ReadOptions {
            // Well under the ~2 MiB each chunk stores.
            max_inflight_bytes: 64 * 1024,
            ..ReadOptions::default()
        },
    )
    .with_compute(pool);

    let ds = h5rs::open_dataset(&file, &["data"]).await.unwrap().unwrap();
    let values = ds.read_full::<u32>(&file).await.unwrap();
    assert_eq!(values.data, sample_data(fixture.values));

    let trace = trace.lock().unwrap();
    let (peak, _) = concurrency(&trace.requests);
    assert!(
        peak >= 2,
        "a ceiling below one chunk collapsed the pipeline to {peak} request(s) in flight"
    );

    let overlapped = intersection(&trace.decodes, &trace.requests);
    let fraction = ratio(overlapped, union(trace.decodes.clone()));
    assert!(
        fraction > 0.5,
        "only {:.0}% of decoding overlapped a request",
        fraction * 100.0
    );
}

/// The property those measurements are about: decoding overlaps downloading.
///
/// If the reader fetched everything and only then decoded, no decode interval
/// would fall inside the window where requests were outstanding.
#[tokio::test(flavor = "current_thread")]
async fn decoding_overlaps_downloading() {
    let fixture = Fixture::build(2_000_000, 50_000);
    let path = fixture.path.canonicalize().unwrap();
    let trace = Arc::new(Mutex::new(Trace::default()));
    let origin = Instant::now();
    let store = LinkStore::new(
        path.parent().unwrap(),
        Duration::from_millis(2),
        Some(200.0e6),
        trace.clone(),
        origin,
    );
    let pool = Arc::new(TracingCompute {
        inner: Arc::new(InlineCompute),
        origin,
        trace: trace.clone(),
    });
    let file = ObjectStoreFile::with_options(
        Box::new(store),
        Path::from(path.file_name().unwrap().to_str().unwrap()),
        ReadOptions::default(),
    )
    .with_compute(pool);

    let ds = h5rs::open_dataset(&file, &["data"]).await.unwrap().unwrap();
    ds.read_full::<u32>(&file).await.unwrap();

    let trace = trace.lock().unwrap();
    assert!(trace.decodes.len() > 10, "expected many chunks to decode");

    let decode_span = union(trace.decodes.clone());
    let overlapped = intersection(&trace.decodes, &trace.requests);
    let fraction = ratio(overlapped, decode_span);

    // If the reader downloaded everything and only then decoded, no decoding
    // would fall inside a window with a request outstanding.
    assert!(
        fraction > 0.5,
        "only {:.0}% of decoding happened while a request was in flight \
         ({} chunks over {:?})",
        fraction * 100.0,
        trace.decodes.len(),
        decode_span
    );
}
