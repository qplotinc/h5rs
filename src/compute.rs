//! Where h5rs runs CPU-bound work.
//!
//! Reading a chunked dataset is two kinds of work at once: waiting on the
//! store, and decompressing what comes back. Whichever of the two is scarcer
//! should be the one that decides how long the read takes, and the other should
//! disappear behind it.
//!
//! h5rs drives the I/O itself — it is async, so the caller's executor already
//! decides how that is scheduled — but it has no business deciding where the
//! decompression runs. That depends entirely on the host: a native program may
//! have a Rayon pool, a Tokio blocking pool, or nothing but the current thread;
//! a browser has the main thread, or web workers over a `SharedArrayBuffer`.
//!
//! So decompression is handed to a [`ComputePool`] the caller supplies.
//! [`InlineCompute`] — the default — runs each job on the async task as it
//! arrives, which is right for WASM and correct everywhere.
//! [`ThreadPoolCompute`] spreads jobs over OS threads and needs no async
//! runtime. Anything else — Rayon, a Tokio blocking pool, a pool of web workers
//! — is two methods: hand the job over, and hand a future back.
//!
//! ```
//! use h5rs::compute::{ComputeFuture, ComputeJob, ComputePool};
//! use h5rs::error::H5Result;
//!
//! /// Whatever the host already uses to run CPU-bound work.
//! #[derive(Debug)]
//! struct HostPool {
//!     workers: usize,
//! }
//!
//! impl ComputePool for HostPool {
//!     fn run(&self, job: ComputeJob) -> ComputeFuture {
//!         let (tx, rx) = futures_channel::oneshot::channel();
//!         // Replace this with the host's own submit call — `rayon::spawn`,
//!         // `tokio::task::spawn_blocking`, a message to a worker, and so on.
//!         std::thread::spawn(move || {
//!             let _ = tx.send(job());
//!         });
//!         Box::pin(async move {
//!             rx.await
//!                 .unwrap_or_else(|_| Err(h5rs::error::H5Error::Unsupported("job dropped".into())))
//!         })
//!     }
//!
//!     fn parallelism(&self) -> usize {
//!         self.workers
//!     }
//! }
//! ```

use std::fmt::Debug;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::error::H5Result;

/// `Send + Sync` everywhere except WASM.
///
/// A browser's scheduling primitives — a promise from `setTimeout`, a handle to
/// a worker — are not `Send`, and there is no second thread for them to be sent
/// to. Requiring it there would rule out exactly the pools that WASM hosts need
/// to write.
#[cfg(not(target_arch = "wasm32"))]
pub trait MaybeSendSync: Send + Sync {}
#[cfg(not(target_arch = "wasm32"))]
impl<T: Send + Sync + ?Sized> MaybeSendSync for T {}

/// `Send + Sync` everywhere except WASM. See the native definition.
#[cfg(target_arch = "wasm32")]
pub trait MaybeSendSync {}
#[cfg(target_arch = "wasm32")]
impl<T: ?Sized> MaybeSendSync for T {}

/// The future a [`ComputePool`] hands back. `Send` except on WASM.
#[cfg(not(target_arch = "wasm32"))]
pub type ComputeFuture = futures_core::future::BoxFuture<'static, H5Result<Vec<u8>>>;
/// The future a [`ComputePool`] hands back. `Send` except on WASM.
#[cfg(target_arch = "wasm32")]
pub type ComputeFuture = futures_core::future::LocalBoxFuture<'static, H5Result<Vec<u8>>>;

/// A future that resolves once the host has had a chance to run something else.
#[cfg(not(target_arch = "wasm32"))]
pub type YieldFuture = futures_core::future::BoxFuture<'static, ()>;
/// A future that resolves once the host has had a chance to run something else.
#[cfg(target_arch = "wasm32")]
pub type YieldFuture = futures_core::future::LocalBoxFuture<'static, ()>;

/// One piece of CPU-bound work: decoding a single chunk's bytes.
pub type ComputeJob = Box<dyn FnOnce() -> H5Result<Vec<u8>> + Send + 'static>;

/// Somewhere to run the CPU-bound part of a read.
///
/// [`parallelism`](Self::parallelism) tells h5rs how far ahead to read: it
/// stops pulling more compressed data once enough is queued to keep the pool
/// busy, so a slow decoder cannot be buried under a fast link. Report the
/// number of workers, or 1 for a pool that runs jobs inline.
pub trait ComputePool: Debug + MaybeSendSync + 'static {
    /// Run `job`, resolving to its result.
    ///
    /// The returned future must stay valid even if the pool is dropped while it
    /// is pending — either by keeping the work alive or by failing.
    fn run(&self, job: ComputeJob) -> ComputeFuture;

    /// How many jobs it is worth having outstanding at once.
    fn parallelism(&self) -> usize {
        1
    }
}

/// Runs each job on the async task that submitted it.
///
/// This is the default, and the right choice wherever there is nothing else to
/// run work on — a browser's main thread, or a single-threaded runtime. It
/// still interleaves with I/O: jobs are submitted as each fetch lands, so
/// decoding one chunk overlaps the download of the next.
///
/// The job runs when the returned future is first polled, not when it is
/// submitted. That is what lets [`YieldingCompute`] interpose between chunks.
#[derive(Debug, Default, Clone, Copy)]
pub struct InlineCompute;

impl ComputePool for InlineCompute {
    fn run(&self, job: ComputeJob) -> ComputeFuture {
        Box::pin(async move { job() })
    }
}

/// Whatever the host uses to hand control back: a `setTimeout` promise, a
/// `MessageChannel` round trip, `scheduler.yield()`, or [`yield_now`].
#[cfg(not(target_arch = "wasm32"))]
pub trait YieldSource: Fn() -> YieldFuture + Send + Sync + 'static {}
#[cfg(not(target_arch = "wasm32"))]
impl<T: Fn() -> YieldFuture + Send + Sync + 'static> YieldSource for T {}

/// Whatever the host uses to hand control back. See the native definition.
#[cfg(target_arch = "wasm32")]
pub trait YieldSource: Fn() -> YieldFuture + 'static {}
#[cfg(target_arch = "wasm32")]
impl<T: Fn() -> YieldFuture + 'static> YieldSource for T {}

/// Hands control back to the host every so often.
///
/// Decompression is a tight loop with no await points in it, so on a
/// single-threaded host — a browser tab above all — a large read would hold the
/// thread for as long as it takes to decode every chunk. This wraps another
/// pool and, once roughly `yield_after_bytes` have been decoded, waits on a
/// future the host supplies before letting the next chunk through.
///
/// What that future *is* has to come from the host, because only the host knows
/// what yielding means to it. In a browser it should be something that gives up
/// the current task — `setTimeout(0)`, a `MessageChannel` message, or
/// `scheduler.yield()` where available. A microtask (`queueMicrotask`, or an
/// immediately-resolved promise) is **not** enough: microtasks run before the
/// browser can paint or handle input, so yielding to one changes nothing that
/// the user can see.
///
/// ```
/// use std::sync::Arc;
/// use h5rs::compute::{InlineCompute, YieldingCompute, yield_now};
///
/// // On a browser, replace `yield_now` with a `setTimeout(0)` promise.
/// let pool = YieldingCompute::new(Arc::new(InlineCompute), 4 << 20, || Box::pin(yield_now()));
/// ```
pub struct YieldingCompute {
    inner: Arc<dyn ComputePool>,
    yield_after_bytes: u64,
    decoded_since_yield: Arc<AtomicU64>,
    make_yield: Box<dyn YieldSource>,
}

impl Debug for YieldingCompute {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("YieldingCompute")
            .field("inner", &self.inner)
            .field("yield_after_bytes", &self.yield_after_bytes)
            .finish()
    }
}

impl YieldingCompute {
    /// Wrap `inner`, yielding roughly every `yield_after_bytes` of decoded
    /// output.
    ///
    /// The count is of decompressed bytes, so the gap between yields is bounded
    /// in work done rather than in chunks — a dataset with one huge chunk and
    /// one with many small ones yield at about the same rate. A chunk is never
    /// interrupted part-way, so in practice the gap is at least one chunk.
    pub fn new(
        inner: Arc<dyn ComputePool>,
        yield_after_bytes: u64,
        make_yield: impl YieldSource,
    ) -> YieldingCompute {
        YieldingCompute {
            inner,
            yield_after_bytes: yield_after_bytes.max(1),
            decoded_since_yield: Arc::new(AtomicU64::new(0)),
            make_yield: Box::new(make_yield),
        }
    }
}

impl ComputePool for YieldingCompute {
    fn run(&self, job: ComputeJob) -> ComputeFuture {
        // Decide before starting this chunk whether enough has been decoded
        // since the last break to owe the host a turn. Building the yield future
        // here, rather than inside the async block, keeps `self` out of it.
        let owed = self.decoded_since_yield.load(Ordering::Relaxed) >= self.yield_after_bytes;
        let waiting = owed.then(|| {
            self.decoded_since_yield.store(0, Ordering::Relaxed);
            (self.make_yield)()
        });

        let inner = self.inner.run(job);
        let decoded = self.decoded_since_yield.clone();

        Box::pin(async move {
            // Yield between chunks rather than mid-decode: a decode cannot be
            // interrupted, so this is the only point where it is possible.
            if let Some(waiting) = waiting {
                waiting.await;
            }
            let result = inner.await;
            if let Ok(bytes) = &result {
                decoded.fetch_add(bytes.len() as u64, Ordering::Relaxed);
            }
            result
        })
    }

    fn parallelism(&self) -> usize {
        self.inner.parallelism()
    }
}

/// Give the executor a chance to run something else.
///
/// Returns `Pending` once, waking immediately. That is enough on a native
/// executor, and on WASM it defers to the next microtask — which lets other
/// pending futures run but still does **not** let the browser paint. For that,
/// give [`YieldingCompute`] a future built from `setTimeout` or
/// `scheduler.yield()` instead.
pub async fn yield_now() {
    let mut yielded = false;
    std::future::poll_fn(move |cx| {
        if yielded {
            std::task::Poll::Ready(())
        } else {
            yielded = true;
            cx.waker().wake_by_ref();
            std::task::Poll::Pending
        }
    })
    .await
}

/// The pool h5rs uses when the caller has not chosen one.
pub(crate) fn default_pool() -> Arc<dyn ComputePool> {
    Arc::new(InlineCompute)
}

/// A pool of OS threads, needing no async runtime.
///
/// Useful on native when the host has no executor to lend: decompression runs
/// on these threads while the async task keeps the store busy.
#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug)]
pub struct ThreadPoolCompute {
    sender: std::sync::mpsc::Sender<Task>,
    threads: usize,
}

#[cfg(not(target_arch = "wasm32"))]
struct Task {
    job: ComputeJob,
    reply: futures_channel::oneshot::Sender<H5Result<Vec<u8>>>,
}

#[cfg(not(target_arch = "wasm32"))]
impl ThreadPoolCompute {
    /// Start a pool of `threads` workers.
    ///
    /// The threads run until the pool is dropped.
    pub fn new(threads: usize) -> ThreadPoolCompute {
        let threads = threads.max(1);
        let (sender, receiver) = std::sync::mpsc::channel::<Task>();
        let receiver = Arc::new(std::sync::Mutex::new(receiver));

        for _ in 0..threads {
            let receiver = receiver.clone();
            std::thread::spawn(move || {
                loop {
                    // Hold the lock only long enough to take one task, so the
                    // other workers can take the next one meanwhile.
                    let task = {
                        let guard = receiver.lock().expect("compute queue poisoned");
                        guard.recv()
                    };
                    let Ok(task) = task else {
                        return;
                    };
                    let result = (task.job)();
                    // A dropped receiver just means the read was abandoned.
                    let _ = task.reply.send(result);
                }
            });
        }

        ThreadPoolCompute { sender, threads }
    }

    /// Start a pool sized to the machine's available parallelism.
    pub fn with_available_parallelism() -> ThreadPoolCompute {
        let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
        ThreadPoolCompute::new(threads)
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl ComputePool for ThreadPoolCompute {
    fn run(&self, job: ComputeJob) -> ComputeFuture {
        let (reply, response) = futures_channel::oneshot::channel();
        if self.sender.send(Task { job, reply }).is_err() {
            return Box::pin(std::future::ready(Err(crate::error::H5Error::corrupt(
                "compute pool has shut down",
            ))));
        }
        Box::pin(async move {
            response
                .await
                .unwrap_or_else(|_| Err(crate::error::H5Error::corrupt("compute job was dropped")))
        })
    }

    fn parallelism(&self) -> usize {
        self.threads
    }
}
