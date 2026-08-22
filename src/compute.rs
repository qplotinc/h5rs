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
//! use futures_core::future::BoxFuture;
//! use h5rs::compute::{ComputeJob, ComputePool};
//! use h5rs::error::H5Result;
//!
//! /// Whatever the host already uses to run CPU-bound work.
//! #[derive(Debug)]
//! struct HostPool {
//!     workers: usize,
//! }
//!
//! impl ComputePool for HostPool {
//!     fn run(&self, job: ComputeJob) -> BoxFuture<'static, H5Result<Vec<u8>>> {
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

use futures_core::future::BoxFuture;

use crate::error::H5Result;

/// One piece of CPU-bound work: decoding a single chunk's bytes.
pub type ComputeJob = Box<dyn FnOnce() -> H5Result<Vec<u8>> + Send + 'static>;

/// Somewhere to run the CPU-bound part of a read.
///
/// [`parallelism`](Self::parallelism) tells h5rs how far ahead to read: it
/// stops pulling more compressed data once enough is queued to keep the pool
/// busy, so a slow decoder cannot be buried under a fast link. Report the
/// number of workers, or 1 for a pool that runs jobs inline.
pub trait ComputePool: Debug + Send + Sync + 'static {
    /// Run `job`, resolving to its result.
    ///
    /// The returned future must stay valid even if the pool is dropped while it
    /// is pending — either by keeping the work alive or by failing.
    fn run(&self, job: ComputeJob) -> BoxFuture<'static, H5Result<Vec<u8>>>;

    /// How many jobs it is worth having outstanding at once.
    fn parallelism(&self) -> usize {
        1
    }
}

/// Runs each job immediately on the async task that submitted it.
///
/// This is the default, and the right choice wherever there is nothing else to
/// run work on — a browser's main thread, or a single-threaded runtime. It
/// still interleaves with I/O: jobs are submitted as each fetch lands, so
/// decoding one chunk overlaps the download of the next.
#[derive(Debug, Default, Clone, Copy)]
pub struct InlineCompute;

impl ComputePool for InlineCompute {
    fn run(&self, job: ComputeJob) -> BoxFuture<'static, H5Result<Vec<u8>>> {
        let result = job();
        Box::pin(std::future::ready(result))
    }
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
    fn run(&self, job: ComputeJob) -> BoxFuture<'static, H5Result<Vec<u8>>> {
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
