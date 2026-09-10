//! Opt-in host-operation diagnostics: `CUBECL_HOST_STALL_TRACE=1`.
//!
//! Only completed operations taking at least 100 ms produce a line. Each line
//! carries process-wide totals for that operation, including faster calls, so
//! silence does NOT mean no allocations or compilation misses. Allocation bytes
//! count requests (including failed ones), not resident memory. A compilation
//! miss means absent from the in-process module map; a disk PTX cache may hit.
//!
//! These are HOST durations around existing calls, with no extra CUDA sync.
//! Allocation, readback and event waits may include prior GPU/peer work; they
//! are not GPU execution times or proof of a cause. Readback waits time the
//! existing CUDA event synchronization when the future is polled, excluding
//! time waiting for that first poll. Enqueue spans measure host submission, not
//! device completion. Totals can overlap across threads and nested operations
//! (for example, a generic event wait inside a drop-flush span).
//! The logging itself is outside the measured interval. A hung call cannot emit
//! its completion line. [`host_operation_snapshot`] also exposes totals when no
//! call crosses the logging threshold. The flag is read once, on the first
//! traced call or explicit snapshot.

use std::{
    io::Write,
    sync::{
        OnceLock,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

const THRESHOLD: Duration = Duration::from_millis(100);

#[derive(Clone, Copy)]
pub(crate) enum Operation {
    CompileMiss,
    AllocAsync,
    AllocSync,
    DropFlush,
    ReadbackEnqueue,
    ReadbackWait,
    EventWait,
    KernelEnqueue,
    CollectiveEnqueue,
}

const OPERATION_COUNT: usize = Operation::CollectiveEnqueue as usize + 1;
const OPERATIONS: [Operation; OPERATION_COUNT] = [
    Operation::CompileMiss,
    Operation::AllocAsync,
    Operation::AllocSync,
    Operation::DropFlush,
    Operation::ReadbackEnqueue,
    Operation::ReadbackWait,
    Operation::EventWait,
    Operation::KernelEnqueue,
    Operation::CollectiveEnqueue,
];

impl Operation {
    fn name(self) -> &'static str {
        match self {
            Self::CompileMiss => "compile_miss",
            Self::AllocAsync => "alloc_async",
            Self::AllocSync => "alloc_sync",
            Self::DropFlush => "drop_flush",
            Self::ReadbackEnqueue => "readback_enqueue",
            Self::ReadbackWait => "readback_wait",
            Self::EventWait => "event_wait",
            Self::KernelEnqueue => "kernel_enqueue",
            Self::CollectiveEnqueue => "collective_enqueue",
        }
    }
}

#[derive(Default)]
struct Totals {
    calls: AtomicU64,
    slow_calls: AtomicU64,
    host_micros: AtomicU64,
    requested_bytes: AtomicU64,
}

impl Totals {
    fn record(&self, elapsed: Duration, bytes: u64) -> bool {
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.host_micros.fetch_add(
            elapsed.as_micros().min(u64::MAX as u128) as u64,
            Ordering::Relaxed,
        );
        self.requested_bytes.fetch_add(bytes, Ordering::Relaxed);
        let slow = elapsed >= THRESHOLD;
        if slow {
            self.slow_calls.fetch_add(1, Ordering::Relaxed);
        }
        slow
    }
}

struct Trace {
    started: Instant,
    totals: [Totals; OPERATION_COUNT],
}

static TRACE: OnceLock<Option<Trace>> = OnceLock::new();

fn active_trace() -> Option<&'static Trace> {
    TRACE.get_or_init(|| {
        enabled_value(std::env::var("CUBECL_HOST_STALL_TRACE").ok().as_deref()).then(|| Trace {
            started: Instant::now(),
            totals: Default::default(),
        })
    }).as_ref()
}

/// Independently loaded, cumulative counters for one kind of completed host call.
///
/// These are relaxed observations, not an atomic transaction. A concurrent
/// completion can become visible in one field before another. Requested bytes
/// include failed requests and are not resident bytes. Durations can include
/// prior GPU/peer work and overlap other operations; they must not be summed as
/// GPU execution time.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HostOperationCounters {
    pub operation: &'static str,
    pub calls_total: u64,
    pub slow_calls_total: u64,
    pub host_micros_total: u64,
    pub requested_bytes_total: u64,
}

/// Process-wide host counters, including calls below the 100 ms logging threshold.
///
/// All counters are sampled independently with relaxed atomic loads. Calls
/// still in progress are not recorded. This snapshot does not synchronize with
/// GPU work, lock an allocator, reset counters, or emit a log line.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HostOperationSnapshot {
    pub pid: u32,
    /// Wall-clock milliseconds since the Unix epoch, if the clock permits it.
    pub unix_ms: Option<u128>,
    /// Monotonic milliseconds since this process initialized host tracing.
    pub trace_ms: u128,
    pub operations: [HostOperationCounters; OPERATION_COUNT],
}

/// Read opt-in host counters without CUDA calls, GPU synchronization, or reset.
///
/// Returns `None` unless `CUBECL_HOST_STALL_TRACE=1` at the first traced call or
/// snapshot. An enabled snapshot taken before any operation contains zero
/// counters. Values are cumulative, relaxed per-counter observations, not an
/// atomic cross-thread snapshot; see [`HostOperationCounters`].
pub fn host_operation_snapshot() -> Option<HostOperationSnapshot> {
    snapshot_for_trace(active_trace())
}

fn snapshot_for_trace(trace: Option<&Trace>) -> Option<HostOperationSnapshot> {
    trace.map(|trace| HostOperationSnapshot {
        pid: std::process::id(),
        unix_ms: SystemTime::now().duration_since(UNIX_EPOCH).ok().map(|d| d.as_millis()),
        trace_ms: trace.started.elapsed().as_millis(),
        operations: OPERATIONS.map(|operation| {
            let totals = &trace.totals[operation as usize];
            HostOperationCounters {
                operation: operation.name(),
                calls_total: totals.calls.load(Ordering::Relaxed),
                slow_calls_total: totals.slow_calls.load(Ordering::Relaxed),
                host_micros_total: totals.host_micros.load(Ordering::Relaxed),
                requested_bytes_total: totals.requested_bytes.load(Ordering::Relaxed),
            }
        }),
    })
}

pub(crate) struct Span {
    trace: &'static Trace,
    operation: Operation,
    bytes: u64,
    started: Instant,
}

impl Span {
    pub(crate) fn requested_bytes(&self) -> u64 {
        self.bytes
    }
}

fn enabled_value(value: Option<&str>) -> bool {
    value == Some("1")
}

pub(crate) fn start(operation: Operation, bytes: u64) -> Option<Span> {
    start_with_bytes(operation, || bytes)
}

/// Compute request metadata only when tracing is enabled.
pub(crate) fn start_with_bytes(
    operation: Operation,
    bytes: impl FnOnce() -> u64,
) -> Option<Span> {
    span_for_trace(active_trace(), operation, bytes)
}

fn span_for_trace(
    trace: Option<&'static Trace>,
    operation: Operation,
    bytes: impl FnOnce() -> u64,
) -> Option<Span> {
    trace.map(|trace| Span {
        trace,
        operation,
        bytes: bytes(),
        started: Instant::now(),
    })
}

pub(crate) fn finish(span: Option<Span>, details: impl FnOnce() -> String) {
    let Some(span) = span else { return };
    let elapsed = span.started.elapsed();
    let totals = &span.trace.totals[span.operation as usize];
    if !totals.record(elapsed, span.bytes) {
        return;
    }
    // Wall time aligns hosts; the local monotonic clock measures the duration.
    // No environment guess at rank: it is the externally formed communicator's
    // rank, or unknown for a process not configured with one (including startup).
    let unix_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis().to_string())
        .unwrap_or_else(|_| "unknown".to_string());
    let rank = super::communication::external_comm()
        .map(|c| format!("{}/{}", c.rank, c.world))
        .unwrap_or_else(|| "unknown".to_string());
    // Totals are process-wide, relaxed snapshots, not an atomic cross-thread
    // transaction. Do not let an unavailable diagnostic sink fail the workload.
    let _ = writeln!(
        std::io::stderr().lock(),
        "[cubecl-host-stall] end_unix_ms={unix_ms} trace_ms={} pid={} thread={:?} external_rank={rank} \
         op={} host_ms={:.3} request_bytes={} calls_total={} slow_calls_total={} \
         host_ms_total={:.3} requested_bytes_total={} timing=host_call_not_gpu {}",
        span.trace.started.elapsed().as_millis(),
        std::process::id(),
        std::thread::current().id(),
        span.operation.name(),
        elapsed.as_secs_f64() * 1000.0,
        span.bytes,
        totals.calls.load(Ordering::Relaxed),
        totals.slow_calls.load(Ordering::Relaxed),
        totals.host_micros.load(Ordering::Relaxed) as f64 / 1000.0,
        totals.requested_bytes.load(Ordering::Relaxed),
        details(),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tracing_requires_explicit_opt_in() {
        assert!(enabled_value(Some("1")));
        for value in [None, Some(""), Some("0"), Some("false"), Some("garbage")] {
            assert!(!enabled_value(value));
        }
    }

    #[test]
    fn totals_include_fast_calls_and_only_report_at_the_threshold() {
        let totals = Totals::default();
        assert!(!totals.record(Duration::from_millis(99), 1024));
        assert!(totals.record(Duration::from_millis(100), 2048));
        assert!(totals.record(Duration::from_millis(101), 4096));
        assert_eq!(totals.calls.load(Ordering::Relaxed), 3);
        assert_eq!(totals.slow_calls.load(Ordering::Relaxed), 2);
        assert_eq!(totals.host_micros.load(Ordering::Relaxed), 300_000);
        assert_eq!(totals.requested_bytes.load(Ordering::Relaxed), 7168);
    }

    #[test]
    fn operation_totals_are_separate() {
        let totals: [Totals; OPERATION_COUNT] = Default::default();
        totals[Operation::AllocAsync as usize].record(THRESHOLD, 512);
        for operation in [
            Operation::CompileMiss,
            Operation::AllocSync,
            Operation::DropFlush,
            Operation::ReadbackEnqueue,
            Operation::ReadbackWait,
            Operation::EventWait,
            Operation::KernelEnqueue,
            Operation::CollectiveEnqueue,
        ] {
            assert_eq!(totals[operation as usize].calls.load(Ordering::Relaxed), 0);
        }
    }

    #[test]
    fn readback_wait_totals_are_not_enqueue_or_generic_event_totals() {
        let totals: [Totals; OPERATION_COUNT] = Default::default();
        let readback = &totals[Operation::ReadbackWait as usize];
        assert!(!readback.record(Duration::from_millis(20), 4096));
        assert!(readback.record(Duration::from_secs(29), 4096));
        assert_eq!(readback.calls.load(Ordering::Relaxed), 2);
        assert_eq!(readback.slow_calls.load(Ordering::Relaxed), 1);
        assert_eq!(readback.host_micros.load(Ordering::Relaxed), 29_020_000);
        assert_eq!(readback.requested_bytes.load(Ordering::Relaxed), 8192);
        for operation in [
            Operation::ReadbackEnqueue,
            Operation::EventWait,
            Operation::KernelEnqueue,
            Operation::CollectiveEnqueue,
        ] {
            assert_eq!(totals[operation as usize].calls.load(Ordering::Relaxed), 0);
        }
    }

    #[test]
    fn disabled_tracing_does_not_format_details() {
        finish(None, || panic!("disabled trace formatted its details"));
    }

    #[test]
    fn disabled_tracing_does_not_count_request_bytes() {
        assert!(
            span_for_trace(None, Operation::ReadbackEnqueue, || {
                panic!("disabled trace evaluated request metadata")
            })
            .is_none()
        );
    }

    #[test]
    fn disabled_snapshot_is_absent() {
        assert!(snapshot_for_trace(None).is_none());
    }

    #[test]
    fn snapshot_exposes_fast_calls_and_zeroes_without_resetting() {
        let trace = Trace { started: Instant::now(), totals: Default::default() };
        let initial = snapshot_for_trace(Some(&trace)).unwrap();
        assert_eq!(initial.pid, std::process::id());
        assert!(initial.operations.iter().all(|op| op.calls_total == 0
            && op.slow_calls_total == 0 && op.host_micros_total == 0
            && op.requested_bytes_total == 0));

        let allocation = &trace.totals[Operation::AllocAsync as usize];
        assert!(!allocation.record(Duration::from_micros(250), 1024));
        assert!(!allocation.record(Duration::from_micros(750), 2048));
        let snapshot = snapshot_for_trace(Some(&trace)).unwrap();
        let expected = HostOperationCounters {
            operation: "alloc_async", calls_total: 2, slow_calls_total: 0,
            host_micros_total: 1000, requested_bytes_total: 3072,
        };
        assert_eq!(snapshot.operations[Operation::AllocAsync as usize], expected);
        assert_eq!(snapshot_for_trace(Some(&trace)).unwrap().operations, snapshot.operations);
        assert!(snapshot.operations.iter().filter(|op| op.operation != "alloc_async")
            .all(|op| op.calls_total == 0 && op.requested_bytes_total == 0));

        assert!(allocation.record(THRESHOLD, 4096));
        let later = snapshot_for_trace(Some(&trace)).unwrap();
        assert_eq!(later.operations[Operation::AllocAsync as usize], HostOperationCounters {
            operation: "alloc_async", calls_total: 3, slow_calls_total: 1,
            host_micros_total: 101_000, requested_bytes_total: 7168,
        });
        assert_eq!(snapshot.operations[Operation::AllocAsync as usize], expected);
    }

    #[test]
    fn snapshot_enumerates_each_operation_once_in_counter_order() {
        for (index, operation) in OPERATIONS.into_iter().enumerate() {
            assert_eq!(operation as usize, index);
        }
        let trace = Trace { started: Instant::now(), totals: Default::default() };
        let snapshot = snapshot_for_trace(Some(&trace)).unwrap();
        assert_eq!(snapshot.operations.map(|op| op.operation), [
            "compile_miss", "alloc_async", "alloc_sync", "drop_flush",
            "readback_enqueue", "readback_wait", "event_wait", "kernel_enqueue",
            "collective_enqueue",
        ]);
    }
}
