use cubecl_common::bytes::Bytes;

/// Defines the thresholds that determine when a [`PendingDropQueue`] should be
/// flushed.
///
/// A flush is triggered when **either** limit is exceeded — whichever comes
/// first. Set a field to `u32::MAX` / `usize::MAX` to effectively disable it.
#[derive(Debug)]
pub struct FlushingPolicy {
    /// Flush when this many allocations have been staged.
    pub max_bytes_count: u32,
    /// Flush when the total staged size reaches this many bytes.
    pub max_bytes_size: u32,
}

/// The staged-allocation count a flush triggers at, overridable at runtime.
///
/// MEASURED AND FOUND NOT TO HELP -- IT HURTS. Recorded here rather than in a
/// commit message, because a knob that was tried and rejected is exactly the
/// artifact that evaporates and gets re-proposed six weeks later.
///
/// FRAMING: ms per decode step, two-node pipeline (head zgx-16ec layers 0:21,
/// tail zgx-0d6e layers 21:42), ctx3732, INK_KV=1, INK_GEN=64, 7 reps, arms
/// interleaved within each rep, byte-identical binaries on both boxes, both
/// gated idle. Median over 62 warm passes per rep.
///
///   64      88.6  88.6  88.2  88.7  88.7  88.2  88.1   median 88.6, spread 0.7%
///   4096    89.3  92.3  94.2  90.2  97.6  96.7  92.2   median 92.3, spread 9.3%
///
/// -4.00% on median tok/s, against ~1.1% resolution at 2 sem for 7 reps. The
/// bands do not overlap: 4096's BEST rep (89.3) is worse than 64's WORST
/// (88.7). It is also nine times noisier, which is the more interesting half --
/// a change that widens the spread that much is doing something structural to
/// the step, not adding a constant.
///
/// What this does NOT establish is that the stalls were free. It establishes
/// that this particular cure costs more than they are worth. The queue holds
/// staged buffers for two cycles, so at 4096 it can hold 8192 pinned host
/// buffers where 64 holds 128, and pinned pages cannot be reclaimed on a box
/// whose weights are read through the page cache. Separating "the stalls were
/// free" from "the retention costs more than the stalls" needs the cure that
/// removes the REASON rather than the frequency: a metadata cache, so that a
/// launch whose shapes have not changed uploads nothing and stages nothing.
/// See mary's docs/METADATA_CACHE.md.
///
/// The knob stays because it is the instrument that produced this row, and
/// because 512 is a different question from 4096. It is not a lever to ship.
///
/// WHY THIS IS A KNOB AND WHAT IT MEASURES. A flush waits on the fence from the
/// PREVIOUS flush cycle, so it blocks the launching thread until the device has
/// drained to a point one batch back. That wait needs no VALUE from the device
/// -- it exists only so staging buffers two batches old can be freed -- which
/// makes it queue-depth SERIALISATION rather than a data dependency, and
/// therefore recoverable. The two are indistinguishable in a profile: both are
/// `cuEventSynchronize` on the launching thread.
///
/// The default of 64 was chosen against uploads of ordinary size. A decode step
/// of this model stages roughly 483 of them -- one per launch that binds a
/// ranked tensor, each carrying that tensor's shape and stride list, 19280
/// bytes for the whole step -- so the count threshold fires about seven times a
/// step while the 64 MiB size threshold never fires at all. Raising the count
/// is the A/B that says how much of the blocked time is serialisation, in one
/// binary and one flag, and it costs only that many small pinned buffers held
/// one cycle longer.
fn flush_count_default() -> u32 {
    static N: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("CUBECL_DROP_FLUSH_COUNT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(64)
    })
}

impl Default for FlushingPolicy {
    fn default() -> Self {
        Self {
            max_bytes_count: flush_count_default(),
            max_bytes_size: 64 * 1024 * 1024, // 64 MiB
        }
    }
}

/// Tracks staged allocations and evaluates them against a [`FlushingPolicy`].
#[derive(Default, Debug)]
pub(crate) struct FlushingPolicyState {
    bytes_count: u32,
    bytes_size: u32,
}

impl FlushingPolicyState {
    /// Record a newly staged [`Bytes`] allocation.
    pub(crate) fn register(&mut self, bytes: &Bytes) {
        self.bytes_count += 1;
        self.bytes_size += bytes.len() as u32;
    }

    /// Reset all counters, typically called after a flush.
    pub(crate) fn reset(&mut self) {
        self.bytes_count = 0;
        self.bytes_size = 0;
    }

    /// Returns `true` if either threshold in `policy` has been reached.
    pub(crate) fn should_flush(&self, policy: &FlushingPolicy) -> bool {
        self.bytes_count >= policy.max_bytes_count || self.bytes_size >= policy.max_bytes_size
    }
}

#[cfg(test)]
mod policy_tests {
    use std::vec;

    use super::*;

    fn policy() -> FlushingPolicy {
        FlushingPolicy {
            max_bytes_count: 4,
            max_bytes_size: 100,
        }
    }

    fn state() -> FlushingPolicyState {
        FlushingPolicyState {
            bytes_count: 0,
            bytes_size: 0,
        }
    }

    #[test]
    fn no_flush_when_below_both_thresholds() {
        let s = state();
        assert!(!s.should_flush(&policy()));
    }

    #[test]
    fn flush_when_count_threshold_reached() {
        let mut s = state();
        for _ in 0..4 {
            s.register(&Bytes::from_elems(vec![0u8]));
        }
        assert!(s.should_flush(&policy()));
    }

    #[test]
    fn flush_when_size_threshold_reached() {
        let mut s = state();
        s.register(&Bytes::from_elems(vec![0u8; 101]));
        assert!(s.should_flush(&policy()));
    }

    #[test]
    fn flush_triggered_by_whichever_limit_comes_first() {
        let mut s = state();
        // Only 2 allocations but already over the size limit.
        s.register(&Bytes::from_elems(vec![0u8; 60]));
        s.register(&Bytes::from_elems(vec![0u8; 60]));
        assert!(s.should_flush(&policy()));
    }

    #[test]
    fn reset_clears_state() {
        let mut s = state();
        for _ in 0..4 {
            s.register(&Bytes::from_elems(vec![0u8]));
        }
        assert!(s.should_flush(&policy()));
        s.reset();
        assert!(!s.should_flush(&policy()));
    }
}
