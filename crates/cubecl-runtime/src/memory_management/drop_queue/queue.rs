use alloc::vec::Vec;
use cubecl_common::bytes::Bytes;

use crate::memory_management::{
    drop_queue::FlushingPolicy, drop_queue::policy::FlushingPolicyState,
};
use crate::server::ServerError;

/// A synchronization primitive that blocks until the device has finished
/// processing all commands submitted before the fence was created.
pub trait Fence {
    /// Block the current thread until the device signals this fence.
    ///
    /// An error means the wait did not prove that the device finished the
    /// work before the fence. The queue then keeps alive every buffer the
    /// fence protects and returns the error from
    /// [`PendingDropQueue::flush`].
    fn sync(self) -> Result<(), ServerError>;
}

/// Defers the drop of CPU-side [`Bytes`] allocations until the device has
/// finished reading them.
///
/// # How it works
///
/// The device uploads are asynchronous: after you copy bytes into a staging buffer
/// and enqueue an upload command, the CPU memory must remain valid until the
/// device is done. `PendingDropQueue` manages this lifetime with a two-phase
/// approach:
///
/// 1. **Stage** – call [`push`](Self::push) to hand over bytes that are
///    in-flight. They land in the `staged` list.
/// 2. **Flush** – call [`flush`](Self::flush) to rotate the lists. The
///    previously staged bytes move to `pending`, a new [`Fence`] is created
///    to mark the end of the current upload batch, and any bytes that were
///    *already* pending (i.e. the batch before that) are freed after syncing
///    the previous fence.
///
/// This double-buffer scheme means CPU memory is held for at most two flush
/// cycles, while avoiding any unnecessary stalls on the hot path.
///
/// # Flushing policy
///
/// Call [`should_flush`](Self::should_flush) to check whether enough bytes
/// have accumulated to warrant a flush. You may also flush unconditionally
/// (e.g. at the end of a frame).
pub struct PendingDropQueue<E: Fence> {
    /// Fence signalling that the device has consumed everything in `pending`.
    fence: Option<E>,
    /// Bytes from the *previous* flush cycle, kept alive until `event` fires.
    pending: Vec<Bytes>,
    /// Bytes queued in the *current* cycle, not yet associated with a fence.
    staged: Vec<Bytes>,
    /// The configuration of the queue.
    policy: FlushingPolicy,
    /// The current state of the policy.
    policy_state: FlushingPolicyState,
}

impl<E: Fence> core::fmt::Debug for PendingDropQueue<E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PendingDropQueue")
            .field("pending", &self.pending)
            .field("staged", &self.staged)
            .field("policy", &self.policy)
            .field("policy_state", &self.policy_state)
            .finish()
    }
}

impl<E: Fence> Default for PendingDropQueue<E> {
    fn default() -> Self {
        Self::new(Default::default())
    }
}

impl<F: Fence> PendingDropQueue<F> {
    /// Creates a new `PendingDropQueue`.
    pub fn new(policy: FlushingPolicy) -> Self {
        Self {
            fence: None,
            pending: Vec::new(),
            staged: Vec::new(),
            policy,
            policy_state: Default::default(),
        }
    }
    /// Enqueue `bytes` to be dropped once the device has finished reading them.
    ///
    /// The bytes are added to the current staged batch and will be freed on
    /// the flush cycle *after* the next call to [`flush`](Self::flush).
    pub fn push(&mut self, bytes: Bytes) {
        self.policy_state.register(&bytes);
        self.staged.push(bytes);
    }

    /// Returns `true` when the staged batch is large enough to justify a
    /// flush.
    pub fn should_flush(&self) -> bool {
        self.policy_state.should_flush(&self.policy)
    }

    /// Rotate the double-buffer and free any memory the device is done with.
    ///
    /// THE POOL IS ALREADY REUSING THESE BUFFERS; THE BLOCKING IS THE PROOF,
    /// NOT THE REUSE. `reserve_pinned` hands out POOLED slices, so nothing here
    /// is allocating fresh host memory per upload -- the only thing this
    /// function adds is the evidence that the device has finished reading a
    /// slice before the pool may hand the same one out again. That is why the
    /// stall it causes is queue-depth serialisation rather than a data
    /// dependency, and why raising the threshold is not merely "flush less
    /// often": past the depth where the fence being waited on was recorded long
    /// enough ago that the device has already passed it, the wait goes to
    /// approximately zero rather than to a fraction of itself.
    ///
    /// The cure that removes the reason for the flush rather than the frequency
    /// of it is to stop staging per launch at all -- see
    /// `docs/METADATA_CACHE.md` in mary, which is where the 483 uploads a
    /// decode step come from.
    ///
    /// `factory` is called to produce a [`Fence`]. It should submit (or
    /// record) a device signal command so that syncing the fence guarantees all
    /// preceding device work is complete.
    ///
    /// # Errors
    ///
    /// When a fence wait fails, the error is returned and nothing is freed or
    /// rotated: `pending` and `staged` keep their bytes, because the device
    /// may still be reading them. The next flush retries with a fresh fence
    /// through the safety net below.
    pub fn flush<Factory: Fn() -> F>(&mut self, factory: Factory) -> Result<(), ServerError> {
        // Sync the fence from the previous flush and free the bytes it was
        // protecting.
        if let Some(event) = self.fence.take() {
            event.sync()?;
            self.pending.clear();
        }

        // Safety net: if pending is still populated with no fence (none was
        // recorded, or the last wait on it failed), stall now rather than
        // freeing memory the GPU might still be reading.
        if !self.pending.is_empty() {
            factory().sync()?;
            self.pending.clear();
        }

        // The current staged batch becomes the new pending batch.
        core::mem::swap(&mut self.pending, &mut self.staged);

        // Record a fence so the *next* flush knows when this batch is safe to
        // free.
        self.fence = Some(factory());
        self.policy_state.reset();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use core::cell::Cell;

    // ---------------------------------------------------------------------------
    // Test helpers
    // ---------------------------------------------------------------------------

    #[derive(Clone)]
    struct MockFence<'a> {
        sync_count: &'a Cell<u32>,
    }

    impl Fence for MockFence<'_> {
        fn sync(self) -> Result<(), ServerError> {
            self.sync_count.set(self.sync_count.get() + 1);
            Ok(())
        }
    }

    fn make_queue<'a>(
        sync_count: &'a Cell<u32>,
    ) -> (
        PendingDropQueue<MockFence<'a>>,
        impl Fn() -> MockFence<'a> + 'a,
    ) {
        let queue = PendingDropQueue::new(test_policy());
        let factory = move || MockFence { sync_count };
        (queue, factory)
    }

    fn sample_bytes() -> Bytes {
        Bytes::from_elems(vec![1u8, 2, 3])
    }

    fn test_policy() -> FlushingPolicy {
        FlushingPolicy {
            max_bytes_count: 2048,
            max_bytes_size: 8,
        }
    }

    // ---------------------------------------------------------------------------
    // push / should_flush
    // ---------------------------------------------------------------------------

    #[test]
    fn push_at_count_threshold_triggers_flush_hint() {
        let sync_count = Cell::new(0u32);
        let (mut queue, _factory) = make_queue(&sync_count);

        for _ in 0..test_policy().max_bytes_count {
            queue.push(sample_bytes());
        }

        assert!(queue.should_flush());
    }

    #[test]
    fn push_large_allocation_triggers_flush_via_size_threshold() {
        let sync_count = Cell::new(0u32);
        let (mut queue, _factory) = make_queue(&sync_count);
        let big = Bytes::from_elems(vec![0u8; test_policy().max_bytes_size as usize + 1]);

        queue.push(big);

        assert!(queue.should_flush());
    }

    // ---------------------------------------------------------------------------
    // flush – fence / sync behaviour
    // ---------------------------------------------------------------------------

    #[test]
    fn first_flush_creates_fence_without_syncing() {
        let sync_count = Cell::new(0u32);
        let (mut queue, factory) = make_queue(&sync_count);

        queue.push(sample_bytes());
        queue.flush(&factory).unwrap();

        // The fence is created but must not be synced yet — that happens on
        // the next flush.
        assert_eq!(
            sync_count.get(),
            0,
            "fence should not be synced on first flush"
        );
    }

    #[test]
    fn second_flush_syncs_fence_from_first_flush() {
        let sync_count = Cell::new(0u32);
        let (mut queue, factory) = make_queue(&sync_count);

        queue.push(sample_bytes());
        queue.flush(&factory).unwrap(); // flush 1 – creates fence A

        queue.push(sample_bytes());
        queue.flush(&factory).unwrap(); // flush 2 – syncs fence A, creates fence B

        assert_eq!(sync_count.get(), 1, "exactly one sync after two flushes");
    }

    #[test]
    fn each_subsequent_flush_syncs_the_previous_fence() {
        let sync_count = Cell::new(0u32);
        let (mut queue, factory) = make_queue(&sync_count);

        for _ in 0..10 {
            queue.push(sample_bytes());
            queue.flush(&factory).unwrap();
        }

        // Each flush except the first syncs the fence from the previous one.
        assert_eq!(sync_count.get(), 9);
    }

    // ---------------------------------------------------------------------------
    // flush – buffer rotation
    // ---------------------------------------------------------------------------

    #[test]
    fn staged_is_empty_after_flush() {
        let sync_count = Cell::new(0u32);
        let (mut queue, factory) = make_queue(&sync_count);

        for _ in 0..5 {
            queue.push(sample_bytes());
        }
        queue.flush(&factory).unwrap();

        assert!(queue.staged.is_empty());
    }

    #[test]
    fn pending_holds_previously_staged_bytes_after_flush() {
        let sync_count = Cell::new(0u32);
        let (mut queue, factory) = make_queue(&sync_count);

        for _ in 0..5 {
            queue.push(sample_bytes());
        }
        queue.flush(&factory).unwrap();

        assert_eq!(queue.pending.len(), 5);
    }

    #[test]
    fn pending_is_replaced_on_second_flush() {
        let sync_count = Cell::new(0u32);
        let (mut queue, factory) = make_queue(&sync_count);

        for _ in 0..5 {
            queue.push(sample_bytes());
        }
        queue.flush(&factory).unwrap(); // pending = 5 items

        queue.push(sample_bytes());
        queue.flush(&factory).unwrap(); // syncs fence → pending cleared, rotated

        // Only the one item staged between the two flushes should be pending.
        assert_eq!(queue.pending.len(), 1);
    }

    // ---------------------------------------------------------------------------
    // flush – policy state reset
    // ---------------------------------------------------------------------------

    #[test]
    fn should_flush_resets_after_flush() {
        let sync_count = Cell::new(0u32);
        let (mut queue, factory) = make_queue(&sync_count);

        for _ in 0..test_policy().max_bytes_count {
            queue.push(sample_bytes());
        }
        assert!(queue.should_flush());

        queue.flush(&factory).unwrap();

        assert!(
            !queue.should_flush(),
            "policy state should be reset after flush"
        );
    }

    // ---------------------------------------------------------------------------
    // Edge cases
    // ---------------------------------------------------------------------------

    #[test]
    fn flush_on_empty_queue_is_safe() {
        let sync_count = Cell::new(0u32);
        let (mut queue, factory) = make_queue(&sync_count);

        // Should not panic regardless of how many times it is called.
        queue.flush(&factory).unwrap();
        queue.flush(&factory).unwrap();
        queue.flush(&factory).unwrap();
    }

    // ---------------------------------------------------------------------------
    // flush – a fence wait that fails
    // ---------------------------------------------------------------------------

    /// A fence whose wait fails while `fail` is set.
    struct FailingFence<'a> {
        fail: &'a Cell<bool>,
    }

    impl Fence for FailingFence<'_> {
        fn sync(self) -> Result<(), ServerError> {
            match self.fail.get() {
                true => Err(ServerError::Generic {
                    reason: "fence wait failed".into(),
                    backtrace: cubecl_common::backtrace::BackTrace::capture(),
                }),
                false => Ok(()),
            }
        }
    }

    fn tagged(tag: u8) -> Bytes {
        Bytes::from_elems(vec![tag; 4])
    }

    fn tags(list: &[Bytes]) -> Vec<u8> {
        list.iter().map(|bytes| bytes[0]).collect()
    }

    #[test]
    fn a_failed_fence_wait_keeps_the_bytes_it_protects_and_is_returned() {
        let fail = Cell::new(false);
        let factory = || FailingFence { fail: &fail };
        let mut queue = PendingDropQueue::new(test_policy());

        queue.push(tagged(1));
        queue.flush(factory).unwrap(); // 1 is pending behind fence A

        queue.push(tagged(2));
        fail.set(true);
        assert!(
            queue.flush(factory).is_err(),
            "a failed wait must reach the caller"
        );
        // Nothing the device may still read was freed, and nothing rotated.
        assert_eq!(tags(&queue.pending), vec![1]);
        assert_eq!(tags(&queue.staged), vec![2]);

        // While waits keep failing, the safety net fails too and frees nothing.
        assert!(queue.flush(factory).is_err());
        assert_eq!(tags(&queue.pending), vec![1]);
        assert_eq!(tags(&queue.staged), vec![2]);

        // Once a fresh fence completes, 1 is freed and 2 becomes pending.
        fail.set(false);
        queue.flush(factory).unwrap();
        assert_eq!(tags(&queue.pending), vec![2]);
        assert!(queue.staged.is_empty());
    }
}
