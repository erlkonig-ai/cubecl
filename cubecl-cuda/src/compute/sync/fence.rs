use cubecl_common::backtrace::BackTrace;
use cubecl_core::server::ServerError;
use cudarc::driver::sys::{CUevent_flags, CUevent_st, CUevent_wait_flags, CUstream_st};

use crate::compute::host_stall_trace::{self, Operation};

/// A fence is simply an [event](CUevent_st) created on a [stream](CUevent_st) that you can wait
/// until completion.
///
/// This is useful for doing synchronization outside of the compute server, which is normally
/// locked by a mutex or a channel. This allows the server to continue accepting other tasks.
#[derive(Debug)]
pub struct Fence {
    event: *mut CUevent_st,
}

// If we don't close the stream or destroy the event, it is safe.
//
// # Safety
//
// Since streams are never closed and we destroy the event after waiting, which consumes the
// [Fence], it is safe.
unsafe impl Send for Fence {}

#[allow(unused)]
impl Fence {
    /// Create a new [Fence] on the given stream.
    ///
    /// # Notes
    ///
    /// The [stream](CUevent_st) must be initialized.
    pub fn new(stream: *mut CUstream_st) -> Self {
        // SAFETY: `stream` must be a valid, initialized CUDA stream (enforced by the doc
        // contract). The event is created and immediately recorded on the stream.
        unsafe {
            let event =
                cudarc::driver::result::event::create(CUevent_flags::CU_EVENT_DEFAULT).unwrap();
            cudarc::driver::result::event::record(event, stream).unwrap();

            Self { event }
        }
    }

    /// Wait for the [Fence] to be reached, ensuring that all previous tasks enqueued to the
    /// [stream](CUstream_st) are completed.
    #[track_caller]
    pub fn wait_sync(self) -> Result<(), ServerError> {
        let caller = std::panic::Location::caller();
        self.wait_sync_traced(Operation::EventWait, 0, || {
            format!(
                "caller={caller} scope=cuda_event_synchronize \
                 may_include_prior_gpu_or_peer_work=true",
            )
        })
    }

    /// The same existing wait, with caller context instead of a second nested
    /// event-wait span. In particular, a readback is not counted twice.
    pub(crate) fn wait_sync_traced(
        self,
        operation: Operation,
        bytes: u64,
        details: impl FnOnce() -> String,
    ) -> Result<(), ServerError> {
        // SAFETY: `self.event` is a valid event created in `Fence::new`. We synchronize
        // (block) until the event completes, then destroy it. `self` is consumed so the
        // event cannot be double-freed.
        unsafe {
            let span = host_stall_trace::start(operation, bytes);
            let result = cudarc::driver::result::event::synchronize(self.event);
            host_stall_trace::finish(span, || {
                format!("success={} {}", result.is_ok(), details())
            });
            result.map_err(|err| {
                ServerError::Generic {
                    reason: format!("{err:?}"),
                    backtrace: BackTrace::capture(),
                }
            })?;
            cudarc::driver::result::event::destroy(self.event).map_err(|err| {
                ServerError::Generic {
                    reason: format!("{err:?}"),
                    backtrace: BackTrace::capture(),
                }
            })?;
        }

        Ok(())
    }

    /// Wait for the [Fence] to be reached, ensuring that all previous tasks enqueued to the
    /// [stream](CUstream_st) are completed on the [original stream](CUstream_st) before new tasks
    /// are registered on the [provided stream](CUstream_st).
    ///
    /// # Notes
    ///
    /// The [stream](CUevent_st) must be initialized.
    pub fn wait_async(self, stream: *mut CUstream_st) {
        // SAFETY: `self.event` is a valid event created in `Fence::new`. `stream` must be
        // a valid CUDA stream. The event is destroyed after the wait, and `self` is consumed
        // so the event cannot be used again.
        unsafe {
            cudarc::driver::result::stream::wait_event(
                stream,
                self.event,
                CUevent_wait_flags::CU_EVENT_WAIT_DEFAULT,
            )
            .unwrap();
            cudarc::driver::result::event::destroy(self.event).unwrap();
        }
    }
}
