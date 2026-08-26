extern crate alloc;
use super::storage::gpu::{GpuResource, GpuStorage};
use crate::{
    CudaCompiler,
    compute::{
        command::Command,
        communication::{external_comm, get_nccl_comm_id, get_nccl_dtype_count, to_nccl_op},
        context::CudaContext,
        stream::CudaStreamBackend,
        sync::Fence,
    },
};
use cubecl_common::{
    backtrace::BackTrace, bytes::Bytes, profile::ProfileDuration, stream_id::StreamId,
};
use cubecl_core::{
    MemoryConfiguration,
    device::DeviceId,
    future::{self, DynFut},
    ir::{ElemType, FloatKind, IntKind, MemoryDeviceProperties, StorageType, UIntKind},
    prelude::*,
    server::{
        Binding, CommunicationId, CopyDescriptor, GraphLaunchParams, GraphLaunchPatch, Handle,
        KernelArguments, LaunchError, ProfileError, ProfilingToken, ReduceOperation,
        ServerCommunication, ServerError, ServerUtilities, StreamErrorMode, TensorMapBinding,
        TensorMapMeta,
    },
};
use cubecl_runtime::{
    allocator::PitchedMemoryLayoutPolicy,
    compiler::CubeTask,
    config::{CubeClRuntimeConfig, RuntimeConfig},
    logging::ServerLogger,
    memory_management::{ArenaStats, ManagedMemoryHandle, MemoryAllocationMode, MemoryUsage},
    server::ComputeServer,
    storage::{ComputeStorage, ManagedResource},
    stream::MultiStream,
};
use cudarc::driver::sys::{
    CUstream_st, CUtensorMapDataType, CUtensorMapFloatOOBfill, CUtensorMapInterleave,
    CUtensorMapL2promotion, CUtensorMapSwizzle, cuTensorMapEncodeIm2col, cuTensorMapEncodeTiled,
};
use std::{
    collections::{HashMap, hash_map::Entry},
    ffi::c_void,
    mem::MaybeUninit,
    sync::Arc,
};

pub(crate) const MB: usize = 1024 * 1024;

#[derive(Debug)]
pub struct CudaServer {
    ctx: CudaContext,
    device_id: DeviceId,
    streams: MultiStream<CudaStreamBackend>,
    utilities: Arc<ServerUtilities<Self>>,
    comm_stream: *mut CUstream_st,
    communicators: HashMap<CommunicationId, *mut cudarc::nccl::sys::ncclComm>,
    /// Captured graphs, by the id handed back to the caller. They are created,
    /// launched and destroyed only here, on the one thread that owns the
    /// server, because a CUgraph is not internally synchronized.
    graphs: HashMap<u64, CapturedGraph>,
    next_graph_id: u64,
    /// Every buffer bound to a launch while a capture is open, moved into the
    /// [`CapturedGraph`] at `graph_capture_end`. See `CapturedGraph::hold` for
    /// why holding them is a correctness requirement and not caution.
    capture_hold: Vec<Binding>,
    /// Every kernel launch made while a capture is open, in launch order, moved
    /// into the [`CapturedGraph`] at `graph_capture_end`. This is the index a
    /// later parameter rewrite names its node by.
    capture_launches: Vec<CapturedLaunch>,
    /// Where in the open arena window the capture began, so `graph_capture_end`
    /// can sign the requests THIS capture made and not the warm pass's.
    capture_arena_mark: usize,
    /// The signature of the most recently closed capture. Every graph still
    /// replayable must agree with it; see [`CapturedGraph::arena_signature`].
    last_arena_signature: Option<u64>,
}

/// One launch a capture recorded, with everything needed to rewrite it.
///
/// CUDA has no "change only this argument" call: `cuGraphExecKernelNodeSetParams`
/// takes a WHOLE `CUDA_KERNEL_NODE_PARAMS`, so a node that is to be patched has
/// to be reconstructible from what is stored here alone. That is why the
/// pointers and the packed blob are owned copies rather than the caller's
/// buffers -- the caller's are gone by the time a patch happens.
///
/// The argument order mirrors the launch exactly: the tensor map structs, then
/// one pointer per bound resource (the tensor maps' own buffers first, then the
/// ordinary buffers, then the metadata buffer if there is one), then the packed
/// blob if it rides as a by-value grid constant.
#[derive(Debug)]
struct CapturedLaunch {
    node: cudarc::driver::sys::CUgraphNode,
    func: cudarc::driver::sys::CUfunction,
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
    shared: u32,
    tensor_maps: Vec<cudarc::driver::sys::CUtensorMap>,
    ptrs: Vec<u64>,
    info: Vec<u64>,
    info_is_grid_constant: bool,
    /// Where the DYNAMIC half of `info` starts.
    ///
    /// The packed blob is two things end to end and only the first reaches the
    /// kernel by value. `info[..dyn_offset]` is the scalars plus the static
    /// metadata -- buffer lengths, ranks, and the offsets into the dynamic half
    /// -- and it rides as a grid constant, so a parameter rewrite moves it.
    /// `info[dyn_offset..]` is every bound tensor's SHAPE and STRIDE list,
    /// which is variable-length and therefore cannot be a fixed-size kernel
    /// parameter: it is uploaded to a device buffer instead, by a memcpy node
    /// that no `cuGraphExecKernelNodeSetParams` reaches. Splitting a moved word
    /// on this offset is what says whether it is patchable or staged, and
    /// before it existed the two were reported as one number.
    dyn_offset: usize,
    /// Which of the graph's pinned staging buffers this launch's dynamic
    /// metadata was uploaded from, if it had any.
    ///
    /// This is the patch channel for the half a parameter rewrite cannot reach.
    /// The buffer belongs to the graph and its address is what the memcpy node
    /// recorded, so writing new bytes into it -- a host memcpy, no driver call
    /// at all -- makes the next replay upload the CURRENT step's shapes and
    /// strides through the node captured for the old ones.
    staging: Option<usize>,
    /// The kernel, named. Diagnostics only, and only populated during a
    /// capture: a region with a hundred moving launches is an inventory, and an
    /// inventory of indices is not one anybody can act on.
    name: alloc::string::String,
}

/// Whether a capture holds the buffers its nodes point at. On, always, in any
/// run whose answer matters -- `CUBECL_GRAPH_HOLD=0` is the arm that shows what
/// it is worth, and what it costs, in one binary.
fn capture_hold_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("CUBECL_GRAPH_HOLD").as_deref() != Ok("0"))
}

/// Whether a capture allocates from the CAPTURE ARENA. On, always, in any run
/// whose answer matters; `CUBECL_GRAPH_ARENA=0` is the arm that shows what it is
/// worth, in the same binary. With it off, a capture allocates from the ordinary
/// pools and every buffer it binds is held -- which is correct, and is where the
/// intra-region allocations that become graph MEMORY nodes come from.
fn capture_arena_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("CUBECL_GRAPH_ARENA").as_deref() != Ok("0"))
}

/// One captured region: the graph, the executable instantiated from it, and
/// every buffer its nodes hold a pointer to.
///
/// `hold` is not bookkeeping. A graph records device ADDRESSES, so a slice any
/// node reads or writes must stay allocated -- and, just as importantly, must
/// not be handed to anything else -- for as long as the graph can be replayed.
/// Without it the allocator recycles a buffer that dies inside the region into
/// a later node of the SAME region, and the second replay reads what the first
/// replay wrote over its own input. The first replay does not show it, because
/// the first replay still finds the pre-region value there; what shows is the
/// step after, reading state the region was supposed to have carried.
#[derive(Debug)]
struct CapturedGraph {
    graph: cudarc::driver::sys::CUgraph,
    exec: cudarc::driver::sys::CUgraphExec,
    hold: Vec<Binding>,
    launches: Vec<CapturedLaunch>,
    /// A hash of the sizes this capture asked the arena for, in order: the
    /// signature of the REGION it recorded.
    ///
    /// Graphs that share an arena share its slices, which is deliberate --
    /// without it two captures of one region would not be comparable and no
    /// graph could stand in for another step. They then share scratch, and
    /// that is safe exactly when they are captures of the same region,
    /// replayed serially. Serial the single stream gives; same-region is this.
    /// A graph whose signature disagrees with the last capture's is a graph
    /// whose scratch is now somebody else's live memory, and it would compute
    /// a plausible wrong answer rather than fail.
    arena_signature: u64,
    /// The pinned host buffers that captured H2D copies read FROM.
    ///
    /// A memcpy node records a source ADDRESS the same way a kernel node
    /// records its arguments. The staging buffer for a metadata upload is
    /// normally handed straight back to the pinned pool -- during a capture the
    /// drop queue defers that, but the deferral ends when the capture does, and
    /// then the pool is free to hand the same host page to something else while
    /// the graph still copies out of it on every replay. Owning them here is
    /// what makes the recorded source address mean what the node says it means.
    staging: Vec<Bytes>,
}

// SAFETY: `CudaServer` is only accessed from one thread at a time via the `DeviceHandle`,
// which serializes all server access. The CUDA context, streams, and NCCL communicators
// it manages are never shared across threads without synchronization.
unsafe impl Send for CudaServer {}

impl ComputeServer for CudaServer {
    type Kernel = Box<dyn CubeTask<CudaCompiler>>;
    type Storage = GpuStorage;
    type MemoryLayoutPolicy = PitchedMemoryLayoutPolicy;
    type Info = ();

    fn logger(&self) -> Arc<ServerLogger> {
        self.streams.logger.clone()
    }

    fn staging(&mut self, sizes: &[usize], stream_id: StreamId) -> Result<Vec<Bytes>, ServerError> {
        let mut command = self.command_no_inputs(
            stream_id,
            StreamErrorMode {
                ignore: true,
                flush: false,
            },
        )?;

        Ok(sizes
            .iter()
            .map(|size| command.reserve_cpu(*size, true, None))
            .collect())
    }

    fn utilities(&self) -> Arc<ServerUtilities<Self>> {
        self.utilities.clone()
    }

    fn read(
        &mut self,
        descriptors: Vec<CopyDescriptor>,
        stream_id: StreamId,
    ) -> DynFut<Result<Vec<Bytes>, ServerError>> {
        match self.command(
            stream_id,
            descriptors.iter().map(|d| &d.handle),
            StreamErrorMode {
                ignore: false,
                flush: true,
            },
        ) {
            Ok(mut command) => Box::pin(command.read_async(descriptors)),
            Err(err) => Box::pin(async move { Err(err) }),
        }
    }

    fn initialize_memory(&mut self, memory: ManagedMemoryHandle, size: u64, stream_id: StreamId) {
        let mut command = match self.command_no_inputs(
            stream_id,
            StreamErrorMode {
                ignore: true,
                flush: false,
            },
        ) {
            Ok(val) => val,
            Err(err) => unreachable!("{err:?}"),
        };

        let reserved = command.reserve(size).unwrap();
        command.bind(reserved, memory);
    }

    /// Zero-copy: expose an mmap'd host region to kernels as a tensor handle.
    ///
    /// Unlike the wgpu/Metal implementation, which has to build a real buffer
    /// object with `newBufferWithBytesNoCopy:` over a page-aligned superset,
    /// this does nothing at all to the memory. GB10 reports
    /// `pageableMemoryAccessUsesHostPageTables = 1`: the device walks the host
    /// page tables, `cudaHostGetDevicePointer` hands back the identical address,
    /// and a kernel dereferences an ordinary file-backed `mmap` correctly with
    /// no registration. [`crate::supports_zero_copy_host`] is asserted here, so a
    /// discrete part refuses loudly instead of silently reading garbage.
    /// `page_len` is accepted for signature compatibility
    /// and used only to bounds-check; there is no page-alignment requirement to
    /// satisfy because there is no buffer being created.
    ///
    /// What still matters, and matters more than on Metal, is lifetime — a
    /// kernel reading unmapped pages does not reliably fault, it reads whatever
    /// the address now holds. `keepalive` owns the mapping and is dropped only
    /// when the storage entry is deallocated.
    fn graph_capture_supported(&self) -> bool {
        true
    }

    /// `INK_GRAPH_CAPTURE_MODE` selects the capture mode. `thread-local` (the
    /// default) makes the driver REJECT a host-blocking call made from this
    /// thread while capture is open, which is what turns a silent
    /// wrong-capture into a loud error. `relaxed` only polices the capturing
    /// stream itself and is here to answer "what exactly is it rejecting?"
    /// during bring-up -- it is not a fix, because the calls it stops
    /// rejecting are the ones that would bake a stale pointer into the graph.
    fn graph_defer_frees(&mut self, defer: bool, stream_id: StreamId) {
        self.raw_stream(stream_id, Some(defer));
    }

    fn graph_arena_begin(&mut self, stream_id: StreamId) -> u64 {
        // The ablation arm has to be the OLD path exactly, not the arena with
        // its reuse suppressed by a full hold. With this off, nothing is ever
        // routed here and a capture allocates from the ordinary pools.
        if !capture_arena_enabled() {
            return 0;
        }
        self.arena_command(stream_id)
            .streams
            .current()
            .memory_management_gpu
            .arena_begin()
    }

    fn graph_arena_end(&mut self, stream_id: StreamId) {
        if !capture_arena_enabled() {
            return;
        }
        self.arena_command(stream_id)
            .streams
            .current()
            .memory_management_gpu
            .arena_end();
    }

    fn graph_arena_stats(&mut self, stream_id: StreamId) -> ArenaStats {
        self.arena_command(stream_id)
            .streams
            .current()
            .memory_management_gpu
            .arena_stats()
    }

    fn graph_arena_reset_counters(&mut self, stream_id: StreamId) {
        self.arena_command(stream_id)
            .streams
            .current()
            .memory_management_gpu
            .arena_reset_counters();
    }

    fn reserve_timing(&mut self, stream_id: StreamId) -> (u64, u64) {
        self.arena_command(stream_id)
            .streams
            .current()
            .memory_management_gpu
            .reserve_timing()
    }

    fn reserve_timing_reset(&mut self, stream_id: StreamId) {
        self.arena_command(stream_id)
            .streams
            .current()
            .memory_management_gpu
            .reserve_timing_reset();
    }

    fn graph_capture_begin(&mut self, stream_id: StreamId) {
        use cudarc::driver::sys::CUstreamCaptureMode;
        let mode = match std::env::var("INK_GRAPH_CAPTURE_MODE").as_deref() {
            Ok("relaxed") => CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED,
            Ok("global") => CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_GLOBAL,
            _ => CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
        };
        self.unsafe_set_current();
        let stream = self.raw_stream(stream_id, Some(true));
        if capture_arena_enabled() {
            let open = self
                .arena_command(stream_id)
                .streams
                .current()
                .memory_management_gpu
                .arena_active();
            assert!(
                open,
                "graph_capture_begin with the capture arena closed. Every address this \
                 capture is about to record would come from the ordinary pools, which are \
                 free to hand the same slice to a later node of this same region and to \
                 anything at all once the capture closes -- and neither shows up as a \
                 failure, only as a wrong answer one step later. Call graph_arena_begin \
                 first (once to warm, then again around the capture), or set \
                 CUBECL_GRAPH_ARENA=0 to take the hold-everything arm deliberately."
            );
        }
        let mark = self
            .arena_command(stream_id)
            .streams
            .current()
            .memory_management_gpu
            .arena_mark();
        self.capture_arena_mark = mark;
        crate::compute::CAPTURE_OPEN.store(true, core::sync::atomic::Ordering::Relaxed);
        self.capture_hold.clear();
        self.capture_launches.clear();
        unsafe { cudarc::driver::result::stream::begin_capture(stream, mode) }
            .expect("stream capture to begin");
    }

    fn graph_capture_end(&mut self, stream_id: StreamId) -> u64 {
        self.unsafe_set_current();
        let stream = self.raw_stream(stream_id, Some(false));
        crate::compute::CAPTURE_OPEN.store(false, core::sync::atomic::Ordering::Relaxed);
        let graph = unsafe { cudarc::driver::result::stream::end_capture(stream) }
            .expect("stream capture to end");
        assert!(
            !graph.is_null(),
            "the capture closed with no graph -- it was invalidated while open"
        );
        // Flags 0. `CUDA_GRAPH_INSTANTIATE_FLAG_UPLOAD` is NOT valid here: it
        // is only honoured by `cuGraphInstantiateWithParams`, which takes the
        // stream to upload on, and passing it to the flags-only entry point is
        // rejected with CUDA_ERROR_INVALID_VALUE. cudarc's `graph::instantiate`
        // types its argument as the flags enum, which has no zero variant, so
        // the raw call is the honest way to say "no flags".
        let exec = unsafe {
            let mut exec = core::mem::MaybeUninit::uninit();
            // AUTO_FREE_ON_LAUNCH (1). A capture that could not avoid allocating
            // holds MEMORY NODES, and a graph with unfreed memory nodes refuses
            // to launch -- `cuGraphInstantiateWithFlags` accepts it and
            // `cuGraphLaunch` then returns CUDA_ERROR_INVALID_VALUE, three calls
            // away from the cause. This flag frees such nodes at each launch,
            // which is what makes the graph relaunchable at all. It costs
            // nothing on a graph that has none, and the pre-warm exists to make
            // that the normal case: measured at 21 layers, exactly ONE
            // allocation escaped into the capture, the KV tail page that grows
            // by a row every step.
            cudarc::driver::sys::cuGraphInstantiateWithFlags(exec.as_mut_ptr(), graph, 1)
                .result()
                .expect("the captured graph to instantiate");
            exec.assume_init()
        };
        // Upload separately, which is what the flag would have done, so the
        // first replay is not paying for the graph's own setup.
        unsafe { cudarc::driver::result::graph::upload(exec, stream) }
            .expect("the graph to upload");
        let id = self.next_graph_id;
        self.next_graph_id += 1;
        let hold = core::mem::take(&mut self.capture_hold);
        let launches = core::mem::take(&mut self.capture_launches);
        let mark = self.capture_arena_mark;
        let (arena_signature, staging) = {
            let mut command = self.arena_command(stream_id);
            let stream = command.streams.current();
            let staging = core::mem::take(&mut stream.capture_staging);
            let sig = stream.memory_management_gpu.arena_signature(mark);
            (sig, staging)
        };
        self.last_arena_signature = Some(arena_signature);
        self.graphs.insert(
            id,
            CapturedGraph {
                graph,
                exec,
                hold,
                launches,
                arena_signature,
                staging,
            },
        );
        id
    }

    fn graph_replay(&mut self, id: u64, stream_id: StreamId) {
        if capture_arena_enabled() && let Some(last) = self.last_arena_signature {
            let built = self
                .graphs
                .get(&id)
                .unwrap_or_else(|| panic!("no captured graph with id {id}"))
                .arena_signature;
            assert_eq!(
                built, last,
                "graph {id} recorded a different sequence of arena requests than the most \
                 recent capture did, so the two are not captures of the same region -- and \
                 they share the arena's slices. Replaying this one now would compute on \
                 scratch the other holds live, and would emit a PLAUSIBLE answer rather \
                 than fail. Give a second region its own arena."
            );
        }
        self.unsafe_set_current();
        let stream = self.raw_stream(stream_id, None);
        let exec = self
            .graphs
            .get(&id)
            .unwrap_or_else(|| panic!("no captured graph with id {id}"))
            .exec;
        unsafe { cudarc::driver::result::graph::launch(exec, stream) }.expect("the graph to launch");
    }

    fn graph_capture_status(&mut self, stream_id: StreamId) -> u32 {
        self.unsafe_set_current();
        let stream = self.raw_stream(stream_id, None);
        match unsafe { cudarc::driver::result::stream::is_capturing(stream) } {
            Ok(cudarc::driver::sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE) => 0,
            Ok(cudarc::driver::sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_ACTIVE) => 1,
            Ok(_) => 2,
            Err(_) => 3,
        }
    }

    fn graph_node_count(&mut self, id: u64) -> usize {
        let graph = self
            .graphs
            .get(&id)
            .unwrap_or_else(|| panic!("no captured graph with id {id}"))
            .graph;
        let mut n: usize = 0;
        unsafe {
            cudarc::driver::sys::cuGraphGetNodes(graph, core::ptr::null_mut(), &mut n)
                .result()
                .expect("the graph to report its node count");
        }
        n
    }

    fn graph_capture_launch_count(&mut self) -> usize {
        self.capture_launches.len()
    }

    fn graph_launch_count(&mut self, id: u64) -> usize {
        self.captured(id).launches.len()
    }

    fn graph_launch_params(&mut self, id: u64, idx: usize) -> GraphLaunchParams {
        let launches = &self.captured(id).launches;
        let l = launches.get(idx).unwrap_or_else(|| {
            panic!(
                "graph {id} recorded {} launches, so there is no launch {idx}",
                launches.len()
            )
        });
        GraphLaunchParams {
            grid: l.grid,
            block: l.block,
            info: l.info.clone(),
            info_is_grid_constant: l.info_is_grid_constant,
            ptrs: l.ptrs.clone(),
            dyn_offset: l.dyn_offset,
            has_staging: l.staging.is_some(),
            name: l.name.clone(),
        }
    }

    /// Rewrite one launch of an instantiated graph.
    ///
    /// The GRAPH is not touched, only the exec, which is what makes this worth
    /// doing per step: re-capturing the region would cost exactly the host time
    /// the region exists to remove.
    fn graph_patch_launch(&mut self, id: u64, idx: usize, patch: GraphLaunchPatch) {
        self.unsafe_set_current();
        let g = self
            .graphs
            .get_mut(&id)
            .unwrap_or_else(|| panic!("no captured graph with id {id}"));
        let exec = g.exec;
        let n = g.launches.len();
        // THE STAGED HALF, rewritten before the by-value half, and not through
        // the driver at all.
        //
        // `info[dyn_offset..]` is the shapes and strides, and it reaches the
        // kernel through a memcpy node that `cuGraphExecKernelNodeSetParams`
        // cannot touch. What it CAN be reached through is the node's recorded
        // SOURCE: a pinned buffer this graph owns, at an address fixed for the
        // graph's life. Writing the new bytes there is a host memcpy -- no
        // driver call, no node edit -- and the next replay uploads them through
        // the node captured for the old ones. That is the whole fix for the
        // half a parameter rewrite was said not to reach.
        //
        // Done first because it borrows `g.staging` while the by-value rewrite
        // borrows `g.launches`, and because a patch that fails half way should
        // fail before it has edited the exec.
        if let Some(info) = patch.info.as_ref() {
            let l = g.launches.get(idx).unwrap_or_else(|| {
                panic!("graph {id} recorded {n} launches, so there is no launch {idx}")
            });
            let (off, slot) = (l.dyn_offset, l.staging);
            if off < info.len() && info[off..] != l.info[off..] {
                let src = &info[off..];
                let slot = slot.unwrap_or_else(|| {
                    panic!(
                        "launch {idx} of graph {id} has moving DYNAMIC metadata -- a bound \
                         tensor's shape or stride changed -- and no staging buffer to write it \
                         into. Its memcpy node would keep uploading the captured step's shapes, \
                         and the replay would compute a plausible wrong answer rather than fail. \
                         The launch was captured with CUBECL_GRAPH_STAGE_OWN=0, which is the arm \
                         that hands staging back to the pool."
                    )
                });
                let dst = g.staging.get_mut(slot).unwrap_or_else(|| {
                    panic!("graph {id} owns no staging buffer {slot} for launch {idx}")
                });
                let bytes: &[u8] = bytemuck::cast_slice(src);
                assert_eq!(
                    bytes.len(),
                    dst.len(),
                    "launch {idx} of graph {id} staged {} bytes of dynamic metadata and the \
                     patch offers {} -- a memcpy node's SIZE is recorded, so a different length \
                     is a different region, not a different value",
                    dst.len(),
                    bytes.len()
                );
                dst.copy_from_slice(bytes);
            }
        }
        let l = g
            .launches
            .get_mut(idx)
            .unwrap_or_else(|| panic!("graph {id} recorded {n} launches, so there is no launch {idx}"));

        if let Some(grid) = patch.grid {
            l.grid = grid;
        }
        if let Some(info) = patch.info {
            assert!(
                l.info_is_grid_constant,
                "launch {idx} of graph {id} passes its scalars in a DEVICE BUFFER, not as a                  by-value grid constant, so no parameter rewrite can move them -- the value                  would have to be written to the buffer instead, and the buffer is shared"
            );
            assert_eq!(
                info.len(),
                l.info.len(),
                "the packed argument blob is a fixed-size kernel parameter: launch {idx} of                  graph {id} was captured with {} words and the patch offers {}",
                l.info.len(),
                info.len()
            );
            l.info = info;
        }
        for (i, ptr) in patch.ptrs {
            assert!(
                i < l.ptrs.len(),
                "launch {idx} of graph {id} bound {} buffers, so there is no binding {i}",
                l.ptrs.len()
            );
            l.ptrs[i] = ptr;
        }

        // The argument array, rebuilt in the same order the launch presented
        // it: tensor map structs, then one pointer per bound resource, then the
        // packed blob when it rides as a grid constant. Every element points
        // into `l`, which outlives this call.
        let mut params: Vec<*mut c_void> =
            Vec::with_capacity(l.tensor_maps.len() + l.ptrs.len() + 1);
        for m in l.tensor_maps.iter() {
            params.push(m as *const _ as *mut c_void);
        }
        for ptr in l.ptrs.iter() {
            params.push(ptr as *const u64 as *mut c_void);
        }
        if l.info_is_grid_constant && !l.info.is_empty() {
            params.push(l.info.as_ptr() as *mut c_void);
        }

        // Zeroed rather than braced, because the struct grew `kern` and `ctx`
        // fields in CUDA 12 and this has to build against both shapes. Zero is
        // the documented "use `func`" value for them.
        let mut np: cudarc::driver::sys::CUDA_KERNEL_NODE_PARAMS =
            unsafe { core::mem::zeroed() };
        np.func = l.func;
        np.gridDimX = l.grid.0;
        np.gridDimY = l.grid.1;
        np.gridDimZ = l.grid.2;
        np.blockDimX = l.block.0;
        np.blockDimY = l.block.1;
        np.blockDimZ = l.block.2;
        np.sharedMemBytes = l.shared;
        np.kernelParams = params.as_mut_ptr();
        np.extra = core::ptr::null_mut();

        // SAFETY: `exec` is a live instantiated graph and `l.node` one of its
        // nodes; `np` names the same function, the same shared memory size and
        // the same number of parameters the node was captured with, which is
        // what `cuGraphExecKernelNodeSetParams` requires. Every parameter
        // pointer targets storage owned by `l` and outliving the call.
        unsafe {
            cudarc::driver::sys::cuGraphExecKernelNodeSetParams_v2(exec, l.node, &np)
                .result()
                .unwrap_or_else(|err| {
                    panic!("launch {idx} of graph {id} refused the parameter rewrite: {err:?}")
                });
        }
    }

    fn graph_node_kinds(&mut self, id: u64) -> Vec<(u32, usize)> {
        use cudarc::driver::sys::{CUgraphNode, cuGraphGetNodes, cuGraphNodeGetType};
        let graph = self.captured(id).graph;
        let mut n: usize = 0;
        // SAFETY: `graph` is a live captured graph. A null node array asks for
        // the count only, which is the documented two-call form.
        unsafe {
            cuGraphGetNodes(graph, core::ptr::null_mut(), &mut n)
                .result()
                .expect("the graph to report its node count");
        }
        let mut nodes: Vec<CUgraphNode> = vec![core::ptr::null_mut(); n];
        // SAFETY: `nodes` has room for exactly the `n` the call above reported.
        unsafe {
            cuGraphGetNodes(graph, nodes.as_mut_ptr(), &mut n)
                .result()
                .expect("the graph to list its nodes");
        }
        let mut counts: HashMap<u32, usize> = HashMap::default();
        for node in nodes.into_iter().take(n) {
            let mut kind = core::mem::MaybeUninit::uninit();
            // SAFETY: `node` came from `cuGraphGetNodes` on a live graph.
            let kind = unsafe {
                cuGraphNodeGetType(node, kind.as_mut_ptr())
                    .result()
                    .expect("a graph node to report its type");
                kind.assume_init()
            };
            *counts.entry(kind as u32).or_default() += 1;
        }
        let mut out: Vec<(u32, usize)> = counts.into_iter().collect();
        out.sort_unstable();
        out
    }

    fn graph_patch_launches(&mut self, id: u64, patches: Vec<(usize, GraphLaunchPatch)>) {
        for (idx, patch) in patches {
            self.graph_patch_launch(id, idx, patch);
        }
    }

    fn graph_alloc_regions(&mut self, id: u64) -> Vec<(u64, u64)> {
        use cudarc::driver::sys::{
            CUDA_MEM_ALLOC_NODE_PARAMS, CUgraphNode, CUgraphNodeType, cuGraphGetNodes,
            cuGraphMemAllocNodeGetParams, cuGraphNodeGetType,
        };
        let graph = self.captured(id).graph;
        let mut n: usize = 0;
        // SAFETY: `graph` is a live captured graph; the null array asks for the
        // count only, which is the documented two-call form.
        unsafe {
            cuGraphGetNodes(graph, core::ptr::null_mut(), &mut n)
                .result()
                .expect("the graph to report its node count");
        }
        let mut nodes: Vec<CUgraphNode> = vec![core::ptr::null_mut(); n];
        // SAFETY: `nodes` has room for exactly the `n` reported above.
        unsafe {
            cuGraphGetNodes(graph, nodes.as_mut_ptr(), &mut n)
                .result()
                .expect("the graph to list its nodes");
        }
        let mut out = Vec::new();
        for node in nodes.into_iter().take(n) {
            // SAFETY: `node` came from `cuGraphGetNodes` on a live graph.
            let kind = unsafe {
                let mut k = core::mem::MaybeUninit::uninit();
                cuGraphNodeGetType(node, k.as_mut_ptr())
                    .result()
                    .expect("a graph node to report its type");
                k.assume_init()
            };
            if kind != CUgraphNodeType::CU_GRAPH_NODE_TYPE_MEM_ALLOC {
                continue;
            }
            // SAFETY: the node's type was just checked to be MEM_ALLOC, which
            // is the precondition for reading its allocation parameters.
            let params: CUDA_MEM_ALLOC_NODE_PARAMS = unsafe {
                let mut p = core::mem::MaybeUninit::zeroed();
                cuGraphMemAllocNodeGetParams(node, p.as_mut_ptr())
                    .result()
                    .expect("a memory node to report what it allocates");
                p.assume_init()
            };
            out.push((params.dptr, params.bytesize as u64));
        }
        out.sort_unstable();
        out
    }

    fn graph_memcpy_specs(&mut self, id: u64) -> Vec<(u64, u64, u64, u32)> {
        use cudarc::driver::sys::{
            CUDA_MEMCPY3D, CUgraphNode, CUgraphNodeType, cuGraphGetNodes,
            cuGraphMemcpyNodeGetParams, cuGraphNodeGetType,
        };
        let graph = self.captured(id).graph;
        let mut n: usize = 0;
        // SAFETY: `graph` is live; the null array asks for the count only.
        unsafe {
            cuGraphGetNodes(graph, core::ptr::null_mut(), &mut n)
                .result()
                .expect("the graph to report its node count");
        }
        let mut nodes: Vec<CUgraphNode> = vec![core::ptr::null_mut(); n];
        // SAFETY: `nodes` has room for exactly the `n` reported above.
        unsafe {
            cuGraphGetNodes(graph, nodes.as_mut_ptr(), &mut n)
                .result()
                .expect("the graph to list its nodes");
        }
        let mut out = Vec::new();
        for node in nodes.into_iter().take(n) {
            // SAFETY: `node` came from `cuGraphGetNodes` on a live graph.
            let kind = unsafe {
                let mut k = core::mem::MaybeUninit::uninit();
                cuGraphNodeGetType(node, k.as_mut_ptr())
                    .result()
                    .expect("a graph node to report its type");
                k.assume_init()
            };
            if kind != CUgraphNodeType::CU_GRAPH_NODE_TYPE_MEMCPY {
                continue;
            }
            // SAFETY: the node's type was just checked to be MEMCPY.
            let p: CUDA_MEMCPY3D = unsafe {
                let mut p = core::mem::MaybeUninit::zeroed();
                cuGraphMemcpyNodeGetParams(node, p.as_mut_ptr())
                    .result()
                    .expect("a memcpy node to report its parameters");
                p.assume_init()
            };
            let src = core::cmp::max(p.srcDevice, p.srcHost as u64);
            let dst = core::cmp::max(p.dstDevice, p.dstHost as u64);
            let bytes = (p.WidthInBytes as u64)
                * (p.Height.max(1) as u64)
                * (p.Depth.max(1) as u64);
            out.push((src, dst, bytes, p.srcMemoryType as u32));
        }
        out.sort_unstable();
        out
    }

    fn graph_destroy(&mut self, id: u64) {
        if let Some(g) = self.graphs.remove(&id) {
            self.unsafe_set_current();
            unsafe {
                let _ = cudarc::driver::result::graph::exec_destroy(g.exec);
                let _ = cudarc::driver::result::graph::destroy(g.graph);
            }
        }
    }

    fn register_external_aliased(
        &mut self,
        ptr: *mut core::ffi::c_void,
        page_len: u64,
        offset: u64,
        size: u64,
        keepalive: alloc::sync::Arc<dyn core::any::Any + Send + Sync>,
        stream_id: StreamId,
    ) -> Handle {
        assert!(
            offset + size <= page_len,
            "external region [{offset}, {}) does not fit in the {page_len} bytes described",
            offset + size
        );
        assert!(
            crate::supports_zero_copy_host(self.device_id.index_id as usize),
            "this device cannot address host memory directly \
             (cudaDevAttrPageableMemoryAccess = 0); handing it a host pointer would \
             read garbage rather than fail, so the seam refuses instead. Use \
             ComputeClient::create and pay the copy."
        );
        let mut command = match self.command_no_inputs(
            stream_id,
            StreamErrorMode {
                ignore: true,
                flush: false,
            },
        ) {
            Ok(val) => val,
            Err(err) => unreachable!("{err:?}"),
        };
        let mm = &mut command.streams.current().memory_management_gpu;
        // SAFETY: the `ComputeClient::register_external_aliased` contract makes
        // the caller responsible for the region being live and immutable for
        // the life of every handle derived from it; `keepalive` carries the
        // owner so that outliving is enforced rather than merely promised.
        let storage_handle = unsafe {
            mm.storage()
                .register_external(ptr as cudarc::driver::sys::CUdeviceptr, offset, size, keepalive)
        };
        let mem = mm.register_external(storage_handle);
        Handle::from_memory(mem, stream_id, size)
    }

    fn write(&mut self, descriptors: Vec<(CopyDescriptor, Bytes)>, stream_id: StreamId) {
        let mut command = match self.command(
            stream_id,
            descriptors.iter().map(|desc| &desc.0.handle),
            StreamErrorMode {
                ignore: true,
                flush: false,
            },
        ) {
            Ok(val) => val,
            Err(err) => unreachable!("{err:?}"),
        };

        for (descriptor, data) in descriptors {
            if let Err(err) = command.write_to_gpu(descriptor, data) {
                command.error(err.into());
                return;
            }
        }
    }

    unsafe fn launch(
        &mut self,
        kernel: Self::Kernel,
        count: CubeCount,
        bindings: KernelArguments,
        mode: ExecutionMode,
        stream_id: StreamId,
    ) {
        if let Err(err) = self.launch_checked(kernel, count, bindings, mode, stream_id) {
            let mut stream = match self.streams.resolve(stream_id, [].into_iter(), false) {
                Ok(stream) => stream,
                Err(err) => unreachable!("{err:?}"),
            };
            stream.current().errors.push(err);
        }
    }

    fn flush(&mut self, stream_id: StreamId) -> Result<(), ServerError> {
        let mut command = self.command_no_inputs(
            stream_id,
            StreamErrorMode {
                ignore: false,
                flush: true,
            },
        )?;

        let current = command.streams.current();
        current.drop_queue.flush(|| Fence::new(current.sys));
        current.memory_management_gpu.storage().flush();

        Ok(())
    }

    fn sync(&mut self, stream_id: StreamId) -> DynFut<Result<(), ServerError>> {
        let command = self.command_no_inputs(
            stream_id,
            StreamErrorMode {
                ignore: false,
                flush: true,
            },
        );

        match command {
            Ok(mut command) => command.sync(),
            Err(err) => Box::pin(async { Err(err) }),
        }
    }

    fn start_profile(&mut self, stream_id: StreamId) -> Result<ProfilingToken, ServerError> {
        cubecl_common::future::block_on(self.sync(stream_id))?;
        Ok(self.ctx.timestamps.start())
    }

    fn end_profile(
        &mut self,
        stream_id: StreamId,
        token: ProfilingToken,
    ) -> Result<ProfileDuration, ProfileError> {
        if let Err(err) = cubecl_common::future::block_on(self.sync(stream_id)) {
            self.ctx
                .timestamps
                .error(ProfileError::Server(Box::new(err)));
        }
        self.ctx.timestamps.stop(token)
    }

    fn get_resource(
        &mut self,
        binding: Binding,
        stream_id: StreamId,
    ) -> Result<ManagedResource<GpuResource>, ServerError> {
        let mut command = self.command(
            stream_id,
            [&binding].into_iter(),
            StreamErrorMode {
                ignore: true,
                flush: false,
            },
        )?;
        let memory = binding.memory.clone();
        let resource = command.resource(binding)?;

        Ok(ManagedResource::new(memory, resource))
    }

    fn memory_usage(&mut self, stream_id: StreamId) -> Result<MemoryUsage, ServerError> {
        let mut command = self.command_no_inputs(
            stream_id,
            StreamErrorMode {
                ignore: false,
                flush: false,
            },
        )?;
        Ok(command.memory_usage())
    }

    fn memory_cleanup(&mut self, stream_id: StreamId) {
        let mut command = match self.command_no_inputs(
            stream_id,
            StreamErrorMode {
                ignore: true,
                flush: false,
            },
        ) {
            Ok(val) => val,
            Err(err) => unreachable!("{err:?}"),
        };
        command.memory_cleanup()
    }

    fn allocation_mode(&mut self, mode: MemoryAllocationMode, stream_id: StreamId) {
        let mut command = match self.command_no_inputs(
            stream_id,
            StreamErrorMode {
                ignore: true,
                flush: false,
            },
        ) {
            Ok(val) => val,
            Err(err) => unreachable!("{err:?}"),
        };
        command.allocation_mode(mode)
    }
}

impl ServerCommunication for CudaServer {
    const SERVER_COMM_ENABLED: bool = true;

    fn comm_init(&mut self, device_ids: Vec<DeviceId>) -> Result<(), ServerError> {
        let id = CommunicationId::from(device_ids.clone());
        if let Entry::Vacant(e) = self.communicators.entry(id.clone()) {
            let mut comm = MaybeUninit::uninit();
            let mut device_ids = device_ids.clone();
            device_ids.sort();
            // A group formed OUTSIDE this process wins, and takes all three
            // numbers with it. The single-process derivation below cannot serve
            // two nodes: each would mint its own id (so the rendezvous hangs
            // rather than fails) and each would derive rank 0 from its own local
            // device 0. See `communication::set_external_comm`.
            let (world, rank, nccl_comm_id) = match external_comm() {
                Some(ext) => (ext.world, ext.rank, ext.id),
                None => {
                    let rank = device_ids
                        .iter()
                        .position(|id| id.index_id == self.device_id.index_id)
                        .expect("Device's peer id should be in the list of device ids.");
                    (
                        device_ids.len() as i32,
                        rank as i32,
                        get_nccl_comm_id(device_ids.clone()),
                    )
                }
            };

            // SAFETY: `comm` is a valid `MaybeUninit`. `nccl_comm_id` is a unique communicator ID
            // shared across all participating ranks. `rank` is this device's position in the
            // group. `comm_init_rank` initializes the communicator, making `assume_init` valid.
            unsafe {
                cudarc::nccl::result::comm_init_rank(
                    comm.as_mut_ptr(),
                    world,
                    nccl_comm_id,
                    rank,
                )
                .map_err(|e| ServerError::Generic {
                    reason: format!("NCCL comm_init_rank failed: {e:?}"),
                    backtrace: BackTrace::capture(),
                })?;
                e.insert(comm.assume_init());
            }

            let mut initialized_comms = self.utilities.initialized_comms.write().unwrap();
            initialized_comms.insert(id);
        }

        Ok(())
    }

    fn all_reduce(
        &mut self,
        src: Binding,
        dst: Binding,
        dtype: ElemType,
        stream_id: StreamId,
        op: ReduceOperation,
        device_ids: Vec<DeviceId>,
    ) -> Result<(), ServerError> {
        // We create a command on the server to retrieve the correct resource of the source and the destination
        // from the memory pools.
        if src.stream != dst.stream {
            for stream in [src.stream, dst.stream].iter() {
                let mut command = self.command_no_inputs(
                    *stream,
                    StreamErrorMode {
                        ignore: false,
                        flush: false,
                    },
                )?;
                command.error(ServerError::Generic {
                    reason: "Source and destination should be on the same stream.".into(),
                    backtrace: BackTrace::capture(),
                });
            }
        }

        let mut command_src = self.command(
            stream_id,
            [&src, &dst].into_iter(),
            StreamErrorMode {
                ignore: false,
                flush: false,
            },
        )?;
        let resource_src = command_src.resource(src)?;
        let resource_dst = command_src.resource(dst)?;

        let stream = command_src.streams.current().sys;

        // We need to free the command before accessing communicators.
        core::mem::drop(command_src);

        // Wait for data to be ready on compute stream.
        Fence::new(stream).wait_async(self.comm_stream);

        // Get the communicator.
        let comm = self
            .communicators
            .get(&CommunicationId::from(device_ids))
            .expect("Communicator for this ID should be initialized");

        // Perform the `cudarc::nccl::result::all_reduce` operation.
        let (nccl_dtype, count) = get_nccl_dtype_count(dtype, resource_src.size);
        // SAFETY: `resource_src.ptr` and `resource_dst.ptr` are valid device pointers.
        // `comm` is a valid NCCL communicator initialized via `comm_init_rank`.
        // `self.comm_stream` is a valid CUDA stream dedicated to collective operations.

        unsafe {
            cudarc::nccl::result::all_reduce(
                resource_src.ptr as *const _,
                resource_dst.ptr as *mut _,
                count,
                nccl_dtype,
                to_nccl_op(op),
                *comm,
                self.comm_stream as _,
            )
            .map_err(|e| ServerError::Generic {
                reason: format!("NCCL all_reduce failed: {e:?}"),
                backtrace: BackTrace::capture(),
            })?;
        }

        Ok(())
    }

    fn sync_collective(&mut self, stream_id: StreamId) -> Result<(), ServerError> {
        let mut command = self.command_no_inputs(
            stream_id,
            StreamErrorMode {
                ignore: true,
                flush: false,
            },
        )?;
        let stream = command.streams.current().sys;

        drop(command);

        Fence::new(self.comm_stream).wait_async(stream);

        Ok(())
    }

    #[cfg_attr(feature = "tracing", tracing::instrument(level = "trace", skip(desc)))]
    fn send(
        &mut self,
        desc: CopyDescriptor,
        dtype: ElemType,
        stream_id: StreamId,
        device_id_dst: DeviceId,
    ) -> Result<(), ServerError> {
        let binding = desc.handle.clone();

        // We create a command on the source server to retrieve the correct resource from the
        // source memory pools. We also make sure the current stream is aligned with the stream of
        // the binding, where the data was first allocated.
        let mut command = self.command(
            stream_id,
            [&desc.handle].into_iter(),
            StreamErrorMode {
                ignore: true,
                flush: false,
            },
        )?;
        let resource = command.resource(binding.clone())?;
        let stream = command.streams.current().sys;

        // We need to free the command before creating another one.
        core::mem::drop(command);

        // Wait for data to be ready on compute stream.
        Fence::new(stream).wait_async(self.comm_stream);

        // Get the communicator.
        let mut device_ids = vec![device_id_dst, self.device_id];
        device_ids.sort();
        let comm_id = CommunicationId::from(device_ids.clone());
        let comm = self
            .communicators
            .get(&comm_id)
            .expect("Communicator for this ID should exist");

        let rank_dst = device_ids
            .iter()
            .position(|id| id.index_id != self.device_id.index_id)
            .unwrap() as i32;

        // Perform the `send` operation.
        let (nccl_dtype, count) = get_nccl_dtype_count(dtype, resource.size);
        // SAFETY: `resource.ptr` is a valid device pointer.
        // `comm` is a valid NCCL communicator initialized via `comm_init_rank`.
        // `self.comm_stream` is a valid CUDA stream dedicated to collective operations.
        unsafe {
            cudarc::nccl::result::send(
                resource.ptr as *const _,
                count,
                nccl_dtype,
                rank_dst,
                *comm,
                self.comm_stream as _,
            )
            .map_err(|e| ServerError::Generic {
                reason: format!("NCCL send failed: {e:?}"),
                backtrace: BackTrace::capture(),
            })?;
        }

        Ok(())
    }

    #[cfg_attr(feature = "tracing", tracing::instrument(level = "trace"))]
    fn recv(
        &mut self,
        handle: Handle,
        dtype: ElemType,
        stream_id: StreamId,
        device_id_src: DeviceId,
    ) -> Result<(), ServerError> {
        // We create a new command on the destination server to reserve the necessary GPU memory.
        let mut command_dst = self.command_no_inputs(
            stream_id,
            StreamErrorMode {
                ignore: true,
                flush: false,
            },
        )?;

        let memory = command_dst.reserve(handle.size()).unwrap();
        command_dst.bind(memory, handle.memory.clone());

        let resource_dst = command_dst.resource(handle.binding())?;

        core::mem::drop(command_dst);

        // Get the communicator.
        let mut device_ids = vec![device_id_src, self.device_id];
        device_ids.sort();
        let comm_id = CommunicationId::from(device_ids.clone());
        let comm = self
            .communicators
            .get(&comm_id)
            .expect("Communicator for this ID should exist");

        let rank_src = device_ids
            .iter()
            .position(|id| id.index_id != self.device_id.index_id)
            .unwrap() as i32;

        // Perform the `recv` operation.
        let (nccl_dtype, count) = get_nccl_dtype_count(dtype, resource_dst.size);
        // SAFETY: `resource.ptr` is a valid device pointer.
        // `comm` is a valid NCCL communicator initialized via `comm_init_rank`.
        // `self.comm_stream` is a valid CUDA stream dedicated to collective operations.
        unsafe {
            cudarc::nccl::result::recv(
                resource_dst.ptr as *mut _,
                count,
                nccl_dtype,
                rank_src,
                *comm,
                self.comm_stream as _,
            )
            .map_err(|e| ServerError::Generic {
                reason: format!("NCCL recv failed: {e:?}"),
                backtrace: BackTrace::capture(),
            })?;
        }

        Ok(())
    }
}

impl CudaServer {
    /// Create a new cuda server.
    pub(crate) fn new(
        ctx: CudaContext,
        mem_props: MemoryDeviceProperties,
        mem_config: MemoryConfiguration,
        mem_alignment: usize,
        device_id: DeviceId,
        utilities: ServerUtilities<Self>,
    ) -> Self {
        let config = CubeClRuntimeConfig::get();
        let max_streams = config.streaming.max_streams;

        ctx.unsafe_set_current().unwrap();

        let comm_stream = cudarc::driver::result::stream::create(
            cudarc::driver::result::stream::StreamKind::NonBlocking,
        )
        .expect("Can create a new stream.");

        Self {
            ctx,
            device_id,
            streams: MultiStream::new(
                utilities.logger.clone(),
                CudaStreamBackend::new(
                    mem_props,
                    mem_config,
                    mem_alignment,
                    utilities.logger.clone(),
                ),
                max_streams,
            ),
            utilities: Arc::new(utilities),
            comm_stream,
            communicators: HashMap::default(),
            graphs: HashMap::default(),
            next_graph_id: 1,
            capture_launches: Vec::new(),
            capture_hold: Vec::new(),
            capture_arena_mark: 0,
            last_arena_signature: None,
        }
    }

    /// The raw CUDA stream this `stream_id` resolves to. Capture is a property
    /// of one stream, so begin/end/replay must all name the same one.
    ///
    /// `set_capturing` also marks the stream, which is what stops the launch
    /// path's drop-queue flush from host-blocking inside the region. Measured:
    /// without the mark, a capture dies at exactly launch 63 of a chain,
    /// because the queue's policy fires at 64 staged `Bytes` and its flush
    /// waits on a fence.
    fn raw_stream(
        &mut self,
        stream_id: StreamId,
        set_capturing: Option<bool>,
    ) -> cudarc::driver::sys::CUstream {
        let mut command = self
            .command_no_inputs(
                stream_id,
                StreamErrorMode {
                    ignore: true,
                    flush: false,
                },
            )
            .expect("a stream to capture on");
        let current = command.streams.current();
        if let Some(v) = set_capturing {
            current.capturing = v;
        }
        let stream = current.sys;
        core::mem::drop(command);
        stream
    }

    fn command_no_inputs(
        &mut self,
        stream_id: StreamId,
        mode: StreamErrorMode,
    ) -> Result<Command<'_>, ServerError> {
        self.command(stream_id, [].into_iter(), mode)
    }

    fn unsafe_set_current(&self) {
        // TODO: Should check if on the same thread before calling it, since now we don't switch
        // thread except for device memory transfer.
        self.ctx.unsafe_set_current().unwrap();
    }

    /// A command on `stream_id` for touching the allocator only: no inputs to
    /// resolve, no flush (a flush waits on a fence, which a capture cannot
    /// contain), and errors ignored because there is no work being enqueued.
    fn arena_command(&mut self, stream_id: StreamId) -> Command<'_> {
        match self.command_no_inputs(
            stream_id,
            StreamErrorMode {
                ignore: true,
                flush: false,
            },
        ) {
            Ok(val) => val,
            Err(err) => unreachable!("{err:?}"),
        }
    }

    fn command<'a>(
        &mut self,
        stream_id: StreamId,
        handles: impl Iterator<Item = &'a Binding>,
        mode: StreamErrorMode,
    ) -> Result<Command<'_>, ServerError> {
        self.unsafe_set_current();

        if mode.flush {
            let errors = self.flush_errors(stream_id);

            if !mode.ignore && !errors.is_empty() {
                return Err(ServerError::ServerUnhealthy {
                    errors,
                    backtrace: BackTrace::capture(),
                });
            }
        }

        let streams = self.streams.resolve(stream_id, handles, !mode.ignore)?;
        Ok(Command::new(&mut self.ctx, streams))
    }

    fn flush_errors(&mut self, stream_id: StreamId) -> Vec<ServerError> {
        let mut stream = match self.streams.resolve(stream_id, [].into_iter(), false) {
            Ok(stream) => stream,
            Err(_) => return Vec::new(),
        };
        let errors = core::mem::take(&mut stream.current().errors);

        // It is very important to tag current profiles as being wrong.
        if !errors.is_empty() {
            self.ctx.timestamps.error(ProfileError::Unknown {
                reason: alloc::format!("{errors:?}"),
                backtrace: BackTrace::capture(),
            });
            stream.current().memory_management_gpu.cleanup(false);
        }

        core::mem::drop(stream);
        errors
    }

    fn launch_checked(
        &mut self,
        kernel: Box<dyn CubeTask<CudaCompiler>>,
        count: CubeCount,
        bindings: KernelArguments,
        mode: ExecutionMode,
        stream_id: StreamId,
    ) -> Result<(), ServerError> {
        let mut kernel_id = kernel.id();
        let logger = self.streams.logger.clone();
        kernel_id.mode(mode);
        let grid_constants = self
            .ctx
            .compilation_options
            .supports_features
            .grid_constants;
        let mut command = self.command(
            stream_id,
            bindings.buffers.iter(),
            StreamErrorMode {
                ignore: true,
                flush: false,
            },
        )?;

        let count = match count {
            CubeCount::Static(x, y, z) => (x, y, z),
            // TODO: CUDA doesn't have an exact equivalent of dynamic dispatch. Instead, kernels are free to launch other kernels.
            // One option is to create a dummy kernel with 1 thread that launches the real kernel with the dynamic dispatch settings.
            // For now, just read the dispatch settings from the buffer.
            CubeCount::Dynamic(binding) => {
                let data = future::block_on(command.read_async(vec![CopyDescriptor::new(
                    binding,
                    [3].into(),
                    [1].into(),
                    4,
                )]))?;
                let data = bytemuck::cast_slice(&data[0]);
                assert!(
                    data.len() == 3,
                    "Dynamic cube count should contain 3 values"
                );
                (data[0], data[1], data[2])
            }
        };

        let capture_open = crate::compute::CAPTURE_OPEN.load(core::sync::atomic::Ordering::Relaxed);
        // WHICH pinned buffer this launch's dynamic metadata was staged
        // through, counted rather than inferred. `write_to_gpu` pushes it onto
        // `capture_staging` while a capture is open -- but only when the
        // capture owns its staging, and `CUBECL_GRAPH_STAGE_OWN=0` is an arm
        // where it does not. So the index is the difference between two
        // lengths, which is right under either arm, instead of "the last one",
        // which would name somebody else's buffer under one of them.
        let staged_before = match capture_open {
            true => command.streams.current().capture_staging.len(),
            false => 0,
        };
        let (info_const, info_binding) = if grid_constants {
            let info = &bindings.info;

            let mut handle = Option::None;
            if info.dynamic_metadata_offset < info.data.len() {
                let dyn_meta = &bytemuck::cast_slice(&info.data[info.dynamic_metadata_offset..]);
                handle = Some(command.create_with_data(dyn_meta)?);
            }

            (Some(info.data.as_ptr() as *mut c_void), handle)
        } else {
            let mut handle = Option::None;
            if !bindings.info.data.is_empty() {
                handle = Some(command.create_with_data(bytemuck::cast_slice(&bindings.info.data))?);
            }
            (None, handle)
        };
        let staging_idx = match capture_open {
            true => {
                let now = command.streams.current().capture_staging.len();
                (now > staged_before).then(|| now - 1)
            }
            false => None,
        };

        // While a capture is open every buffer this launch binds becomes a
        // POINTER inside a graph node and stays one for the graph's whole life,
        // so none of them may go back to the allocator. Collected into a local
        // because `command` holds the server borrow until it is dropped.
        let mut hold_now: Vec<Binding> = Vec::new();
        // The packed scalar+metadata blob, copied while it is still the
        // caller's. A node holds the BYTES, so a later rewrite has to be able
        // to reproduce them, and by then this vector is gone.
        let info_snapshot = match capture_open {
            true => Some(bindings.info.data.clone()),
            false => None,
        };
        let dyn_offset = bindings.info.dynamic_metadata_offset;
        let capture_name = match capture_open {
            true => alloc::format!("{:?}", kernel_id),
            false => alloc::string::String::new(),
        };
        let capturing = capture_open && capture_hold_enabled();
        if capturing {
            // WHICH buffers this launch owes a hold, and why it is not all of
            // them once there is an arena.
            //
            // A hold does two different jobs at once, and the arena splits
            // them. It keeps a slice ALLOCATED, and it keeps the slice from
            // being handed to anyone else. For a buffer born inside the
            // captured region the arena already does both -- it never returns
            // a page to the storage and it is only ever allocated from while a
            // capture is open -- and holding one on top of that does active
            // harm: it stops the arena RECYCLING the slice inside the region,
            // so every intra-region allocation has to be a fresh one, and a
            // fresh allocation made while the stream is capturing is a graph
            // memory node rather than a pointer.
            //
            // A buffer that was already live when the region opened is a
            // different case and still owes a hold. It belongs to the ordinary
            // pools; if it dies inside the region its slice goes back there,
            // and once the capture closes the pool is free to hand it to
            // anything, while this graph still reads it on every replay.
            let arena = capture_arena_enabled();
            let mut owe = |command: &mut Command<'_>, b: &Binding| {
                if !arena || !command.arena_owns(b) {
                    hold_now.push(b.clone());
                }
            };
            for it in bindings.tensor_maps.iter() {
                owe(&mut command, &it.binding);
            }
            for b in bindings.buffers.iter() {
                owe(&mut command, b);
            }
            // The metadata buffer `create_with_data` just produced is bound
            // below and dies at the end of this call, which makes it the FIRST
            // slice the allocator offers the next launch of the same region --
            // and with an arena that is exactly right, so it is exactly what
            // must NOT be held.
            if let Some(h) = info_binding.as_ref() {
                let b = h.clone().binding();
                owe(&mut command, &b);
            }
        }

        let mut resources = bindings
            .tensor_maps
            .iter()
            .map(|it| it.binding.clone())
            .chain(bindings.buffers)
            .map(|binding| command.resource(binding).expect("Resource to exist."))
            .collect::<Vec<_>>();

        let mut tensor_maps = Vec::with_capacity(bindings.tensor_maps.len());

        for TensorMapBinding { map, binding } in bindings.tensor_maps.into_iter() {
            let resource = command
                .resource(binding)
                .expect("Tensor map resource exists.");
            let device_ptr = resource.ptr as *mut c_void;

            let mut map_ptr = MaybeUninit::zeroed();

            let shape: Vec<_> = map
                .metadata
                .shape()
                .iter()
                .rev()
                .map(|s| *s as u64)
                .collect();
            let strides: Vec<_> = map
                .metadata
                .strides()
                .iter()
                .rev()
                .skip(1)
                .map(|s| *s as u64 * map.storage_ty.size() as u64)
                .collect();
            let elem_stride: Vec<_> = map.elem_stride.iter().rev().map(|s| *s as u32).collect();

            match &map.format {
                // SAFETY: `map_ptr` is a zeroed `MaybeUninit<CUtensorMap>`. `device_ptr` is a
                // valid device pointer. Shape, strides, tile_size, and elem_stride vectors
                // are constructed from validated metadata and outlive this call.
                TensorMapFormat::Tiled(TiledArgs { tile_size }) => unsafe {
                    let tile_size: Vec<_> =
                        tile_size.iter().rev().copied().map(|s| s as u32).collect();

                    cuTensorMapEncodeTiled(
                        map_ptr.as_mut_ptr(),
                        elem_to_tensor_map_type(map.storage_ty),
                        map.metadata.rank() as u32,
                        device_ptr,
                        shape.as_ptr(),
                        strides.as_ptr(),
                        tile_size.as_ptr(),
                        elem_stride.as_ptr(),
                        interleave_to_cuda(map.interleave),
                        swizzle_to_cuda(map.swizzle),
                        prefetch_to_cuda(map.prefetch),
                        oob_to_cuda(map.oob_fill),
                    )
                    .result()
                    .map_err(|err| {
                        let generic_err =
                            check_tma_generic(&map, device_ptr, &shape, &strides, &elem_stride)
                                .err();
                        let tiled_err = check_tma_tiled(&map, &tile_size).err();
                        generic_err
                            .or(tiled_err)
                            .unwrap_or_else(|| LaunchError::Unknown {
                                reason: format!("{err}"),
                                backtrace: BackTrace::capture(),
                            })
                    })?;
                },
                // SAFETY: Same invariants as `Tiled` above. Additionally, `lower_corner` and
                // `upper_corner` are valid pixel box bounds derived from the tensor map args.
                TensorMapFormat::Im2col(args) => unsafe {
                    let lower_corner: Vec<_> =
                        args.pixel_box_lower_corner.iter().rev().copied().collect();
                    let upper_corner: Vec<_> =
                        args.pixel_box_upper_corner.iter().rev().copied().collect();

                    cuTensorMapEncodeIm2col(
                        map_ptr.as_mut_ptr(),
                        elem_to_tensor_map_type(map.storage_ty),
                        map.metadata.rank() as u32,
                        device_ptr,
                        shape.as_ptr(),
                        strides.as_ptr(),
                        lower_corner.as_ptr(),
                        upper_corner.as_ptr(),
                        args.channels_per_pixel,
                        args.pixels_per_column,
                        elem_stride.as_ptr(),
                        interleave_to_cuda(map.interleave),
                        swizzle_to_cuda(map.swizzle),
                        prefetch_to_cuda(map.prefetch),
                        oob_to_cuda(map.oob_fill),
                    )
                    .result()
                    .map_err(|err| {
                        let generic_err =
                            check_tma_generic(&map, device_ptr, &shape, &strides, &elem_stride)
                                .err();
                        let tiled_err = check_tma_im2col(
                            &map,
                            &lower_corner,
                            &upper_corner,
                            args.channels_per_pixel,
                            args.pixels_per_column,
                        )
                        .err();
                        generic_err
                            .or(tiled_err)
                            .unwrap_or_else(|| LaunchError::Unknown {
                                reason: format!("{err}"),
                                backtrace: BackTrace::capture(),
                            })
                    })?;
                },
                // SAFETY: Same invariants as `Im2col` above. Requires CUDA 12.8+.
                #[cfg(cuda_12080)]
                TensorMapFormat::Im2colWide(args) => unsafe {
                    use cudarc::driver::sys::{
                        CUtensorMapIm2ColWideMode, cuTensorMapEncodeIm2colWide,
                    };
                    cuTensorMapEncodeIm2colWide(
                        map_ptr.as_mut_ptr(),
                        elem_to_tensor_map_type(map.storage_ty),
                        map.metadata.rank() as u32,
                        device_ptr,
                        shape.as_ptr(),
                        strides.as_ptr(),
                        args.pixel_box_lower_corner_width,
                        args.pixel_box_upper_corner_width,
                        args.channels_per_pixel,
                        args.pixels_per_column,
                        elem_stride.as_ptr(),
                        interleave_to_cuda(map.interleave),
                        CUtensorMapIm2ColWideMode::CU_TENSOR_MAP_IM2COL_WIDE_MODE_W,
                        swizzle_to_cuda(map.swizzle),
                        prefetch_to_cuda(map.prefetch),
                        oob_to_cuda(map.oob_fill),
                    )
                    .result()
                    .map_err(|err| {
                        let generic_err =
                            check_tma_generic(&map, device_ptr, &shape, &strides, &elem_stride)
                                .err();
                        generic_err.unwrap_or_else(|| LaunchError::Unknown {
                            reason: format!("{err}"),
                            backtrace: BackTrace::capture(),
                        })
                    })?;
                },
                #[cfg(not(cuda_12080))]
                TensorMapFormat::Im2colWide(_) => {
                    return Err(LaunchError::Unknown {
                        reason: "CUDA version 12.8 required for tensor map format Im2colWide"
                            .into(),
                        backtrace: BackTrace::capture(),
                    }
                    .into());
                }
            };
            // SAFETY: `map_ptr` was fully initialized by one of the `cuTensorMapEncode*`
            // calls above, which all succeeded (errors are propagated before reaching here).
            let binding = unsafe { map_ptr.assume_init() };
            tensor_maps.push(binding);
        }

        resources.extend(
            info_binding
                .into_iter()
                .map(|s| command.resource(s.binding()).expect("Resource to exist")),
        );

        let captured = command.kernel(
            kernel_id,
            kernel,
            mode,
            count,
            &tensor_maps,
            &resources,
            info_const,
            logger,
        )?;
        core::mem::drop(command);
        self.capture_hold.append(&mut hold_now);
        if let Some(node) = captured {
            self.capture_launches.push(CapturedLaunch {
                node: node.node,
                func: node.func,
                grid: count,
                block: node.block,
                shared: node.shared,
                tensor_maps: tensor_maps.clone(),
                ptrs: resources.iter().map(|r| r.ptr).collect(),
                info: info_snapshot.unwrap_or_default(),
                info_is_grid_constant: grid_constants,
                dyn_offset,
                staging: staging_idx,
                name: capture_name,
            });
        }

        Ok(())
    }

    /// The graph named by `id`, or a panic that says which id was asked for.
    fn captured(&self, id: u64) -> &CapturedGraph {
        self.graphs
            .get(&id)
            .unwrap_or_else(|| panic!("no captured graph with id {id}"))
    }

    pub(crate) fn utilities(&self) -> Arc<ServerUtilities<Self>> {
        self.utilities.clone()
    }
}

fn elem_to_tensor_map_type(ty: StorageType) -> CUtensorMapDataType {
    use cudarc::driver::sys::CUtensorMapDataType::*;
    match ty {
        // packed fp4 should be treated as single 4-bit values to simplify indexing/shape handling
        // So a tile of width 16 with fp4 elements is 8 x fp4x2 elements wide.
        #[cfg(cuda_12080)]
        StorageType::Packed(ty, 2) if ty.size_bits() == 4 => CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B,
        StorageType::Scalar(ElemType::Float(kind)) => match kind {
            // There's no special handling for FP8, so load as u8. `0u8 == 0.0` when reinterpreting.
            FloatKind::E2M1 // single fp4s are padded to a full byte
            | FloatKind::E4M3
            | FloatKind::E5M2
            | FloatKind::UE8M0
            | FloatKind::E2M3
            | FloatKind::E3M2 => CU_TENSOR_MAP_DATA_TYPE_UINT8,
            FloatKind::F16 => CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
            FloatKind::BF16 => CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            FloatKind::Flex32 | FloatKind::F32 => CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
            FloatKind::TF32 => CU_TENSOR_MAP_DATA_TYPE_TFLOAT32,
            FloatKind::F64 => CU_TENSOR_MAP_DATA_TYPE_FLOAT64,
        },
        StorageType::Scalar(ElemType::Int(kind)) => match kind {
            // UInt is fine because zero bits and size is the same between both
            IntKind::I8 => CU_TENSOR_MAP_DATA_TYPE_UINT8,
            IntKind::I16 => CU_TENSOR_MAP_DATA_TYPE_UINT16,
            IntKind::I32 => CU_TENSOR_MAP_DATA_TYPE_INT32,
            IntKind::I64 => CU_TENSOR_MAP_DATA_TYPE_INT64,
        },
        StorageType::Scalar(ElemType::UInt(kind)) => match kind {
            UIntKind::U8 => CU_TENSOR_MAP_DATA_TYPE_UINT8,
            UIntKind::U16 => CU_TENSOR_MAP_DATA_TYPE_UINT16,
            UIntKind::U32 => CU_TENSOR_MAP_DATA_TYPE_UINT32,
            UIntKind::U64 => CU_TENSOR_MAP_DATA_TYPE_UINT64,
        },
        _ => unimplemented!("Not supported for tensor map type"),
    }
}

fn interleave_to_cuda(interleave: TensorMapInterleave) -> CUtensorMapInterleave {
    use cudarc::driver::sys::CUtensorMapInterleave::*;
    match interleave {
        TensorMapInterleave::None => CU_TENSOR_MAP_INTERLEAVE_NONE,
        TensorMapInterleave::B16 => CU_TENSOR_MAP_INTERLEAVE_16B,
        TensorMapInterleave::B32 => CU_TENSOR_MAP_INTERLEAVE_32B,
    }
}

fn swizzle_to_cuda(swizzle: TensorMapSwizzle) -> CUtensorMapSwizzle {
    use cudarc::driver::sys::CUtensorMapSwizzle::*;
    match swizzle {
        TensorMapSwizzle::None => CU_TENSOR_MAP_SWIZZLE_NONE,
        TensorMapSwizzle::B32 => CU_TENSOR_MAP_SWIZZLE_32B,
        TensorMapSwizzle::B64 => CU_TENSOR_MAP_SWIZZLE_64B,
        TensorMapSwizzle::B128 => CU_TENSOR_MAP_SWIZZLE_128B,
        #[cfg(cuda_12080)]
        TensorMapSwizzle::B128Atom32B => CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B,
        #[cfg(cuda_12080)]
        TensorMapSwizzle::B128Atom32BFlip8B => CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B_FLIP_8B,
        #[cfg(cuda_12080)]
        TensorMapSwizzle::B128Atom64B => CU_TENSOR_MAP_SWIZZLE_128B_ATOM_64B,
        #[cfg(not(cuda_12080))]
        _ => unimplemented!("Swizzle atomicity requires CUDA 12.8 or higher"),
    }
}

fn prefetch_to_cuda(prefetch: TensorMapPrefetch) -> CUtensorMapL2promotion {
    use cudarc::driver::sys::CUtensorMapL2promotion::*;
    match prefetch {
        TensorMapPrefetch::None => CU_TENSOR_MAP_L2_PROMOTION_NONE,
        TensorMapPrefetch::B64 => CU_TENSOR_MAP_L2_PROMOTION_L2_64B,
        TensorMapPrefetch::B128 => CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        TensorMapPrefetch::B256 => CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
    }
}

fn oob_to_cuda(fill: OobFill) -> CUtensorMapFloatOOBfill {
    use cudarc::driver::sys::CUtensorMapFloatOOBfill::*;
    match fill {
        OobFill::Zero => CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
        OobFill::NaN => CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA,
    }
}

macro_rules! launch_check {
    ($assertion: expr, $($arg:tt)+) => {
        if $assertion {
            Ok(())
        } else {
            Err(LaunchError::Unknown {
                reason: format!($($arg)*),
                backtrace: BackTrace::capture(),
            })
        }
    };
}

fn check_tma_generic(
    map: &TensorMapMeta,
    device_ptr: *mut c_void,
    shape: &[u64],
    strides: &[u64],
    elem_strides: &[u32],
) -> Result<(), LaunchError> {
    // globalAddress invariants
    launch_check!(
        (device_ptr as usize).is_multiple_of(16),
        "Tensor pointer must be 16 byte aligned"
    )?;
    if !matches!(map.interleave, TensorMapInterleave::None) {
        launch_check!(
            (device_ptr as usize).is_multiple_of(32),
            "Tensor pointer must be 32 byte aligned"
        )?;
    }

    // tensorRank invariants
    launch_check!(
        (1..=5).contains(&map.metadata.rank()),
        "Rank must be between 1 and 5"
    )?;
    launch_check!(
        matches!(map.interleave, TensorMapInterleave::None) || map.metadata.rank() >= 3,
        "When interleave is enabled, rank must be >= 3"
    )?;

    // globalDim invariants
    launch_check!(
        shape.iter().all(|it| *it <= u32::MAX as u64),
        "Shape must be <= u32::MAX"
    )?;
    #[cfg(cuda_12080)]
    if matches!(map.storage_ty, StorageType::Packed(ty, 2) if ty.size_bits() == 4) {
        launch_check!(
            shape[0].is_multiple_of(2),
            "Packed tensor map must have multiple of 2 for the innermost dimension"
        )?;
    }

    // globalStrides invariants
    launch_check!(
        strides.iter().all(|it| it.is_multiple_of(16)),
        "Strides must be 16 byte aligned"
    )?;
    if matches!(map.interleave, TensorMapInterleave::B32) {
        launch_check!(
            strides.iter().all(|it| it.is_multiple_of(32)),
            "Strides must be 32 byte aligned when interleave is B32"
        )?;
    }

    // elementStrides invariants
    launch_check!(
        elem_strides.iter().all(|it| *it > 0 && *it <= 8),
        "Element strides must be non-zero and <= 8"
    )?;
    if matches!(map.interleave, TensorMapInterleave::None) {
        launch_check!(
            elem_strides[0] == 1,
            "Innermost element stride is ignored without interleaving"
        )?;
    }

    // oobFill invariants
    if matches!(map.oob_fill, OobFill::NaN) {
        launch_check!(
            map.storage_ty.is_float(),
            "NaN fill is only supported for float types"
        )?;
    }

    Ok(())
}

fn check_tma_tiled(map: &TensorMapMeta, tile_size: &[u32]) -> Result<(), LaunchError> {
    launch_check!(
        tile_size.len() == map.metadata.rank(),
        "Tile shape should match rank"
    )?;
    launch_check!(
        tile_size.iter().all(|it| *it > 0 && *it <= 256),
        "Tile shape must be non-zero and <= 256"
    )?;
    let tile_size_0_bytes = tile_size[0] as usize * map.storage_ty.size();
    if matches!(map.interleave, TensorMapInterleave::None) {
        let max_tile_bytes = match map.swizzle {
            TensorMapSwizzle::None => usize::MAX,
            TensorMapSwizzle::B32 => 32,
            TensorMapSwizzle::B64 => 64,
            TensorMapSwizzle::B128
            | TensorMapSwizzle::B128Atom32B
            | TensorMapSwizzle::B128Atom32BFlip8B
            | TensorMapSwizzle::B128Atom64B => 128,
        };
        launch_check!(
            tile_size_0_bytes <= max_tile_bytes,
            "Innermost tile dim must be <= swizzle size"
        )?;
    }
    if matches!(map.interleave, TensorMapInterleave::B32) {
        launch_check!(
            map.swizzle == TensorMapSwizzle::B32,
            "If interleave is B32, swizzle must be B32"
        )?;
    }

    Ok(())
}

fn check_tma_im2col(
    map: &TensorMapMeta,
    lower_corner: &[i32],
    upper_corner: &[i32],
    channels_per_pixel: u32,
    pixels_per_column: u32,
) -> Result<(), LaunchError> {
    launch_check!(
        lower_corner.len() == map.metadata.rank() - 2,
        "Lower corner must be rank - 2 elements"
    )?;
    launch_check!(
        upper_corner.len() == map.metadata.rank() - 2,
        "Upper corner must be rank - 2 elements"
    )?;

    launch_check!(
        map.metadata.rank() >= 3 && map.metadata.rank() <= 5,
        "im2col requires rank to be between 3 and 5"
    )?;

    let (range_lower, range_upper) = match map.metadata.rank() {
        3 => (-32768, 32767),
        4 => (-128, 127),
        5 => (-16, 15),
        _ => unreachable!(),
    };
    launch_check!(
        lower_corner
            .iter()
            .all(|it| *it >= range_lower && *it <= range_upper),
        "Lower corner must be in range [{range_lower}, {range_upper}] for {}D im2col",
        map.metadata.rank()
    )?;
    launch_check!(
        upper_corner
            .iter()
            .all(|it| *it >= range_lower && *it <= range_upper),
        "Upper corner must be in range [{range_lower}, {range_upper}] for {}D im2col",
        map.metadata.rank()
    )?;

    launch_check!(
        channels_per_pixel <= 256,
        "Channels per pixel must be <= 256"
    )?;
    launch_check!(
        pixels_per_column <= 1024,
        "Pixels per column must be <= 1024"
    )?;

    Ok(())
}
