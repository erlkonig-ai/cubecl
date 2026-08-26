use crate::{
    CudaCompiler,
    compute::{
        MB, context::CudaContext, io::controller::PinnedMemoryManagedAllocController,
        storage::gpu::GpuResource, stream::CudaStreamBackend, sync::Fence,
    },
};
use cubecl_common::{
    backtrace::BackTrace,
    bytes::{AllocationProperty, Bytes},
    stream_id::StreamId,
};
#[cfg(debug_assertions)]
use cubecl_core::zspace::striding::try_check_pitched_row_major_strides;
use cubecl_core::{
    MemoryUsage,
    future::DynFut,
    server::{
        Binding, CopyDescriptor, ExecutionMode, Handle, IoError, LaunchError, ProfileError,
        ServerError,
    },
    zspace::{Shape, Strides, striding::has_pitched_row_major_strides},
};
use cubecl_runtime::{
    compiler::CubeTask,
    id::KernelId,
    logging::ServerLogger,
    memory_management::{ManagedMemoryHandle, MemoryAllocationMode, MemoryHandle},
    stream::ResolvedStreams,
};
use cudarc::driver::sys::{
    CUDA_MEMCPY2D_st, CUmemorytype, CUstream_st, CUtensorMap, cuMemcpy2DAsync_v2,
};
use std::{ffi::c_void, ops::DerefMut, sync::Arc};

/// The host-to-device write sizes worth copying into pinned memory first.
///
/// Below the floor the extra host copy costs more than the pageable driver
/// path, and the pinned pool would be churned by scalars and index vectors.
/// Above the ceiling the pool would pin hundreds of megabytes permanently —
/// an unembedding table is 3.3 GB — and pinned pages cannot be reclaimed,
/// which on a box whose checkpoint already exceeds its RAM comes straight out
/// of the page cache the weights are read through.
const PINNED_STAGING_MIN: usize = 256 * 1024;
const PINNED_STAGING_MAX: usize = 128 * 1024 * 1024;

/// Whether to use it at all. Default OFF: on GB10 the pinned pool measured
/// SLOWER than the pageable path it replaces (see `inkling_expert_probe`), and
/// a switch keeps both lanes runnable from one binary instead of one build.
/// Whether to print every launch's bound device addresses. Read once: a
/// `getenv` per kernel launch would be a cost on the path this exists to study.
fn trace_ptrs() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("CUBECL_TRACE_PTRS").as_deref() == Ok("1"))
}

fn pinned_staging_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        matches!(std::env::var("CUBECL_PIN_STAGING").as_deref(), Ok("1") | Ok("on"))
    })
}

#[derive(new)]
/// The `Command` struct encapsulates a CUDA context and a set of resolved CUDA streams, providing an
/// interface for executing GPU-related operations such as memory allocation, data transfers, kernel
/// registration, and task execution.
pub struct Command<'a> {
    ctx: &'a mut CudaContext,
    pub(crate) streams: ResolvedStreams<'a, CudaStreamBackend>,
}

impl<'a> Command<'a> {
    /// Retrieves a GPU resource associated with the provided binding.
    ///
    /// # Parameters
    ///
    /// * `binding` - The binding specifying the stream, memory, and offsets for the resource.
    ///
    /// # Returns
    ///
    /// * `Ok(GpuResource)` - The GPU resource associated with the binding.
    /// * `Err(IoError::InvalidHandle)` - If the binding does not correspond to a valid resource.
    pub fn resource(&mut self, binding: Binding) -> Result<GpuResource, IoError> {
        self.streams
            .get(&binding.stream)
            .memory_management_gpu
            .get_resource(binding.memory, binding.offset_start, binding.offset_end)
    }

    /// Get the stream cursor.
    pub fn cursor(&self) -> u64 {
        self.streams.cursor
    }

    /// Retrieves the gpu memory usage of the current stream.
    ///
    /// # Returns
    ///
    /// * The [`MemoryUsage`] struct.
    pub fn memory_usage(&mut self) -> MemoryUsage {
        self.streams.current().memory_management_gpu.memory_usage()
    }

    /// Explicitly cleanup gpu memory on the current stream.
    pub fn memory_cleanup(&mut self) {
        self.streams.current().memory_management_gpu.cleanup(true)
    }

    /// Set the [`MemoryAllocationMode`] for the current stream.
    ///
    /// # Parameters
    ///
    /// * `mode` - The allocation mode to be used.
    pub fn allocation_mode(&mut self, mode: MemoryAllocationMode) {
        self.streams.current().memory_management_gpu.mode(mode)
    }

    /// Allocates a new GPU memory buffer of the specified size.
    ///
    /// # Parameters
    ///
    /// * `size` - The size of the memory to allocate (in bytes).
    ///
    /// # Returns
    ///
    /// * `Ok(Handle)` - A handle to the newly allocated GPU memory.
    /// * `Err(IoError)` - If the allocation fails.
    #[cfg_attr(feature = "tracing", tracing::instrument(level = "trace", skip(self)))]
    pub fn reserve(&mut self, size: u64) -> Result<ManagedMemoryHandle, IoError> {
        let handle = self.streams.current().memory_management_gpu.reserve(size)?;

        Ok(handle)
    }

    #[cfg_attr(feature = "tracing", tracing::instrument(level = "trace", skip(self)))]
    pub fn empty(&mut self, size: u64) -> Result<Handle, IoError> {
        let handle = Handle::new(self.streams.current, size);
        let reserved = self.reserve(size)?;
        self.bind(reserved, handle.memory.clone());

        Ok(handle)
    }

    #[cfg_attr(feature = "tracing", tracing::instrument(level = "trace", skip(self)))]
    pub fn bind(&mut self, reserved: ManagedMemoryHandle, new: ManagedMemoryHandle) {
        let cursor = self.cursor();
        self.streams
            .current()
            .memory_management_gpu
            .bind(reserved, new, cursor)
            .unwrap();
    }

    /// Creates a [Bytes] instance from pinned memory, if suitable for the given size.
    ///
    /// For small data transfers (<= 100 MB) or when explicitly marked as pinned, this function
    /// uses pinned memory to optimize performance. For larger transfers, it falls back to regular memory.
    ///
    /// # Arguments
    ///
    /// * `size` - The number of bytes to allocate.
    /// * `marked_pinned` - Whether to force the use of pinned memory.
    ///
    /// # Returns
    ///
    /// A [Bytes] instance of the correct size.
    #[cfg_attr(feature = "tracing", tracing::instrument(level = "trace", skip(self)))]
    pub fn reserve_cpu(
        &mut self,
        size: usize,
        marked_pinned: bool,
        origin: Option<StreamId>,
    ) -> Bytes {
        // Use pinned memory for small transfers (<= 100 MB) or when explicitly marked.
        if !marked_pinned && size > 100 * MB {
            return Bytes::from_bytes_vec(vec![0; size]);
        }

        self.reserve_pinned(size, origin)
            .unwrap_or_else(|| Bytes::from_bytes_vec(vec![0; size]))
    }

    #[cfg_attr(feature = "tracing", tracing::instrument(level = "trace", skip(self)))]
    fn reserve_pinned(&mut self, size: usize, origin: Option<StreamId>) -> Option<Bytes> {
        let stream = match origin {
            Some(id) => self.streams.get(&id),
            None => self.streams.current(),
        };
        let handle = stream.memory_management_cpu.reserve(size as u64).ok()?;

        let binding = MemoryHandle::binding(handle);
        let resource = stream
            .memory_management_cpu
            .get_resource(binding.clone(), None, None)
            .ok()?;

        let controller = Box::new(PinnedMemoryManagedAllocController::init(binding, resource));
        // SAFETY: The binding has initialized memory for at least `size` bytes.
        Some(unsafe { Bytes::from_controller(controller, size) })
    }

    /// Asynchronously reads data from GPU memory to host memory based on the provided copy descriptors.
    ///
    /// # Parameters
    ///
    /// * `descriptors` - A vector of descriptors specifying the source GPU memory and its layout.
    ///
    /// # Returns
    ///
    /// * A `Future` resolving to:
    ///   * `Ok(Vec<Bytes>)` - The data read from the GPU as a vector of byte arrays.
    ///   * `Err(IoError)` - If the read operation fails.
    pub fn read_async(
        &mut self,
        descriptors: Vec<CopyDescriptor>,
    ) -> impl Future<Output = Result<Vec<Bytes>, ServerError>> + Send + use<> {
        let descriptors_moved = descriptors
            .iter()
            .map(|b| b.handle.clone())
            .collect::<Vec<_>>();

        let result = self.copies_to_bytes(descriptors, true);
        let fence = Fence::new(self.streams.current().sys);

        async move {
            let sync = fence.wait_sync();
            // Release memory handle.
            core::mem::drop(descriptors_moved);

            sync?;
            let bytes = result?;

            Ok(bytes)
        }
    }

    #[allow(unused)]
    /// TODO: Read data using the origin stream where the data was allocated.
    pub fn read_async_origin(
        &mut self,
        descriptors: Vec<CopyDescriptor>,
    ) -> impl Future<Output = Result<Vec<Bytes>, IoError>> + Send + use<> {
        let results = self.copies_to_bytes_origin(descriptors, true);

        async move {
            let (bytes, fences) = results?;

            for fence in fences {
                fence.wait_sync();
            }
            Ok(bytes)
        }
    }

    fn copies_to_bytes(
        &mut self,
        descriptors: Vec<CopyDescriptor>,
        pinned: bool,
    ) -> Result<Vec<Bytes>, IoError> {
        let mut result = Vec::with_capacity(descriptors.len());

        for descriptor in descriptors {
            result.push(self.copy_to_bytes(descriptor, pinned, None)?);
        }

        Ok(result)
    }

    fn copies_to_bytes_origin(
        &mut self,
        descriptors: Vec<CopyDescriptor>,
        pinned: bool,
    ) -> Result<(Vec<Bytes>, Vec<Fence>), IoError> {
        let mut data = Vec::with_capacity(descriptors.len());
        let mut fences = Vec::with_capacity(descriptors.len());
        let mut fenced = Vec::with_capacity(descriptors.len());

        for descriptor in descriptors {
            let stream = descriptor.handle.stream;
            let bytes = self.copy_to_bytes(descriptor, pinned, Some(stream))?;

            if !fenced.contains(&stream) {
                let fence = Fence::new(self.streams.get(&stream).sys);
                fenced.push(stream);
                fences.push(fence);
            }

            data.push(bytes);
        }

        Ok((data, fences))
    }

    pub fn copy_to_bytes(
        &mut self,
        descriptor: CopyDescriptor,
        pinned: bool,
        stream_id: Option<StreamId>,
    ) -> Result<Bytes, IoError> {
        let num_bytes = descriptor.shape.iter().product::<usize>() * descriptor.elem_size;
        let mut bytes = self.reserve_cpu(num_bytes, pinned, stream_id);
        self.write_to_cpu(descriptor, &mut bytes, stream_id)?;

        Ok(bytes)
    }

    /// Writes data to the host from the GPU memory as specified by the copy descriptor.
    ///
    /// # Parameters
    ///
    /// * `descriptor` - Describes the source GPU memory, its shape, strides, and element size.
    /// * `bytes` - The host bytes to write from the GPU.
    ///
    /// # Returns
    ///
    /// * `Ok(())` - If the write operation succeeds.
    /// * `Err(IoError)` - If the strides are invalid or the resource cannot be accessed.
    pub fn write_to_cpu(
        &mut self,
        descriptor: CopyDescriptor,
        bytes: &mut Bytes,
        stream_id: Option<StreamId>,
    ) -> Result<(), IoError> {
        let CopyDescriptor {
            handle: binding,
            shape,
            strides,
            elem_size,
        } = descriptor;

        if !has_pitched_row_major_strides(&shape, &strides) {
            return Err(IoError::UnsupportedStrides {
                backtrace: BackTrace::capture(),
            });
        }

        let resource = self.resource(binding)?;
        let stream = match stream_id {
            Some(id) => self.streams.get(&id),
            None => self.streams.current(),
        };

        // SAFETY: `resource.ptr` is a valid device pointer obtained from the memory manager,
        // `stream.sys` is an initialized CUDA stream, and `bytes` is pre-allocated with
        // sufficient capacity for the copy.
        unsafe {
            write_to_cpu(&shape, &strides, elem_size, bytes, resource.ptr, stream.sys)?;
        }

        Ok(())
    }

    /// Registers an error on the stream.
    pub fn error(&mut self, error: ServerError) {
        let stream = self.streams.current();
        stream.errors.push(error);
    }

    /// Writes data from the host to GPU memory as specified by the copy descriptor.
    ///
    /// # Parameters
    ///
    /// * `descriptor` - Describes the destination GPU memory, its shape, strides, and element size.
    /// * `data` - The host data to write to the GPU.
    ///
    /// # Returns
    ///
    /// * `Ok(())` - If the write operation succeeds.
    /// * `Err(IoError)` - If the strides are invalid or the resource cannot be accessed.
    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(level = "trace", skip(self, descriptor, data))
    )]
    pub fn write_to_gpu(&mut self, descriptor: CopyDescriptor, data: Bytes) -> Result<(), IoError> {
        let CopyDescriptor {
            handle,
            shape,
            strides,
            elem_size,
        } = descriptor;
        if !has_pitched_row_major_strides(&shape, &strides) {
            return Err(IoError::UnsupportedStrides {
                backtrace: BackTrace::capture(),
            });
        }

        let resource = self.resource(handle)?;

        let size = data.len();
        let data = match data.property() {
            AllocationProperty::File => {
                let mut buffer = self.reserve_pinned(size, None).unwrap();
                data.copy_into(&mut buffer);
                buffer
            }
            // Pageable host memory has no fast path: the driver stages the
            // transfer through its own bounce buffer, in chunks, ON THIS
            // THREAD. Measured on GB10 at 2.7 GB/s for a 14 MB write, against
            // 29.5 GB/s for a plain host memcpy of the same bytes and 50 GB/s
            // for the DMA engine itself. Copying into the pinned pool first
            // and letting the engine read from there is one extra host copy
            // and an ASYNCHRONOUS transfer — strictly cheaper above the size
            // where the copy is not itself the cost.
            AllocationProperty::Native | AllocationProperty::Other
                if pinned_staging_enabled()
                    && (PINNED_STAGING_MIN..=PINNED_STAGING_MAX).contains(&size) =>
            {
                match self.reserve_pinned(size, None) {
                    Some(mut buffer) => {
                        data.copy_into(&mut buffer);
                        buffer
                    }
                    // The pinned pool is a pool, not a guarantee. A refusal
                    // means fall back to the slow path, never fail the write.
                    None => data,
                }
            }
            _ => data,
        };
        let current = self.streams.current();

        // SAFETY: `resource.ptr` is a valid GPU allocation, `data` is a valid host buffer,
        // and `current.sys` is an initialized CUDA stream. The shape/strides have been
        // validated above to be pitched row-major.
        unsafe {
            write_to_gpu(
                &shape,
                &strides,
                elem_size,
                &data,
                resource.ptr,
                current.sys,
            )
        }?;

        current.drop_queue.push(data);

        Ok(())
    }

    /// Allocates a new GPU memory buffer and immediately copies contiguous host data into it.
    ///
    /// # Parameters
    ///
    /// * `data` - The host data to copy to the GPU.
    ///
    /// # Returns
    ///
    /// * `Ok(Handle)` - A handle to the newly allocated and populated GPU memory.
    /// * `Err(IoError)` - If the allocation or data copy fails.
    pub fn create_with_data(&mut self, data: &[u8]) -> Result<Handle, IoError> {
        let mut staging =
            self.reserve_pinned(data.len(), None)
                .ok_or_else(|| IoError::Unknown {
                    backtrace: BackTrace::capture(),
                    description: "Unable to reserve pinned memory".into(),
                })?;

        staging.copy_from_slice(data);

        let handle = self.empty(staging.len() as u64)?;

        self.write_to_gpu(
            CopyDescriptor {
                handle: handle.clone().binding(),
                shape: [data.len()].into(),
                strides: [1].into(),
                elem_size: 1,
            },
            staging,
        )?;

        Ok(handle)
    }

    /// Synchronizes the current stream, ensuring all pending operations are complete.
    ///
    /// # Returns
    ///
    /// * A `DynFut<()>` future that resolves when the stream is synchronized.
    pub fn sync(&mut self) -> DynFut<Result<(), ServerError>> {
        let fence = Fence::new(self.streams.current().sys);

        Box::pin(async { fence.wait_sync() })
    }

    /// Executes a registered CUDA kernel with the specified parameters.
    ///
    /// # Parameters
    ///
    /// * `kernel_id` - The identifier of the kernel to execute.
    /// * `kernel` - The cube task to compile if not cached.
    /// * `mode` - The execution mode for the current kernel.
    /// * `dispatch_count` - The number of thread blocks in the x, y, and z dimensions.
    /// * `tensor_maps` - Tensor maps for structured memory access.
    /// * `resources` - GPU resources (e.g., buffers) used by the kernel.
    /// * `scalars` - Scalar arguments passed to the kernel.
    /// * `logger` - The logger to use to write compilation & runtime info.
    ///
    /// # Panics
    ///
    /// * If the execution fails, with an error message or profiling error.
    #[allow(clippy::too_many_arguments)]
    pub fn kernel(
        &mut self,
        kernel_id: KernelId,
        kernel: Box<dyn CubeTask<CudaCompiler>>,
        mode: ExecutionMode,
        dispatch_count: (u32, u32, u32),
        tensor_maps: &[CUtensorMap],
        resources: &[GpuResource],
        const_info: Option<*mut c_void>,
        logger: Arc<ServerLogger>,
    ) -> Result<Option<CapturedNode>, LaunchError> {
        if !self.ctx.module_names.contains_key(&kernel_id) {
            self.ctx.compile_kernel(&kernel_id, kernel, mode, logger)?;
        }

        let stream = self.streams.current();

        // `CUBECL_TRACE_PTRS=1`: the device ADDRESS of every buffer each launch
        // binds. A captured graph records addresses, so whether a region is
        // replayable on a LATER step is first of all a question about whether
        // the allocator hands the same slices out again -- and that is a
        // question with an answer you can diff, not one to reason about.
        if trace_ptrs() {
            use core::fmt::Write as _;
            let mut line = alloc::string::String::new();
            let _ = write!(line, "[ptr] {kernel_id:?}");
            for r in resources {
                let _ = write!(line, " {:#x}", r.ptr as usize);
            }
            eprintln!("{line}");
        }

        // Looked up BEFORE the launch, because `execute_task` consumes the id
        // and because a lookup is the whole cost -- there is nothing to gain by
        // doing it after and a moved value to work around.
        let capture_shape = match stream.capturing {
            true => self.ctx.kernel_launch_shape(&kernel_id),
            false => None,
        };
        let result = self.ctx.execute_task(
            stream,
            kernel_id,
            dispatch_count,
            tensor_maps,
            resources,
            const_info,
        );

        // `flush` waits on a fence -- `cuEventSynchronize` -- which is exactly
        // the host block a capture cannot contain. Deferring it is safe: the
        // queue only grows, and the first launch after the capture closes
        // drains it. Callers are expected to pre-flush before capturing so the
        // deferral spans one region, not the whole run.
        if !stream.capturing && stream.drop_queue.should_flush() {
            stream.drop_queue.flush(|| Fence::new(stream.sys));
        }

        if let Err(err) = result {
            match self.ctx.timestamps.is_empty() {
                true => return Err(err),
                false => self.ctx.timestamps.error(ProfileError::Launch(err)),
            }
        };

        // The node this launch just became, while the capture still knows.
        //
        // A capture's FRONTIER -- the dependency set a subsequent launch would
        // hang off -- is, immediately after a launch on a linear single-stream
        // capture, exactly the one node that launch added. That is the only
        // moment the mapping from "the Nth launch of this region" to "this
        // CUgraphNode" is available at all: `cuGraphGetNodes` afterwards
        // returns the graph's nodes in NO defined order, and a graph holds
        // memcpy and memset nodes that no launch made, so there is nothing to
        // index by. The `assert` is not caution; it is the statement that the
        // capture is linear, which is what makes the index meaningful.
        let captured = if stream.capturing {
            use cudarc::driver::sys::{
                CUgraph, CUgraphEdgeData, CUgraphNode, CUstreamCaptureStatus,
                cuStreamGetCaptureInfo_v3,
            };
            let mut status = CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE;
            let mut cap_id: u64 = 0;
            let mut graph: CUgraph = core::ptr::null_mut();
            let mut deps: *const CUgraphNode = core::ptr::null();
            let mut edges: *const CUgraphEdgeData = core::ptr::null();
            let mut n_deps: usize = 0;
            // SAFETY: `stream.sys` is a live stream with a capture open on it;
            // every out-parameter is a live local. The returned `deps` array is
            // owned by the driver and read before any further capture call.
            unsafe {
                cuStreamGetCaptureInfo_v3(
                    stream.sys,
                    &mut status,
                    &mut cap_id,
                    &mut graph,
                    &mut deps,
                    &mut edges,
                    &mut n_deps,
                )
                .result()
                .map_err(|err| LaunchError::Unknown {
                    reason: format!("the open capture would not report its frontier: {err:?}"),
                    backtrace: BackTrace::capture(),
                })?;
            }
            if status == CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_ACTIVE && !deps.is_null() {
                assert_eq!(
                    n_deps, 1,
                    "a capture that forks has no launch order to index by:                      the frontier after one launch held {n_deps} nodes, not 1"
                );
                let (func, block, shared) =
                    capture_shape.expect("the kernel that just launched to be a loaded module");
                // SAFETY: `deps` points at `n_deps == 1` driver-owned nodes.
                Some(CapturedNode {
                    node: unsafe { *deps },
                    func,
                    block,
                    shared,
                })
            } else {
                None
            }
        } else {
            None
        };

        Ok(captured)
    }
}

/// One kernel launch that a graph capture recorded, named by the node it
/// became plus the parts of its parameters that come from the loaded module
/// rather than from the caller's bindings.
///
/// It is deliberately not the whole parameter set: the buffers, the cube count
/// and the packed argument blob are the caller's, and the caller is where they
/// stay owned.
#[derive(Debug, Clone, Copy)]
pub struct CapturedNode {
    /// The graph node this launch produced.
    pub node: cudarc::driver::sys::CUgraphNode,
    /// The function the node calls.
    pub func: cudarc::driver::sys::CUfunction,
    /// The cube dim it was launched with.
    pub block: (u32, u32, u32),
    /// Its dynamic shared memory size, in bytes.
    pub shared: u32,
}

/// Internal write to GPU command.
///
/// Writes data from a CPU buffer to a CUDA resource.
///
/// # Safety
///
/// - `dst_ptr` must be a valid CUDA device pointer with sufficient space for `data`.
/// - `stream` must be a valid, initialized CUDA stream.
/// - `data` must remain valid until the stream is synchronized.
/// - `shape`/`strides` must describe a valid pitched row-major layout (debug-asserted).
#[cfg_attr(
    feature = "tracing",
    tracing::instrument(level = "trace", skip(strides, data, dst_ptr, stream))
)]
pub(crate) unsafe fn write_to_gpu(
    shape: &Shape,
    strides: &Strides,
    elem_size: usize,
    data: &[u8],
    dst_ptr: u64,
    stream: *mut CUstream_st,
) -> Result<(), IoError> {
    #[cfg(debug_assertions)]
    try_check_pitched_row_major_strides(shape, strides).map_err(|e| IoError::Unknown {
        description: format!("write_to_gpu: invalid strides: {e}"),
        backtrace: BackTrace::capture(),
    })?;

    let rank = shape.len();
    if rank <= 1 {
        // SAFETY: For rank <= 1 data is contiguous. `dst_ptr` is a valid device pointer
        // and `data` is a valid host slice.
        unsafe {
            cudarc::driver::result::memcpy_htod_async(dst_ptr, data, stream).map_err(|e| {
                IoError::Unknown {
                    description: format!("CUDA memcpy_htod failed: {e}"),
                    backtrace: BackTrace::capture(),
                }
            })
        }
    } else {
        // As we've enforced that the strides are contiguous row-major,
        // and we know that the rank >= 2, we can construct a 2D view
        // for the aligned GPU pitched memcpy.

        let dim_x_shape = shape[rank - 1];
        let width_bytes = dim_x_shape * elem_size;

        // the second "dim"'s shape is the product of the rest of the space.
        let dim_y_shape: usize = shape[..rank - 1].iter().product();
        let pitch = strides[rank - 2] * elem_size;

        let cpy = CUDA_MEMCPY2D_st {
            srcMemoryType: CUmemorytype::CU_MEMORYTYPE_HOST,
            srcHost: data.as_ptr() as *const c_void,
            srcPitch: width_bytes,
            dstMemoryType: CUmemorytype::CU_MEMORYTYPE_DEVICE,
            dstDevice: dst_ptr,
            dstPitch: pitch,
            WidthInBytes: width_bytes,
            Height: dim_y_shape,
            srcXInBytes: Default::default(),
            srcY: Default::default(),
            srcDevice: Default::default(),
            srcArray: Default::default(),
            dstXInBytes: Default::default(),
            dstY: Default::default(),
            dstHost: Default::default(),
            dstArray: Default::default(),
        };

        // SAFETY: The `CUDA_MEMCPY2D_st` is fully initialized with valid source/dest
        // pointers, memory types, and dimensions derived from the validated shape/strides.
        unsafe {
            cuMemcpy2DAsync_v2(&cpy, stream)
                .result()
                .map_err(|e| IoError::Unknown {
                    description: format!("CUDA memcpy failed: {e}"),
                    backtrace: BackTrace::capture(),
                })
        }
    }
}

/// Internal write to CPU command.
///
/// Writes data from a CUDA resource to a CPU buffer.
///
/// # Safety
///
/// - `resource_ptr` must be a valid CUDA device pointer with at least `bytes.len()` readable bytes.
/// - `stream` must be a valid, initialized CUDA stream.
/// - `bytes` must have sufficient capacity for the copy.
/// - The caller must synchronize the stream before reading from `bytes`.
/// - `shape`/`strides` must describe a valid pitched row-major layout (debug-asserted).
#[cfg_attr(
    feature = "tracing",
    tracing::instrument(level = "trace", skip(strides, bytes, resource_ptr, stream))
)]
pub(crate) unsafe fn write_to_cpu(
    shape: &Shape,
    strides: &Strides,
    elem_size: usize,
    bytes: &mut Bytes,
    resource_ptr: u64,
    stream: *mut CUstream_st,
) -> Result<(), IoError> {
    #[cfg(debug_assertions)]
    try_check_pitched_row_major_strides(shape, strides).map_err(|e| IoError::Unknown {
        description: format!("write_to_cpu: invalid strides: {e}"),
        backtrace: BackTrace::capture(),
    })?;

    let rank = shape.len();
    let bytes = bytes.deref_mut();
    if rank <= 1 {
        // SAFETY: For rank <= 1 data is contiguous. `resource_ptr` is a valid device pointer
        // and `bytes` has sufficient capacity.
        unsafe {
            cudarc::driver::result::memcpy_dtoh_async(bytes, resource_ptr, stream).map_err(|e| {
                IoError::Unknown {
                    description: format!("CUDA memcpy_dtoh failed: {e}"),
                    backtrace: BackTrace::capture(),
                }
            })
        }
    } else {
        // As we've enforced that the strides are contiguous row-major,
        // and we know that the rank >= 2, we can construct a 2D view
        // for the aligned GPU pitched memcpy.

        let dim_x_shape = shape[rank - 1];
        let width_bytes = dim_x_shape * elem_size;

        // the second "dim"'s shape is the product of the rest of the space.
        let dim_y_shape: usize = shape[..rank - 1].iter().product();
        let pitch = strides[rank - 2] * elem_size;

        let cpy = CUDA_MEMCPY2D_st {
            srcMemoryType: CUmemorytype::CU_MEMORYTYPE_DEVICE,
            srcDevice: resource_ptr,
            srcPitch: pitch,
            dstMemoryType: CUmemorytype::CU_MEMORYTYPE_HOST,
            dstHost: bytes.as_mut_ptr() as *mut c_void,
            dstPitch: width_bytes,
            WidthInBytes: width_bytes,
            Height: dim_y_shape,
            srcXInBytes: Default::default(),
            srcY: Default::default(),
            srcArray: Default::default(),
            dstXInBytes: Default::default(),
            dstY: Default::default(),
            dstArray: Default::default(),
            srcHost: Default::default(),
            dstDevice: Default::default(),
        };

        // SAFETY: The `CUDA_MEMCPY2D_st` is fully initialized with valid source/dest
        // pointers, memory types, and dimensions derived from the validated shape/strides.
        unsafe {
            cuMemcpy2DAsync_v2(&cpy, stream)
                .result()
                .map_err(|e| IoError::Unknown {
                    description: format!("CUDA 2D memcpy failed: {e}"),
                    backtrace: BackTrace::capture(),
                })
        }
    }
}
