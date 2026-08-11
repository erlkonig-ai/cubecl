use crate::compute::uninit_vec;
extern crate alloc;
use cubecl_common::backtrace::BackTrace;
use cubecl_core::server::IoError;
use cubecl_runtime::storage::{ComputeStorage, StorageHandle, StorageId, StorageUtilization};
use cudarc::driver::DriverError;
use std::collections::HashMap;

enum AllocationKind {
    Async,
    Sync,
    /// Memory this storage did NOT allocate and must never free: a host region
    /// the GPU reads in place.
    ///
    /// On a part with `pageableMemoryAccessUsesHostPageTables` the device walks
    /// the host page tables, so an mmap'd file page IS addressable by a kernel
    /// at its host address and there is nothing to allocate, copy, or map. What
    /// there IS to get wrong is lifetime: the mapping must outlive every kernel
    /// that reads it. The `Arc` is that guarantee made structural — it owns the
    /// backing (the `Mmap`, boxed as `Any`) and is dropped only when this
    /// storage entry is deallocated, which cannot happen while a handle derived
    /// from it is alive.
    External(alloc::sync::Arc<dyn core::any::Any + Send + Sync>),
}

impl core::fmt::Debug for AllocationKind {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            AllocationKind::Async => write!(f, "Async"),
            AllocationKind::Sync => write!(f, "Sync"),
            AllocationKind::External(_) => write!(f, "External"),
        }
    }
}

/// Buffer storage for NVIDIA GPUs.
///
/// This struct manages memory resources for CUDA kernels, allowing them to be used as bindings
/// for launching kernels.
pub struct GpuStorage {
    memory: HashMap<StorageId, (cudarc::driver::sys::CUdeviceptr, AllocationKind)>,
    deallocations: Vec<StorageId>,
    ptr_bindings: PtrBindings,
    stream: cudarc::driver::sys::CUstream,
    mem_alignment: usize,
}

/// A GPU memory resource allocated for CUDA using [`GpuStorage`].
#[derive(Debug)]
pub struct GpuResource {
    /// The GPU memory pointer.
    pub ptr: u64,
    /// The CUDA binding pointer.
    pub binding: *mut std::ffi::c_void,
    /// The size of the resource.
    pub size: u64,
}

impl GpuResource {
    /// Creates a new [`GpuResource`].
    pub fn new(ptr: u64, binding: *mut std::ffi::c_void, size: u64) -> Self {
        Self { ptr, binding, size }
    }
}

impl GpuStorage {
    /// Creates a new [`GpuStorage`] instance for the specified CUDA stream.
    ///
    /// # Arguments
    ///
    /// * `mem_alignment` - The memory alignment requirement in bytes.
    pub fn new(mem_alignment: usize, stream: cudarc::driver::sys::CUstream) -> Self {
        Self {
            memory: HashMap::new(),
            deallocations: Vec::new(),
            ptr_bindings: PtrBindings::new(),
            stream,
            mem_alignment,
        }
    }

    /// Deallocates buffers marked for deallocation.
    ///
    /// This method processes all pending deallocations by freeing the associated GPU memory.
    fn perform_deallocations(&mut self) {
        self.deallocations
            .drain(..)
            .filter_map(|id| self.memory.remove(&id))
            // SAFETY: Each `ptr` was obtained from a prior `malloc_async` or `malloc_sync`
            // call and has not been freed yet. The deallocation method matches the allocation kind.
            .for_each(|(ptr, kind)| unsafe {
                match kind {
                    AllocationKind::Async => {
                        let _ = cudarc::driver::result::free_async(ptr, self.stream);
                    }
                    AllocationKind::Sync => {
                        if let Err(e) = cudarc::driver::result::free_sync(ptr) {
                            eprintln!("CUDA free error: {}", e);
                        }
                    }
                    // Not ours. Freeing a host mapping through the CUDA
                    // allocator would corrupt the heap; the only thing that
                    // ends here is our claim on the backing, as the `Arc`
                    // drops.
                    AllocationKind::External(keepalive) => drop(keepalive),
                }
            });
    }
}

// SAFETY: `GpuResource` contains CUDA device pointers that are safe to send between
// threads as long as proper stream synchronization is maintained by the caller.
unsafe impl Send for GpuResource {}
// SAFETY: `GpuStorage` is only accessed from one thread at a time via the `DeviceHandle`,
// which serializes all server access. The raw CUDA pointers it contains are never shared
// across threads without synchronization.
unsafe impl Send for GpuStorage {}

impl core::fmt::Debug for GpuStorage {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("GpuStorage").finish()
    }
}

/// Manages active CUDA buffer bindings in a ring buffer.
///
/// This ensures that pointers remain valid during kernel execution, preventing use-after-free errors.
struct PtrBindings {
    slots: Vec<cudarc::driver::sys::CUdeviceptr>,
    cursor: usize,
}

impl PtrBindings {
    /// Creates a new [`PtrBindings`] instance with a fixed-size ring buffer.
    fn new() -> Self {
        Self {
            // SAFETY: `CUdeviceptr` is a `u64`, valid for any bit pattern. All slots are
            // written via `register()` before being read, so uninitialized values are never observed.
            slots: unsafe { uninit_vec(crate::device::CUDA_MAX_BINDINGS as usize) },
            cursor: 0,
        }
    }

    /// Registers a new pointer in the ring buffer.
    ///
    /// # Arguments
    ///
    /// * `ptr` - The CUDA device pointer to register.
    ///
    /// # Returns
    ///
    /// A reference to the registered pointer.
    fn register(&mut self, ptr: u64) -> &u64 {
        self.slots[self.cursor] = ptr;
        let ptr_ref = self.slots.get(self.cursor).unwrap();

        self.cursor += 1;

        // Reset the cursor when the ring buffer is full.
        if self.cursor >= self.slots.len() {
            self.cursor = 0;
        }

        ptr_ref
    }
}

impl GpuStorage {
    /// Register a host pointer the GPU can read in place, WITHOUT allocating or
    /// copying, and return a [`StorageHandle`] addressing it.
    ///
    /// `ptr` is a **host** address. This is sound only where the device can
    /// address host memory directly — `cudaDevAttrPageableMemoryAccess` — which
    /// [`super::super::super::runtime`] checks before ever routing here. On such
    /// parts `cudaHostGetDevicePointer` returns the host address unchanged, so
    /// no translation is needed and none is done.
    ///
    /// `keepalive` owns whatever backs the region and is held until this entry
    /// is deallocated. See [`AllocationKind::External`].
    ///
    /// # Safety
    /// `ptr + offset .. ptr + offset + size` must be a live host region that
    /// stays valid and unmoved for as long as `keepalive` is held, and must not
    /// be mutated while a kernel may be reading it.
    pub unsafe fn register_external(
        &mut self,
        ptr: cudarc::driver::sys::CUdeviceptr,
        offset: u64,
        size: u64,
        keepalive: alloc::sync::Arc<dyn core::any::Any + Send + Sync>,
    ) -> StorageHandle {
        let id = StorageId::new();
        self.memory
            .insert(id, (ptr, AllocationKind::External(keepalive)));
        StorageHandle::new(id, StorageUtilization { offset, size })
    }
}

impl ComputeStorage for GpuStorage {
    type Resource = GpuResource;

    fn alignment(&self) -> usize {
        self.mem_alignment
    }

    fn get(&mut self, handle: &StorageHandle) -> Self::Resource {
        let (ptr, _) = self
            .memory
            .get(&handle.id)
            .expect("Storage handle not found");

        let offset = handle.offset();
        let size = handle.size();
        let ptr = self.ptr_bindings.register(ptr + offset);

        GpuResource::new(
            *ptr,
            ptr as *const cudarc::driver::sys::CUdeviceptr as *mut std::ffi::c_void,
            size,
        )
    }

    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(level = "trace", skip(self, size))
    )]
    fn alloc(&mut self, size: u64) -> Result<StorageHandle, IoError> {
        let id = StorageId::new();
        // SAFETY: Calling CUDA driver FFI to allocate device memory. First tries async
        // allocation on the stream; falls back to synchronous allocation if that fails.
        // The returned pointer is stored in `self.memory` and freed on deallocation.
        let ptr = unsafe { cudarc::driver::result::malloc_async(self.stream, size as usize) };
        let (ptr, kind) = match ptr {
            Ok(ptr) => (ptr, AllocationKind::Async),
            Err(_) => unsafe {
                match cudarc::driver::result::malloc_sync(size as usize) {
                    Ok(ptr) => (ptr, AllocationKind::Sync),
                    Err(DriverError(cudarc::driver::sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY)) => {
                        return Err(IoError::BufferTooBig {
                            size,
                            backtrace: BackTrace::capture(),
                        });
                    }
                    Err(other) => {
                        return Err(IoError::Unknown {
                            description: format!("CUDA allocation error: {other}"),
                            backtrace: BackTrace::capture(),
                        });
                    }
                }
            },
        };

        self.memory.insert(id, (ptr, kind));
        Ok(StorageHandle::new(
            id,
            StorageUtilization { offset: 0, size },
        ))
    }

    #[cfg_attr(feature = "tracing", tracing::instrument(level = "trace", skip(self)))]
    fn dealloc(&mut self, id: StorageId) {
        self.deallocations.push(id);
    }

    #[cfg_attr(feature = "tracing", tracing::instrument(level = "trace", skip(self)))]
    fn flush(&mut self) {
        self.perform_deallocations();
    }
}
