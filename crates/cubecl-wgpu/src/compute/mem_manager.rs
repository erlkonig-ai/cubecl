use crate::{WgpuResource, WgpuStorage};
use cubecl_common::stub::Arc;
#[cfg(target_os = "macos")]
use cubecl_common::stream_id::StreamId;
use cubecl_core::{
    MemoryConfiguration,
    server::{Binding, IoError},
};
use cubecl_ir::MemoryDeviceProperties;
use cubecl_runtime::{
    logging::ServerLogger,
    memory_management::{
        ManagedMemoryBinding, ManagedMemoryHandle, MemoryAllocationMode, MemoryHandle,
        MemoryManagement, MemoryManagementOptions,
    },
    storage::ComputeStorage,
};
use wgpu::BufferUsages;

#[derive(Debug)]
pub(crate) struct WgpuMemManager {
    memory_pool: MemoryManagement<WgpuStorage>,
    memory_uniforms: MemoryManagement<WgpuStorage>,
    memory_pool_staging: MemoryManagement<WgpuStorage>,
    uniforms: Vec<ManagedMemoryHandle>,
}

impl WgpuMemManager {
    pub(crate) fn new(
        device: wgpu::Device,
        memory_properties: MemoryDeviceProperties,
        memory_config: MemoryConfiguration,
        logger: Arc<ServerLogger>,
    ) -> Self {
        // Allocate storage & memory management for the main memory buffers. Any calls
        // to empty() or create() with a small enough size will be allocated from this
        // main memory pool.
        let memory_main = MemoryManagement::from_configuration(
            WgpuStorage::new(
                memory_properties.alignment as usize,
                device.clone(),
                BufferUsages::STORAGE
                    | BufferUsages::COPY_SRC
                    | BufferUsages::COPY_DST
                    | BufferUsages::INDIRECT,
            ),
            &memory_properties,
            memory_config,
            logger.clone(),
            MemoryManagementOptions::new("Main GPU Memory"),
        );

        let memory_staging = MemoryManagement::from_configuration(
            WgpuStorage::new(
                wgpu::COPY_BUFFER_ALIGNMENT as usize,
                device.clone(),
                wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            ),
            &memory_properties,
            // Unfortunately, we can't reuse a different part of a buffer for different reads, so we
            // can't have a single binding with multiple slices allocated.
            MemoryConfiguration::ExclusivePages,
            logger.clone(),
            MemoryManagementOptions::new("Staging CPU Memory").mode(MemoryAllocationMode::Auto),
        );

        // TODO: In the future this should not need STORAGE, if cube writes out all
        // uniforms as having <uniform> usage.
        let memory_uniforms = MemoryManagement::from_configuration(
            WgpuStorage::new(
                memory_properties.alignment as usize,
                device.clone(),
                BufferUsages::UNIFORM | BufferUsages::STORAGE | BufferUsages::COPY_DST,
            ),
            &memory_properties,
            MemoryConfiguration::ExclusivePages,
            logger,
            MemoryManagementOptions::new("Uniform GPU Memory").mode(MemoryAllocationMode::Auto),
        );

        Self {
            memory_pool: memory_main,
            memory_pool_staging: memory_staging,
            memory_uniforms,
            uniforms: vec![],
        }
    }

    pub(crate) fn bind(&mut self, old: ManagedMemoryHandle, new: ManagedMemoryHandle) {
        self.memory_pool.bind(old, new, 0).unwrap();
    }

    pub(crate) fn reserve(&mut self, size: u64) -> Result<ManagedMemoryHandle, IoError> {
        match self.memory_pool.reserve(size) {
            Ok(handle) => Ok(handle),
            Err(err) => Err(err),
        }
    }

    pub(crate) fn reserve_staging(
        &mut self,
        size: u64,
    ) -> Result<(WgpuResource, ManagedMemoryBinding), IoError> {
        let handle = self.memory_pool_staging.reserve(size)?;
        let binding = MemoryHandle::binding(handle);
        let resource = self
            .memory_pool_staging
            .get_resource(binding.clone(), None, None)
            .unwrap();

        Ok((resource, binding))
    }

    pub(crate) fn get_resource(&mut self, binding: Binding) -> Result<WgpuResource, IoError> {
        self.memory_pool
            .get_resource(binding.memory, binding.offset_start, binding.offset_end)
    }

    /// Zero-copy: import an mmap'd host region as a GPU buffer (Metal
    /// `newBufferWithBytesNoCopy:`) and register it as a tensor handle. `ptr` +
    /// `page_len` must be a page-aligned superset; `offset`/`size` locate the
    /// tensor inside it. No allocation, no copy — the GPU reads the host pages.
    #[cfg(target_os = "macos")]
    pub(crate) fn register_external_aliased(
        &mut self,
        ptr: *mut core::ffi::c_void,
        page_len: u64,
        offset: u64,
        size: u64,
        keepalive: std::sync::Arc<dyn std::any::Any + Send + Sync>,
        stream_id: StreamId,
    ) -> cubecl_core::server::Handle {
        let device = self.memory_pool.storage().device().clone();
        // SAFETY: caller guarantees `ptr`/`page_len` is a live, page-aligned,
        // immutable host region (mmap'd pile pages) outliving the handle —
        // `keepalive` (the backing's owner) enforces exactly that.
        let buffer = unsafe { make_aliased_buffer(&device, ptr, page_len) };
        let storage_handle = self
            .memory_pool
            .storage()
            .register_external(buffer, offset, size, keepalive);
        let mem = self.memory_pool.register_external(storage_handle);
        cubecl_core::server::Handle::from_memory(mem, stream_id, size)
    }

    pub(crate) fn reserve_uniform(&mut self, size: u64) -> WgpuResource {
        let slice = self
            .memory_uniforms
            .reserve(size)
            .expect("Must have enough memory for a uniform");
        // Keep track of this uniform until it is released.
        self.uniforms.push(slice.clone());
        let handle = self
            .memory_uniforms
            .get_storage(slice.binding())
            .expect("Failed to find storage!");
        self.memory_uniforms.storage().get(&handle)
    }

    pub(crate) fn memory_usage(&self) -> cubecl_runtime::memory_management::MemoryUsage {
        self.memory_pool.memory_usage()
    }

    pub(crate) fn memory_cleanup(&mut self, explicit: bool) {
        self.memory_pool.cleanup(explicit);
    }

    pub(crate) fn mode(&mut self, mode: MemoryAllocationMode) {
        self.memory_pool.mode(mode);
    }

    pub(crate) fn release_uniforms(&mut self) {
        self.uniforms.clear();
    }
}

/// Build a `wgpu::Buffer` that ALIASES the host memory at `ptr` (length
/// `page_len`) with no copy, via Metal's `newBufferWithBytesNoCopy:`. On Apple
/// Silicon's unified memory the GPU reads the very pages the pile is mmap'd into.
///
/// # Safety
/// `ptr` must be page-aligned (16 KiB) and point to at least `page_len` bytes of
/// live, immutable host memory that outlives the returned buffer; `page_len` must
/// be a whole number of pages. The buffer takes no ownership (deallocator =
/// `None`) — freeing it does NOT free the host memory.
#[cfg(target_os = "macos")]
unsafe fn make_aliased_buffer(
    device: &wgpu::Device,
    ptr: *mut core::ffi::c_void,
    page_len: u64,
) -> wgpu::Buffer {
    use objc2_metal::{MTLDevice, MTLResourceOptions};
    use wgpu::hal::api::Metal;

    let nn = core::ptr::NonNull::new(ptr).expect("null pointer for aliased buffer");
    let mtl_buffer = {
        let hal_device = unsafe { device.as_hal::<Metal>() }.expect("metal backend not active");
        let mtl_device = hal_device.raw_device();
        unsafe {
            mtl_device.newBufferWithBytesNoCopy_length_options_deallocator(
                nn,
                page_len as usize,
                MTLResourceOptions::StorageModeShared,
                None,
            )
        }
        .expect("newBufferWithBytesNoCopy returned nil (page alignment / length?)")
    };
    let hal_buffer = unsafe { wgpu::hal::metal::Device::buffer_from_raw(mtl_buffer, page_len) };
    unsafe {
        device.create_buffer_from_hal::<Metal>(
            hal_buffer,
            &wgpu::BufferDescriptor {
                label: Some("aliased-mmap-weight"),
                size: page_len,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            },
        )
    }
}
