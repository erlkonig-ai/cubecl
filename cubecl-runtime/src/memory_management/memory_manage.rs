use super::{
    MemoryConfiguration, MemoryPoolOptions, MemoryUsage, PoolType,
    memory_pool::{ExclusiveMemoryPool, MemoryPool, PersistentPool, SlicedPool},
};
use crate::{
    config::{
        CubeClRuntimeConfig, RuntimeConfig,
        memory::{MemoryLogLevel, PersistentMemory},
    },
    logging::ServerLogger,
    memory_management::{BytesFormat, memory_pool::Slice},
    server::IoError,
    storage::{ComputeStorage, StorageHandle},
};

use alloc::format;
use alloc::string::{String, ToString};
#[cfg(not(exclusive_memory_only))]
use alloc::vec;
use alloc::vec::Vec;
use cubecl_common::{backtrace::BackTrace, stub::Arc};
use cubecl_ir::MemoryDeviceProperties;

pub use super::memory_pool::{ManagedMemoryBinding, handle::*};

// These are 288 bytes vs 64 bytes. Adding boxing isn't really worth
// saving the 200 bytes.
#[allow(clippy::large_enum_variant)]
enum DynamicPool {
    Sliced(SlicedPool),
    Exclusive(ExclusiveMemoryPool),
}

impl MemoryPool for DynamicPool {
    fn accept(&self, size: u64) -> bool {
        match self {
            DynamicPool::Sliced(pool) => pool.accept(size),
            DynamicPool::Exclusive(pool) => pool.accept(size),
        }
    }

    fn find(&self, binding: &ManagedMemoryBinding) -> Result<&Slice, IoError> {
        match self {
            DynamicPool::Sliced(m) => m.find(binding),
            DynamicPool::Exclusive(m) => m.find(binding),
        }
    }

    #[cfg_attr(feature = "tracing", tracing::instrument(level = "trace", skip(self)))]
    fn try_reserve(&mut self, size: u64) -> Option<ManagedMemoryHandle> {
        match self {
            DynamicPool::Sliced(m) => m.try_reserve(size),
            DynamicPool::Exclusive(m) => m.try_reserve(size),
        }
    }

    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(level = "trace", skip(self, storage))
    )]
    fn alloc<Storage: ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        size: u64,
    ) -> Result<ManagedMemoryHandle, IoError> {
        match self {
            DynamicPool::Sliced(m) => m.alloc(storage, size),
            DynamicPool::Exclusive(m) => m.alloc(storage, size),
        }
    }

    fn get_memory_usage(&self) -> MemoryUsage {
        match self {
            DynamicPool::Sliced(m) => m.get_memory_usage(),
            DynamicPool::Exclusive(m) => m.get_memory_usage(),
        }
    }

    fn cleanup<Storage: ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        alloc_nr: u64,
        explicit: bool,
    ) {
        match self {
            DynamicPool::Sliced(m) => m.cleanup(storage, alloc_nr, explicit),
            DynamicPool::Exclusive(m) => m.cleanup(storage, alloc_nr, explicit),
        };
        storage.flush();
    }

    fn bind(
        &mut self,
        reserved: ManagedMemoryHandle,
        assigned: ManagedMemoryHandle,
        cursor: u64,
    ) -> Result<(), IoError> {
        match self {
            DynamicPool::Sliced(m) => m.bind(reserved, assigned, cursor),
            DynamicPool::Exclusive(m) => m.bind(reserved, assigned, cursor),
        }
    }
}

#[derive(Default, Clone, Copy, Debug)]
/// The mode of allocation used.
pub enum MemoryAllocationMode {
    /// Use the automatic memory management strategy for allocation.
    #[default]
    Auto,
    /// Use a persistent memory management strategy, meaning that all allocations are for data that is
    /// likely never going to be freed.
    Persistent,
}

/// What the capture arena is holding, and how it got there.
///
/// `misses` is the load-bearing one: a miss taken while a stream capture is
/// open is a driver allocation recorded as a graph MEMORY node, which is both a
/// per-replay cost and an address the next capture will not reproduce.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ArenaStats {
    /// How many times the arena has been opened.
    pub generation: u64,
    /// Reservations served from the arena since the last counter reset.
    pub served: u64,
    /// Of those, the ones that needed a fresh allocation from the storage.
    pub misses: u64,
    /// Slices the arena owns.
    pub slices: u64,
    /// Bytes the arena keeps reserved, whether in use or free.
    pub bytes_reserved: u64,
    /// Bytes of that currently held by a live handle.
    pub bytes_in_use: u64,
}

/// Reserves and keeps track of chunks of memory in the storage, and slices upon these chunks.
pub struct MemoryManagement<Storage> {
    name: String,
    persistent: PersistentPool,
    /// The capture-scoped arena. See [`MemoryManagement::arena_begin`].
    arena: PersistentPool,
    /// The `pool` byte an arena slice's location carries, so `find` and `bind`
    /// can route back here. One past the persistent pool's.
    arena_pool_pos: u8,
    /// While true, EVERY `reserve` is served from `arena` and none from the
    /// ordinary pools.
    arena_active: bool,
    /// Bumped on every [`MemoryManagement::arena_begin`]. Reporting only: what
    /// decides whether two graphs may share this arena is `arena_sizes`, not
    /// how many times it was opened.
    arena_generation: u64,
    /// Requests the arena served, and the subset it could not satisfy from an
    /// already-owned slice. A miss inside a capture is exactly a graph memory
    /// node, so this is the number the warm pass exists to drive to zero.
    arena_served: u64,
    arena_misses: u64,
    /// Every size the arena has served since it was last opened, in order.
    ///
    /// This is the region's IDENTITY as far as the arena is concerned. Two
    /// captures that share an arena share its slices -- that is the point, it
    /// is what makes their addresses comparable -- and sharing scratch is safe
    /// exactly when both are captures of the same region, replayed serially.
    /// The same region issues the same request sequence; a different one does
    /// not, and a hash over the window is what lets a replay say so.
    arena_sizes: Vec<u64>,
    pools: Vec<DynamicPool>,
    storage: Storage,
    alloc_reserve_count: u64,
    mode: MemoryAllocationMode,
    config: PersistentMemory,
    logger: Arc<ServerLogger>,
    /// Externally-registered storage handles (e.g. GPU buffers aliasing mmap'd
    /// host memory), keyed by memory-handle id. Resolved in `get_storage`,
    /// bypassing the allocation pools entirely — never reserved, reused, or
    /// reclaimed by the pool machinery. See [`Self::register_external`].
    external: hashbrown::HashMap<usize, StorageHandle>,
}

fn generate_bucket_sizes(
    start_size: u64,
    end_size: u64,
    max_buckets: usize,
    alignment: u64,
) -> Vec<u64> {
    let mut buckets = Vec::with_capacity(max_buckets);
    let log_min = (start_size as f64).ln();
    let log_max = (end_size as f64).ln();
    let log_range = log_max - log_min;

    // Pure exponential performed best, but let's try slightly denser in lower-mid range
    for i in 0..max_buckets {
        let p = i as f64 / (max_buckets - 1) as f64;
        // Slight bias toward lower-mid range with less aggressive curve than sigmoid
        let log_size = log_min + log_range * p;
        let size = log_size.exp() as u64;
        let aligned_size = size.next_multiple_of(alignment);
        buckets.push(aligned_size);
    }

    buckets.dedup();
    buckets
}

const DEALLOC_SCALE_MB: u64 = 1024 * 1024 * 1024;
const BASE_DEALLOC_PERIOD: u64 = 5000;

/// The options for creating a new [`MemoryManagement`] instance.
#[derive(Debug)]
pub struct MemoryManagementOptions {
    /// The name of the memory management.
    name: String,
    /// The [`MemoryAllocationOption`] used by this instance.
    memory: MemoryAllocationOption,
}

impl MemoryManagementOptions {
    /// Creates a new [`MemoryManagementOptions`].
    pub fn new<S: Into<String>>(name: S) -> Self {
        Self {
            name: name.into(),
            memory: MemoryAllocationOption::FromConfig,
        }
    }

    /// Forces the [`MemoryAllocationMode`] during execution to always be the provided one.
    pub fn mode(mut self, mode: MemoryAllocationMode) -> Self {
        self.memory = MemoryAllocationOption::Provided(mode);
        self
    }
}

#[derive(Default, Debug)]
/// Determines which [`MemoryAllocationMode`] is used during allocations.
enum MemoryAllocationOption {
    #[default]
    /// Uses the [`GlobalConfig`] to determine the mode of allocation.
    FromConfig,
    /// Use the provided [`MemoryAllocationMode`].
    Provided(MemoryAllocationMode),
}

impl<Storage: ComputeStorage> MemoryManagement<Storage> {
    /// Creates the options from device limits.
    pub fn from_configuration(
        storage: Storage,
        properties: &MemoryDeviceProperties,
        config: MemoryConfiguration,
        logger: Arc<ServerLogger>,
        options: MemoryManagementOptions,
    ) -> Self {
        let pool_options = match config {
            #[cfg(not(exclusive_memory_only))]
            MemoryConfiguration::SubSlices => {
                // Round chunk size to be aligned.
                let memory_alignment = properties.alignment;
                let max_page = properties.max_page_size;
                let mut pools = Vec::new();

                const MB: u64 = 1024 * 1024;

                // Add in a pool for allocations that are smaller than the min alignment,
                // as they can't use offsets at all (on wgpu at least).
                pools.push(MemoryPoolOptions {
                    pool_type: PoolType::ExclusivePages { max_alloc_size: 0 },
                    dealloc_period: None,
                });

                let mut current = max_page;
                let mut max_sizes = vec![];
                let mut page_sizes = vec![];
                let mut base = pools.len() as u32;

                while current >= 32 * MB {
                    current /= 4;

                    // Make sure every pool has an aligned size.
                    current = current.next_multiple_of(memory_alignment);

                    max_sizes.push(current / 2u64.pow(base));
                    page_sizes.push(current);
                    base += 1;
                }

                max_sizes.reverse();
                page_sizes.reverse();

                for i in 0..max_sizes.len() {
                    let max = max_sizes[i];
                    let page_size = page_sizes[i];

                    pools.push(MemoryPoolOptions {
                        // Creating max slices lower than the chunk size reduces fragmentation.
                        pool_type: PoolType::SlicedPages {
                            page_size,
                            max_slice_size: max,
                        },
                        dealloc_period: None,
                    });
                }

                // Add pools from big to small.
                pools.push(MemoryPoolOptions {
                    pool_type: PoolType::SlicedPages {
                        page_size: max_page / memory_alignment * memory_alignment,
                        max_slice_size: max_page / memory_alignment * memory_alignment,
                    },
                    dealloc_period: None,
                });
                pools
            }
            MemoryConfiguration::ExclusivePages => {
                // Add all bin sizes. Nb: because of alignment some buckets
                // end up as the same size, so only want unique ones,
                // but also keep the order, so a BTree will do.
                const MIN_BUCKET_SIZE: u64 = 1024 * 32;
                const NUM_POOLS: usize = 24;

                let sizes = generate_bucket_sizes(
                    MIN_BUCKET_SIZE,
                    properties.max_page_size,
                    NUM_POOLS,
                    properties.alignment,
                );

                sizes
                    .iter()
                    .map(|&size| {
                        let dealloc_period = (BASE_DEALLOC_PERIOD as f64
                            * (1.0 + size as f64 / (DEALLOC_SCALE_MB as f64)).round())
                            as u64;

                        MemoryPoolOptions {
                            pool_type: PoolType::ExclusivePages {
                                max_alloc_size: size,
                            },
                            dealloc_period: Some(dealloc_period),
                        }
                    })
                    .collect()
            }
            MemoryConfiguration::Custom { pool_options } => pool_options,
        };

        logger.log_memory(
            |level| !matches!(level, MemoryLogLevel::Disabled),
            || {
                let mut msg = String::new();
                for pool in pool_options.iter() {
                    msg += &format!("[{}] Using memory pool: \n {pool:?}\n", options.name);
                }
                msg
            },
        );

        let pools: Vec<_> = pool_options
            .iter()
            .enumerate()
            .map(|(pool_pos, options)| {
                let pool_pos = pool_pos as u8;

                match options.pool_type {
                    PoolType::SlicedPages {
                        page_size,
                        max_slice_size,
                    } => DynamicPool::Sliced(SlicedPool::new(
                        page_size,
                        max_slice_size,
                        properties.alignment,
                        pool_pos,
                    )),
                    PoolType::ExclusivePages { max_alloc_size } => {
                        DynamicPool::Exclusive(ExclusiveMemoryPool::new(
                            max_alloc_size,
                            properties.alignment,
                            options.dealloc_period.unwrap_or(u64::MAX),
                            pool_pos,
                        ))
                    }
                }
            })
            .collect();

        let config = CubeClRuntimeConfig::get().memory.persistent_memory.clone();

        let mode = match options.memory {
            MemoryAllocationOption::Provided(mode) => mode,
            MemoryAllocationOption::FromConfig => match config {
                PersistentMemory::Enabled => MemoryAllocationMode::Auto,
                PersistentMemory::Disabled => MemoryAllocationMode::Auto,
                PersistentMemory::Enforced => MemoryAllocationMode::Persistent,
            },
        };

        Self {
            name: options.name,
            persistent: PersistentPool::new(
                properties.max_page_size,
                properties.alignment,
                pools.len() as u8,
            ),
            // The arena accepts any size the storage will accept: a captured
            // region does not get to choose which of its buffers are small.
            arena: PersistentPool::new(u64::MAX, properties.alignment, pools.len() as u8 + 1),
            arena_pool_pos: pools.len() as u8 + 1,
            arena_active: false,
            arena_generation: 0,
            arena_served: 0,
            arena_misses: 0,
            arena_sizes: Vec::new(),
            pools,
            storage,
            alloc_reserve_count: 0,
            mode,
            config,
            logger,
            external: hashbrown::HashMap::new(),
        }
    }

    /// Change the mode of allocation.
    pub fn mode(&mut self, mode: MemoryAllocationMode) {
        // We override the mode based on the cubecl config.
        let mode = match self.config {
            PersistentMemory::Enabled => mode,
            PersistentMemory::Disabled | PersistentMemory::Enforced => return,
        };

        self.logger.log_memory(
            |level| !matches!(level, MemoryLogLevel::Disabled),
            || {
                format!(
                    "[{}] Setting memory allocation mode: from {:?} => {mode:?}",
                    self.name, self.mode
                )
            },
        );
        self.mode = mode;
    }

    /// Route every subsequent `reserve` into the capture arena, and return the
    /// generation this opening is.
    ///
    /// The arena exists because a captured graph records ADDRESSES. Three
    /// things have to be true at once, and no ordinary pool gives all three:
    ///
    /// 1. **Reuse inside the region.** A buffer born inside the captured region
    ///    and dead inside it must hand its slice back, so the region's peak
    ///    footprint is its live set and not its allocation count. Without this
    ///    every intra-region allocation becomes a driver allocation made while
    ///    the stream is capturing, i.e. a graph MEMORY node.
    /// 2. **Whole-arena reservation.** Once the region is captured, every slice
    ///    the arena ever handed out is baked into a node, so NOTHING outside a
    ///    capture may be given one for as long as the graph can be replayed.
    ///    The arena is only ever consulted while it is open, and it never
    ///    returns pages to the storage, so this holds for the whole arena and
    ///    not just for the slices that happened to be live at capture end.
    /// 3. **A deterministic base.** Reopening the arena resets no slice and
    ///    frees nothing; slices are matched by exact effective size in a fixed
    ///    order. A periodic region therefore issues an identical request
    ///    sequence against an identical free set and gets identical addresses,
    ///    which is what makes two captures of the same region comparable and a
    ///    single graph legitimate for every step.
    ///
    /// Note what this is NOT: it is not "hold the buffers a capture touched".
    /// Holding gets reservation right and reuse wrong -- it is precisely what
    /// stops (1) -- and holding only the buffers that were live when the region
    /// opened gets reuse right and reservation wrong for everything born inside
    /// it. The two properties have to be separated onto different memory, which
    /// is what the arena is.
    ///
    /// Buffers that were live when the arena opened are NOT in it. They belong
    /// to the ordinary pools, and if one dies inside the region its slice goes
    /// back to a pool the region is not allocating from -- so it cannot be
    /// handed to a later node of the same region. That is the whole of the
    /// intra-region aliasing bug, closed structurally rather than by holding.
    /// The caller still owes such a buffer a hold for the graph's lifetime, to
    /// stop the pool handing it out AFTER the capture closes.
    ///
    /// TWO GRAPHS MAY SHARE ONE ARENA, and under (3) they positively should:
    /// that is what makes their recorded addresses comparable, and it is what
    /// lets a graph captured on one step stand in for the next. They then share
    /// scratch, which is safe exactly when they are captures of the SAME region
    /// and are replayed serially. Serial is a property of the one stream they
    /// replay on; same-region is what [`Self::arena_signature`] is for.
    pub fn arena_begin(&mut self) -> u64 {
        self.arena_generation += 1;
        self.arena_active = true;
        self.arena_sizes.clear();
        self.logger.log_memory(
            |level| !matches!(level, MemoryLogLevel::Disabled),
            || {
                format!(
                    "[{}] Capture arena open, generation {}",
                    self.name, self.arena_generation
                )
            },
        );
        self.arena_generation
    }

    /// Stop routing `reserve` into the arena. The arena keeps everything it
    /// owns: closing it is the end of allocation from it, never the end of its
    /// reservation.
    pub fn arena_end(&mut self) {
        self.arena_active = false;
    }

    /// Whether `reserve` is currently being served from the arena.
    pub fn arena_active(&self) -> bool {
        self.arena_active
    }

    /// What the arena is holding, and how it got there.
    pub fn arena_stats(&self) -> ArenaStats {
        let usage = self.arena.get_memory_usage();
        ArenaStats {
            generation: self.arena_generation,
            served: self.arena_served,
            misses: self.arena_misses,
            slices: self.arena.slice_count() as u64,
            bytes_reserved: usage.bytes_reserved,
            bytes_in_use: usage.bytes_in_use,
        }
    }

    /// Zero the served/miss counters, so a capture pass can be counted apart
    /// from the warm pass that preceded it. Does not free or reset anything.
    pub fn arena_reset_counters(&mut self) {
        self.arena_served = 0;
        self.arena_misses = 0;
    }

    /// Give the arena's free pages back to the storage.
    ///
    /// Only legal once no captured graph built against it can still be
    /// replayed -- every such graph holds pointers into these pages. Slices
    /// still held by a live handle are kept.
    pub fn arena_release(&mut self) {
        assert!(
            !self.arena_active,
            "[{}] arena_release while the arena is open",
            self.name
        );
        self.arena
            .cleanup(&mut self.storage, self.alloc_reserve_count, true);
    }

    /// Whether this binding's memory came from the arena.
    pub fn arena_owns(&self, binding: &ManagedMemoryBinding) -> bool {
        binding.descriptor().location().pool == self.arena_pool_pos
    }

    /// How many requests the open arena window has served so far. A caller
    /// brackets a region with two of these to name the window it owns.
    pub fn arena_mark(&self) -> usize {
        self.arena_sizes.len()
    }

    /// A hash of every size served since `from`, in order: the signature of the
    /// region that allocated them. See [`Self::arena_sizes`].
    pub fn arena_signature(&self, from: usize) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for size in &self.arena_sizes[from.min(self.arena_sizes.len())..] {
            h ^= *size;
            h = h.wrapping_mul(0x1000_0000_01b3);
        }
        h
    }

    fn arena_reserve(&mut self, size: u64) -> Result<ManagedMemoryHandle, IoError> {
        self.arena_served += 1;
        self.arena_sizes.push(size);
        if let Some(handle) = self.arena.try_reserve(size) {
            return Ok(handle);
        }
        // A miss is a real driver allocation. Inside a capture that is a graph
        // memory node; the warm pass exists so this branch is not taken then.
        self.arena_misses += 1;
        self.arena.alloc(&mut self.storage, size)
    }

    /// Cleanup allocations in pools that are deemed unnecessary.
    pub fn cleanup(&mut self, explicit: bool) {
        self.logger.log_memory(
            |level| !matches!(level, MemoryLogLevel::Disabled) && explicit,
            || "Manual memory cleanup ...".to_string(),
        );

        self.persistent
            .cleanup(&mut self.storage, self.alloc_reserve_count, explicit);

        for pool in self.pools.iter_mut() {
            pool.cleanup(&mut self.storage, self.alloc_reserve_count, explicit);
        }
    }

    /// Returns the storage from the specified binding
    pub fn get_cursor(&self, binding: ManagedMemoryBinding) -> Result<u64, IoError> {
        // External storage is registered rather than reserved: it has no
        // `Slice`, so it has no cursor to read. Returning 0 is not a fallback,
        // it is the right answer -- the cursor exists to order a read after the
        // stream operation that last WROTE the memory, and external storage is
        // pinned `can_mut() == false` precisely so that nothing ever writes it.
        // There is no producing operation to wait for.
        //
        // Only the CUDA backend reaches this: its multi-stream scheduler calls
        // `handle_cursor` on every binding, while the wgpu backend the seam was
        // first written for has no cursors at all. That is why an external
        // handle used to die here with "Memory page 0 doesn't exist".
        if self.external.contains_key(&binding.descriptor().id.value) {
            return Ok(0);
        }
        let slice = self.find(binding)?;
        Ok(slice.cursor)
    }

    /// Returns the storage from the specified binding
    fn find(&self, binding: ManagedMemoryBinding) -> Result<&Slice, IoError> {
        let id = binding.descriptor();

        if id.location().pool == self.arena_pool_pos {
            return self.arena.find(&binding);
        }

        if id.location().pool >= self.pools.len() as u8 {
            return self.persistent.find(&binding);
        }

        let pool =
            self.pools
                .get(id.location().pool as usize)
                .ok_or_else(|| IoError::NotFound {
                    backtrace: BackTrace::capture(),
                    reason: format!("Pool {} doesn't exist", id.location().pool).into(),
                })?;

        let slice = pool.find(&binding)?;

        assert_eq!(slice.handle.descriptor(), binding.descriptor());

        Ok(slice)
    }

    /// Returns the storage from the specified binding
    pub fn get_storage(&mut self, binding: ManagedMemoryBinding) -> Result<StorageHandle, IoError> {
        if let Some(storage) = self.external.get(&binding.descriptor().id.value) {
            return Ok(storage.clone());
        }
        let slice = self.find(binding)?;
        Ok(slice.storage.clone())
    }

    /// Register an externally-created [`StorageHandle`] (e.g. a GPU buffer that
    /// aliases mmap'd host memory) as a tensor-addressable memory handle,
    /// BYPASSING the allocation pools. The returned handle resolves to `storage`
    /// via [`Self::get_storage`]/[`Self::get_resource`] but is never reserved,
    /// reused, sub-sliced, or reclaimed — the caller owns the underlying storage
    /// lifetime. This is the zero-copy seam for weights-as-tribles: an mmap'd f16
    /// pile page becomes a GPU tensor with no copy and no allocation.
    ///
    /// The returned handle (and every clone of it) permanently reports
    /// `can_mut() == false`: external storage aliases READ-ONLY host memory,
    /// so it must never be picked as an in-place kernel destination. Burn's
    /// elementwise kernels reuse an "owned" input handle as the output buffer
    /// (`can_mut()` = a handle-count heuristic, `strong_count <= 2`); on a
    /// read-only mmap the GPU's in-place write is silently dropped by the
    /// pager and the op returns its INPUT bytes — e.g. voxtral's folded-ear
    /// load saw `q.mul_scalar(1/sqrt(d))` come back unscaled (2026-07-12,
    /// caught by the aliased-vs-materialized A/B being token-divergent).
    /// Pinning two extra handle clones for the registration's life (external
    /// entries are never removed anyway) pushes every user-facing clone over
    /// the heuristic's threshold, forcing all consumers to allocate outputs.
    pub fn register_external(&mut self, storage: StorageHandle) -> ManagedMemoryHandle {
        let handle = ManagedMemoryHandle::new();
        core::mem::forget(handle.clone());
        core::mem::forget(handle.clone());
        self.external.insert(handle.descriptor().id.value, storage);
        handle
    }

    /// Returns the resource from the storage at the specified handle
    pub fn get_resource(
        &mut self,
        binding: ManagedMemoryBinding,
        offset_start: Option<u64>,
        offset_end: Option<u64>,
    ) -> Result<Storage::Resource, IoError> {
        let handle = self.get_storage(binding)?;

        let handle = match offset_start {
            Some(offset) => handle.offset_start(offset),
            None => handle,
        };
        let handle = match offset_end {
            Some(offset) => handle.offset_end(offset),
            None => handle,
        };
        Ok(self.storage().get(&handle))
    }

    /// Finds a spot in memory for a resource with the given size in bytes, and returns a handle to it
    #[cfg_attr(feature = "tracing", tracing::instrument(level = "trace", skip(self)))]
    pub fn reserve(&mut self, size: u64) -> Result<ManagedMemoryHandle, IoError> {
        // If this happens every nanosecond, counts overflows after 585 years, so not worth thinking too
        // hard about overflow here.
        self.alloc_reserve_count += 1;

        // The arena is not one pool among the others: while it is open it is
        // the ONLY one, because an address a captured node records has to come
        // from memory nothing outside the capture can be given.
        if self.arena_active {
            return self.arena_reserve(size);
        }

        if let Some(val) = self.persistent.try_reserve(size) {
            self.logger.log_memory(
                |level| matches!(level, MemoryLogLevel::Full),
                || {
                    format!(
                        "[{}] Reserved memory {size} using persistent memory",
                        self.name
                    )
                },
            );
            return Ok(val);
        }

        if matches!(self.mode, MemoryAllocationMode::Persistent) || self.persistent.has_size(size) {
            let allocated = self.persistent.alloc(&mut self.storage, size);

            self.logger.log_memory(
                |level| !matches!(level, MemoryLogLevel::Disabled),
                || {
                    format!(
                        "[{}] Allocated a new memory page using persistent memory, \n{}",
                        self.name, self,
                    )
                },
            );
            return allocated;
        }

        self.logger.log_memory(
            |level| matches!(level, MemoryLogLevel::Full),
            || {
                format!(
                    "[{}] Reserved memory {} using dynamic pool",
                    self.name,
                    BytesFormat::new(size)
                )
            },
        );

        // Find first pool that fits this allocation
        let pool = self
            .pools
            .iter_mut()
            .find(|p| p.accept(size))
            .ok_or(IoError::BufferTooBig {
                size,
                backtrace: BackTrace::capture(),
            })?;

        if let Some(slice) = pool.try_reserve(size) {
            return Ok(slice);
        }

        let allocated = pool.alloc(&mut self.storage, size);

        self.logger.log_memory(
            |level| matches!(level, MemoryLogLevel::Full),
            || {
                format!(
                    "[{}], Allocated a new memory page, current usage: \n{}",
                    self.name, self
                )
            },
        );

        allocated
    }

    /// Fetch the storage used by the memory manager.
    ///
    /// # Notes
    ///
    /// The storage should probably not be used for allocations since the handles won't be
    /// compatible with the ones provided by the current trait. Prefer using the
    /// [alloc](ComputeStorage::alloc) and [dealloc](ComputeStorage::dealloc) functions.
    ///
    /// This is useful if you need to time the deallocations based on async computation, or to
    /// change the mode of storage for different reasons.
    pub fn storage(&mut self) -> &mut Storage {
        &mut self.storage
    }

    /// Get the current memory usage.
    pub fn memory_usage(&self) -> MemoryUsage {
        let memory_usage = self.pools.iter().map(|x| x.get_memory_usage()).fold(
            MemoryUsage {
                number_allocs: 0,
                bytes_in_use: 0,
                bytes_padding: 0,
                bytes_reserved: 0,
            },
            |m1, m2| m1.combine(m2),
        );
        memory_usage
            .combine(self.persistent.get_memory_usage())
            .combine(self.arena.get_memory_usage())
    }

    /// Print out a report of the current memory usage.
    pub fn print_memory_usage(&self) {
        #[cfg(feature = "std")]
        log::info!("{}", self.memory_usage());
    }

    /// Binds the given [handle](HandleId) to a [`MemorySlot`].
    pub fn bind(
        &mut self,
        reserved: ManagedMemoryHandle,
        assigned: ManagedMemoryHandle,
        cursor: u64,
    ) -> Result<(), IoError> {
        let descriptor = reserved.descriptor();

        if descriptor.location().init == 0 {
            return Err(IoError::NotFound {
                backtrace: BackTrace::capture(),
                reason: "Reserved memory isn't initialized".into(),
            });
        }

        let pool_index = descriptor.location().pool as usize;
        if pool_index == self.arena_pool_pos as usize {
            return self.arena.bind(reserved, assigned, cursor);
        }
        if pool_index >= self.pools.len() {
            return self.persistent.bind(reserved, assigned, cursor);
        }

        self.pools
            .get_mut(pool_index)
            .map(|p| p.bind(reserved, assigned, cursor))
            .ok_or_else(|| IoError::NotFound {
                backtrace: BackTrace::capture(),
                reason: format!("Memory pool {} doesn't exist", pool_index).into(),
            })?
    }
}

impl<Storage: ComputeStorage> core::fmt::Display for MemoryManagement<Storage> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("\n# MemoryManagement\n\n")?;
        f.write_fmt(format_args!(" - name: {:?}\n", self.name))?;
        f.write_fmt(format_args!("\n## Persistent\n\n{}", self.persistent))?;
        f.write_fmt(format_args!(
            "\n## Capture arena (generation {}, {})\n\n{}",
            self.arena_generation,
            match self.arena_active {
                true => "open",
                false => "closed",
            },
            self.arena
        ))?;
        f.write_str("\n## Dynamic\n\n")?;

        for pool in self.pools.iter() {
            match pool {
                DynamicPool::Sliced(pool) => f.write_fmt(format_args!("{pool}\n"))?,
                DynamicPool::Exclusive(pool) => f.write_fmt(format_args!("{pool}\n"))?,
            }
        }
        let memory_usage = self.memory_usage();
        f.write_fmt(format_args!("\n## Summary\n\n{memory_usage}"))?;

        Ok(())
    }
}

impl<Storage> core::fmt::Debug for MemoryManagement<Storage> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(
            alloc::format!(
                "DynamicMemoryManagement {:?}",
                core::any::type_name::<Storage>(),
            )
            .as_str(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{memory_management::MemoryManagement, storage::BytesStorage};
    use alloc::vec;

    const DUMMY_MEM_PROPS: MemoryDeviceProperties = MemoryDeviceProperties {
        max_page_size: 128 * 1024 * 1024,
        alignment: 32,
    };

    fn options() -> MemoryManagementOptions {
        MemoryManagementOptions {
            name: "test".into(),
            memory: MemoryAllocationOption::FromConfig,
        }
    }

    // Test pools with slices.
    #[test_log::test]
    #[cfg(not(exclusive_memory_only))]
    fn test_handle_mutability() {
        let mut memory_management = MemoryManagement::from_configuration(
            BytesStorage::default(),
            &DUMMY_MEM_PROPS,
            MemoryConfiguration::SubSlices,
            Arc::new(ServerLogger::default()),
            options(),
        );
        let handle = memory_management.reserve(10).unwrap();
        let other_ref = handle.clone();
        assert!(!handle.can_mut(), "Handle can't be mut when multiple ref.");
        drop(other_ref);
        assert!(handle.can_mut(), "Handle should be mut when only one ref.");
    }

    // Test pools with slices.
    #[test_log::test]
    #[cfg(not(exclusive_memory_only))]
    fn test_memory_usage() {
        let max_page_size = 512;

        let mut memory_management = MemoryManagement::from_configuration(
            BytesStorage::default(),
            &DUMMY_MEM_PROPS,
            MemoryConfiguration::Custom {
                pool_options: vec![MemoryPoolOptions {
                    pool_type: PoolType::ExclusivePages {
                        max_alloc_size: max_page_size,
                    },
                    dealloc_period: None,
                }],
            },
            Arc::new(ServerLogger::default()),
            options(),
        );
        let handle = memory_management.reserve(100);
        let usage = memory_management.memory_usage();

        assert_eq!(usage.bytes_in_use, 100);
        assert!(usage.bytes_reserved >= 100 && usage.bytes_reserved <= max_page_size);

        // Drop and re-alloc.
        drop(handle);
        let _handle = memory_management.reserve(100);
        let usage_new = memory_management.memory_usage();
        assert_eq!(usage, usage_new);
    }

    #[test_log::test]
    fn alloc_two_chunks_on_one_page() {
        let page_size = 2048;

        let mut memory_management = MemoryManagement::from_configuration(
            BytesStorage::default(),
            &DUMMY_MEM_PROPS,
            MemoryConfiguration::Custom {
                pool_options: vec![MemoryPoolOptions {
                    pool_type: PoolType::SlicedPages {
                        page_size,
                        max_slice_size: page_size,
                    },
                    dealloc_period: None,
                }],
            },
            Arc::new(ServerLogger::default()),
            options(),
        );

        let alloc_size = 512;
        let _handle = memory_management.reserve(alloc_size);
        let _new_handle = memory_management.reserve(alloc_size);

        let usage = memory_management.memory_usage();
        assert_eq!(usage.number_allocs, 2);
        assert_eq!(usage.bytes_in_use, alloc_size * 2);
        assert_eq!(usage.bytes_reserved, page_size);
    }

    #[test_log::test]
    fn alloc_reuses_storage() {
        // If no storage is re-used, this will allocate two pages.
        let page_size = 512;

        let mut memory_management = MemoryManagement::from_configuration(
            BytesStorage::default(),
            &DUMMY_MEM_PROPS,
            MemoryConfiguration::Custom {
                pool_options: vec![MemoryPoolOptions {
                    pool_type: PoolType::SlicedPages {
                        page_size,
                        max_slice_size: page_size,
                    },
                    dealloc_period: None,
                }],
            },
            Arc::new(ServerLogger::default()),
            options(),
        );

        let alloc_size = 512;
        let _handle = memory_management.reserve(alloc_size);
        drop(_handle);
        let _new_handle = memory_management.reserve(alloc_size);

        let usage = memory_management.memory_usage();
        assert_eq!(usage.number_allocs, 1);
        assert_eq!(usage.bytes_in_use, alloc_size);
        assert_eq!(usage.bytes_reserved, page_size);
    }

    #[test_log::test]
    fn alloc_allocs_new_storage() {
        let page_size = 1024;

        let mut memory_management = MemoryManagement::from_configuration(
            BytesStorage::default(),
            &DUMMY_MEM_PROPS,
            MemoryConfiguration::Custom {
                pool_options: vec![MemoryPoolOptions {
                    pool_type: PoolType::SlicedPages {
                        page_size,
                        max_slice_size: page_size,
                    },
                    dealloc_period: None,
                }],
            },
            Arc::new(ServerLogger::default()),
            options(),
        );

        let alloc_size = 768;
        let _handle = memory_management.reserve(alloc_size);
        let _new_handle = memory_management.reserve(alloc_size);

        let usage = memory_management.memory_usage();
        assert_eq!(usage.number_allocs, 2);
        assert_eq!(usage.bytes_in_use, alloc_size * 2);
        assert_eq!(usage.bytes_reserved, page_size * 2);
    }

    #[test_log::test]
    fn alloc_respects_alignment_size() {
        let page_size = 500;
        let mut memory_management = MemoryManagement::from_configuration(
            BytesStorage::default(),
            &MemoryDeviceProperties {
                max_page_size: page_size,
                alignment: 50,
            },
            MemoryConfiguration::Custom {
                pool_options: vec![MemoryPoolOptions {
                    pool_type: PoolType::SlicedPages {
                        page_size,
                        max_slice_size: page_size,
                    },
                    dealloc_period: None,
                }],
            },
            Arc::new(ServerLogger::default()),
            options(),
        );
        let alloc_size = 40;
        let _handle = memory_management.reserve(alloc_size);
        let _new_handle = memory_management.reserve(alloc_size);
        let usage = memory_management.memory_usage();
        // Each slice should be aligned to 50 bytes, so 20 padding bytes.
        assert_eq!(usage.bytes_padding, 10 * 2);
    }

    #[test_log::test]
    fn allocs_on_correct_page() {
        let sizes = [100, 200, 300, 400];

        let pools = sizes
            .iter()
            .map(|size| MemoryPoolOptions {
                pool_type: PoolType::SlicedPages {
                    page_size: *size,
                    max_slice_size: *size,
                },
                dealloc_period: None,
            })
            .collect();
        let mut memory_management = MemoryManagement::from_configuration(
            BytesStorage::default(),
            &MemoryDeviceProperties {
                max_page_size: 128 * 1024 * 1024,
                alignment: 10,
            },
            MemoryConfiguration::Custom {
                pool_options: pools,
            },
            Arc::new(ServerLogger::default()),
            options(),
        );
        // Allocate one thing on each page.
        let alloc_sizes = [50, 150, 250, 350];
        let _handles = alloc_sizes.map(|s| memory_management.reserve(s));

        let usage = memory_management.memory_usage();

        // Total memory should be size of all pages, and no more.
        assert_eq!(usage.bytes_in_use, alloc_sizes.iter().sum::<u64>());
        assert!(usage.bytes_reserved >= sizes.iter().sum::<u64>());
    }

    #[test_log::test]
    #[cfg(not(exclusive_memory_only))]
    fn allocate_deallocate_reallocate() {
        let mut memory_management = MemoryManagement::from_configuration(
            BytesStorage::default(),
            &MemoryDeviceProperties {
                max_page_size: 128 * 1024 * 1024,
                alignment: 32,
            },
            MemoryConfiguration::SubSlices,
            Arc::new(ServerLogger::default()),
            options(),
        );
        // Allocate a bunch
        let handles: Vec<_> = (0..5)
            .map(|i| memory_management.reserve(1000 * (i + 1)))
            .collect();
        let usage_before = memory_management.memory_usage();
        // Deallocate
        drop(handles);
        // Reallocate
        let _new_handles: Vec<_> = (0..5)
            .map(|i| memory_management.reserve(1000 * (i + 1)))
            .collect();
        let usage_after = memory_management.memory_usage();
        assert_eq!(usage_before.number_allocs, usage_after.number_allocs);
        assert_eq!(usage_before.bytes_in_use, usage_after.bytes_in_use);
        // Usage after can actually be _less_ because of defragging.
        assert!(usage_before.bytes_reserved >= usage_after.bytes_reserved);
    }

    #[test_log::test]
    #[cfg(not(exclusive_memory_only))]
    fn test_fragmentation_resistance() {
        let mut memory_management = MemoryManagement::from_configuration(
            BytesStorage::default(),
            &MemoryDeviceProperties {
                max_page_size: 128 * 1024 * 1024,
                alignment: 32,
            },
            MemoryConfiguration::SubSlices,
            Arc::new(ServerLogger::default()),
            options(),
        );
        // Allocate a mix of small and large chunks
        let sizes = [50, 1000, 100, 5000, 200, 10000, 300];
        let handles: Vec<_> = sizes
            .iter()
            .map(|&size| memory_management.reserve(size).unwrap())
            .collect();
        let usage_before = memory_management.memory_usage();
        // Deallocate every other allocation
        for i in (0..handles.len()).step_by(2) {
            drop(handles[i].clone());
        }
        // Reallocate similar sizes
        for &size in &sizes[0..sizes.len() / 2] {
            memory_management.reserve(size).unwrap();
        }
        let usage_after = memory_management.memory_usage();
        // Check that we haven't increased our memory usage significantly
        assert!(usage_after.bytes_reserved <= (usage_before.bytes_reserved as f64 * 1.1) as u64);
    }

    // Test pools without slices. More or less same as tests above.
    #[test_log::test]
    fn noslice_test_handle_mutability() {
        let mut memory_management = MemoryManagement::from_configuration(
            BytesStorage::default(),
            &(MemoryDeviceProperties {
                max_page_size: 128 * 1024 * 1024,
                alignment: 32,
            }),
            MemoryConfiguration::ExclusivePages,
            Arc::new(ServerLogger::default()),
            options(),
        );
        let handle = memory_management.reserve(10).unwrap();
        let other_ref = handle.clone();
        assert!(!handle.can_mut(), "Handle can't be mut when multiple ref.");
        drop(other_ref);
        assert!(handle.can_mut(), "Handle should be mut when only one ref.");
    }

    #[test_log::test]
    fn noslice_alloc_two_chunk() {
        let mut memory_management = MemoryManagement::from_configuration(
            BytesStorage::default(),
            &DUMMY_MEM_PROPS,
            MemoryConfiguration::Custom {
                pool_options: vec![MemoryPoolOptions {
                    pool_type: PoolType::ExclusivePages {
                        max_alloc_size: 1024,
                    },
                    dealloc_period: None,
                }],
            },
            Arc::new(ServerLogger::default()),
            options(),
        );

        let alloc_size = 512;
        let _handle = memory_management.reserve(alloc_size);
        let _new_handle = memory_management.reserve(alloc_size);

        let usage = memory_management.memory_usage();
        assert_eq!(usage.number_allocs, 2);
        assert_eq!(usage.bytes_in_use, alloc_size * 2);
        assert!(usage.bytes_reserved >= alloc_size * 2);
    }

    #[test_log::test]
    fn noslice_alloc_reuses_storage() {
        // If no storage is re-used, this will allocate two pages.
        let mut memory_management = MemoryManagement::from_configuration(
            BytesStorage::default(),
            &DUMMY_MEM_PROPS,
            MemoryConfiguration::Custom {
                pool_options: vec![MemoryPoolOptions {
                    pool_type: PoolType::ExclusivePages {
                        max_alloc_size: 1024,
                    },
                    dealloc_period: None,
                }],
            },
            Arc::new(ServerLogger::default()),
            options(),
        );

        let alloc_size = 512;
        let _handle = memory_management.reserve(alloc_size);
        drop(_handle);
        let _new_handle = memory_management.reserve(alloc_size);

        let usage = memory_management.memory_usage();
        assert_eq!(usage.number_allocs, 1);
        assert_eq!(usage.bytes_in_use, alloc_size);
        assert!(usage.bytes_reserved >= alloc_size);
    }

    #[test_log::test]
    fn noslice_alloc_allocs_new_storage() {
        let mut memory_management = MemoryManagement::from_configuration(
            BytesStorage::default(),
            &DUMMY_MEM_PROPS,
            MemoryConfiguration::Custom {
                pool_options: vec![MemoryPoolOptions {
                    pool_type: PoolType::ExclusivePages {
                        max_alloc_size: 1024,
                    },
                    dealloc_period: None,
                }],
            },
            Arc::new(ServerLogger::default()),
            options(),
        );

        let alloc_size = 768;
        let _handle = memory_management.reserve(alloc_size);
        let _new_handle = memory_management.reserve(alloc_size);
        let usage = memory_management.memory_usage();
        assert_eq!(usage.number_allocs, 2);
        assert_eq!(usage.bytes_in_use, alloc_size * 2);
        assert!(usage.bytes_reserved >= alloc_size * 2);
    }

    #[test_log::test]
    fn noslice_alloc_respects_alignment_size() {
        let mut memory_management = MemoryManagement::from_configuration(
            BytesStorage::default(),
            &MemoryDeviceProperties {
                max_page_size: DUMMY_MEM_PROPS.max_page_size,
                alignment: 50,
            },
            MemoryConfiguration::Custom {
                pool_options: vec![MemoryPoolOptions {
                    pool_type: PoolType::ExclusivePages {
                        max_alloc_size: 50 * 20,
                    },
                    dealloc_period: None,
                }],
            },
            Arc::new(ServerLogger::default()),
            options(),
        );
        let alloc_size = 40;
        let _handle = memory_management.reserve(alloc_size);
        let _new_handle = memory_management.reserve(alloc_size);
        let usage = memory_management.memory_usage();
        // Each slice should be aligned to 60 bytes, so 20 padding bytes.
        assert_eq!(usage.bytes_padding, 10 * 2);
    }

    #[test_log::test]
    fn noslice_allocs_on_correct_page() {
        let pools = [100, 200, 300, 400]
            .iter()
            .map(|&size| MemoryPoolOptions {
                pool_type: PoolType::SlicedPages {
                    page_size: size,
                    max_slice_size: size,
                },
                dealloc_period: None,
            })
            .collect();
        let mut memory_management = MemoryManagement::from_configuration(
            BytesStorage::default(),
            &MemoryDeviceProperties {
                max_page_size: DUMMY_MEM_PROPS.max_page_size,
                alignment: 10,
            },
            MemoryConfiguration::Custom {
                pool_options: pools,
            },
            Arc::new(ServerLogger::default()),
            options(),
        );
        // Allocate one thing on each page.
        let alloc_sizes = [50, 150, 250, 350];
        let _handles = alloc_sizes.map(|s| memory_management.reserve(s));
        let usage = memory_management.memory_usage();
        // Total memory should be size of all pages, and no more.
        assert_eq!(usage.bytes_in_use, alloc_sizes.iter().sum::<u64>());
    }

    #[test_log::test]
    fn noslice_allocate_deallocate_reallocate() {
        let mut memory_management = MemoryManagement::from_configuration(
            BytesStorage::default(),
            &MemoryDeviceProperties {
                max_page_size: 128 * 1024 * 1024,
                alignment: 32,
            },
            MemoryConfiguration::ExclusivePages,
            Arc::new(ServerLogger::default()),
            options(),
        );
        // Allocate a bunch
        let handles: Vec<_> = (0..5)
            .map(|i| memory_management.reserve(1000 * (i + 1)))
            .collect();
        let usage_before = memory_management.memory_usage();
        // Deallocate
        drop(handles);
        // Reallocate
        let _new_handles: Vec<_> = (0..5)
            .map(|i| memory_management.reserve(1000 * (i + 1)))
            .collect();
        let usage_after = memory_management.memory_usage();
        assert_eq!(usage_before.number_allocs, usage_after.number_allocs);
        assert_eq!(usage_before.bytes_in_use, usage_after.bytes_in_use);
        assert_eq!(usage_before.bytes_reserved, usage_after.bytes_reserved);
    }
}
