use crate::{
    memory_management::{BytesFormat, MemoryLocation, MemoryUsage},
    server::IoError,
    storage::{ComputeStorage, StorageUtilization},
};

use alloc::vec::Vec;
use cubecl_common::backtrace::BackTrace;

use super::{ManagedMemoryBinding, ManagedMemoryHandle, MemoryPool, Slice, calculate_padding};

/// A memory pool that allocates buffers in a range of sizes and reuses them to minimize allocations.
///
/// - Only one slice is supported per page, due to the limitations in WGPU where each buffer should only bound with
///   either read only or `read_write` slices but not a mix of both.
/// - The pool uses a ring buffer to efficiently manage and reuse pages.
pub struct ExclusiveMemoryPool {
    pages: Vec<MemoryPage>,
    pages_tmp: Vec<MemoryPage>,
    alignment: u64,
    dealloc_period: u64,
    last_dealloc_check: u64,
    max_alloc_size: u64,
    cur_avg_size: f64,
    location_base: MemoryLocation,
}

impl core::fmt::Display for ExclusiveMemoryPool {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_fmt(format_args!(
            " - Exclusive Pool max_alloc_size={}\n",
            BytesFormat::new(self.max_alloc_size)
        ))?;

        for page in self.pages.iter() {
            let is_free = page.slice.is_free();
            let size = BytesFormat::new(page.slice.effective_size());

            f.write_fmt(format_args!("   - Page {size} is_free={is_free}\n"))?;
        }

        if !self.pages.is_empty() {
            f.write_fmt(format_args!("\n{}\n", self.get_memory_usage()))?;
        }

        Ok(())
    }
}

const SIZE_AVG_DECAY: f64 = 0.01;

// How many times to find the allocation 'free' before deallocating it.
const ALLOC_AFTER_FREE: u32 = 5;

struct MemoryPage {
    slice: Slice,
    alloc_size: u64,
    free_count: u32,
}

impl ExclusiveMemoryPool {
    pub(crate) fn new(
        max_alloc_size: u64,
        alignment: u64,
        dealloc_period: u64,
        pool_pos: u8,
    ) -> Self {
        // Pages should be allocated to be aligned.
        assert_eq!(max_alloc_size % alignment, 0);

        Self {
            pages: Vec::new(),
            pages_tmp: Vec::new(),
            alignment,
            dealloc_period,
            last_dealloc_check: 0,
            max_alloc_size,
            cur_avg_size: max_alloc_size as f64 / 2.0,
            location_base: MemoryLocation::new(pool_pos, 0, 0),
        }
    }

    /// Finds a free page that can contain the given size
    /// Returns a slice on that page if successful.
    fn get_free_page(&mut self, size: u64) -> Option<&mut MemoryPage> {
        let mut best: Option<&mut MemoryPage> = None;
        for page in &mut self.pages {
            if page.alloc_size < size || !page.slice.is_free() {
                continue;
            }
            // A fitting free page at zero already has the lowest possible count.
            if page.free_count == 0 {
                return Some(page);
            }
            if best
                .as_ref()
                .is_none_or(|old| page.free_count < old.free_count)
            {
                best = Some(page);
            }
        }
        best
    }

    fn alloc_page<Storage: ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        size: u64,
    ) -> Result<(usize, &mut MemoryPage), IoError> {
        let alloc_size = (self.cur_avg_size as u64)
            .max(size)
            .next_multiple_of(self.alignment);

        let storage = storage.alloc(alloc_size)?;

        let padding = calculate_padding(size, self.alignment);
        let mut slice = Slice::new(storage, padding);

        // Return a smaller part of the slice. By construction, we only ever
        // get a page with a big enough size, so this is ok to do.
        slice.storage.utilization = StorageUtilization { offset: 0, size };
        slice.padding = padding;

        self.pages.push(MemoryPage {
            slice,
            alloc_size,
            // Start the allocation at 'almost ready to free'. Every use will decrement this.
            // This means allocations start as "suspected as unused" and over time will be kept for longer.
            free_count: ALLOC_AFTER_FREE - 1,
        });

        let idx = self.pages.len() - 1;
        Ok((idx, &mut self.pages[idx]))
    }
}

impl MemoryPool for ExclusiveMemoryPool {
    fn accept(&self, size: u64) -> bool {
        self.max_alloc_size >= size
    }

    /// Reserves memory of specified size using the reserve algorithm, and return
    /// a handle to the reserved memory.
    ///
    /// Also clean ups, merging free slices together if permitted by the merging strategy
    fn try_reserve(&mut self, size: u64) -> Option<ManagedMemoryHandle> {
        self.cur_avg_size =
            self.cur_avg_size * (1.0 - SIZE_AVG_DECAY) + size as f64 * SIZE_AVG_DECAY;

        let padding = calculate_padding(size, self.alignment);

        self.get_free_page(size).map(|page| {
            // Return a smaller part of the slice. By construction, we only ever
            // get a page with a big enough size, so this is ok to do.
            page.slice.storage.utilization = StorageUtilization { offset: 0, size };
            page.slice.padding = padding;
            page.free_count = page.free_count.saturating_sub(1);
            page.slice.handle.clone()
        })
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
        if size > self.max_alloc_size {
            return Err(IoError::BufferTooBig {
                size,
                backtrace: BackTrace::capture(),
            });
        }

        let (idx, page) = self.alloc_page(storage, size)?;
        let handle = page.slice.handle.clone();
        let mut location = self.location_base;
        location.page = idx as u16;
        handle.descriptor().update_location(location);

        Ok(handle)
    }

    fn get_memory_usage(&self) -> MemoryUsage {
        let used_slices: Vec<_> = self
            .pages
            .iter()
            .filter(|page| !page.slice.is_free())
            .collect();

        MemoryUsage {
            number_allocs: used_slices.len() as u64,
            bytes_in_use: used_slices
                .iter()
                .map(|page| page.slice.storage.size())
                .sum(),
            bytes_padding: used_slices.iter().map(|page| page.slice.padding).sum(),
            bytes_reserved: self.pages.iter().map(|page| page.alloc_size).sum(),
        }
    }

    fn cleanup<Storage: ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        alloc_nr: u64,
        explicit: bool,
    ) {
        // Check such that an alloc is free after at most dealloc_period.
        let check_period = self.dealloc_period / (ALLOC_AFTER_FREE as u64);

        if explicit || alloc_nr - self.last_dealloc_check >= check_period {
            self.last_dealloc_check = alloc_nr;

            for mut page in self.pages.drain(..) {
                if page.slice.is_free() {
                    page.free_count += 1;

                    // If free found is sufficiently high (ie. we've seen this alloc as free multiple times,
                    // without it being used in the meantime), deallocate it.
                    if page.free_count >= ALLOC_AFTER_FREE || explicit {
                        storage.dealloc(page.slice.storage.id);
                        continue;
                    }
                }

                let page_index = self.pages_tmp.len();
                page.slice
                    .handle
                    .descriptor()
                    .update_page(page_index as u16);
                self.pages_tmp.push(page);
            }

            core::mem::swap(&mut self.pages, &mut self.pages_tmp);
        }
    }

    fn bind(
        &mut self,
        old: ManagedMemoryHandle,
        new: ManagedMemoryHandle,
        cursor: u64,
    ) -> Result<(), IoError> {
        let id_old = old.descriptor();
        let page = &mut self.pages[id_old.page()];
        new.descriptor().update_location(id_old.location());

        page.slice.handle = new;
        page.slice.cursor = cursor;

        Ok(())
    }

    fn find(&self, binding: &ManagedMemoryBinding) -> Result<&Slice, IoError> {
        let binding_descriptor = binding.descriptor();
        let page_index = binding_descriptor.page();

        let page = self
            .pages
            .get(page_index)
            .ok_or_else(|| IoError::NotFound {
                backtrace: BackTrace::capture(),
                reason: alloc::format!("Memory page {} doesn't exist", page_index).into(),
            })?;

        Ok(&page.slice)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::BytesStorage;

    #[test]
    fn reuse_selects_the_first_fitting_free_page_of_minimum_count() {
        let mut pool = ExclusiveMemoryPool::new(256, 1, 100, 0);
        let mut storage = BytesStorage::default();
        let small = pool.alloc(&mut storage, 32).unwrap();
        let live = pool.alloc(&mut storage, 256).unwrap().binding();
        let first = pool.alloc(&mut storage, 192).unwrap();
        let second = pool.alloc(&mut storage, 256).unwrap();
        drop((small, first, second));

        pool.pages[0].free_count = 0; // Too small for the requests below.
        pool.pages[1].free_count = 0; // Still held by a binding.
        for (first_count, second_count, expected) in [(3, 2, 3), (2, 2, 2), (2, 0, 3), (0, 0, 2)] {
            pool.pages[2].free_count = first_count;
            pool.pages[3].free_count = second_count;
            let handle = pool.try_reserve(192).unwrap();
            assert_eq!(handle.descriptor().page(), expected);
        }

        let last = pool.try_reserve(256).unwrap();
        assert!(pool.try_reserve(256).is_none());
        drop(live);
        assert_eq!(pool.try_reserve(256).unwrap().descriptor().page(), 1);
        drop(last);
    }
}
