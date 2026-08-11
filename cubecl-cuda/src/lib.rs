#[macro_use]
extern crate derive_new;
extern crate alloc;

mod compute;
mod device;
mod runtime;

pub use device::*;
pub use runtime::*;

#[cfg(feature = "ptx-wmma")]
pub(crate) type WmmaCompiler = cubecl_cpp::cuda::mma::PtxWmmaCompiler;

#[cfg(not(feature = "ptx-wmma"))]
pub(crate) type WmmaCompiler = cubecl_cpp::cuda::mma::CudaWmmaCompiler;

pub mod install {
    use std::path::PathBuf;

    pub fn include_path() -> PathBuf {
        let mut path = cuda_path().expect("
        CUDA installation not found.
        Please ensure that CUDA is installed and the CUDA_PATH environment variable is set correctly.
        Note: Default paths are used for Linux (/usr/local/cuda) and Windows (C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/), which may not be correct.
    ");
        path.push("include");
        path
    }

    pub fn cccl_include_path() -> PathBuf {
        let mut path = include_path();
        path.push("cccl");
        path
    }

    pub fn cuda_path() -> Option<PathBuf> {
        if let Ok(path) = std::env::var("CUDA_PATH") {
            return Some(PathBuf::from(path));
        }

        #[cfg(target_os = "linux")]
        {
            // If it is installed as part of the distribution
            return if std::fs::exists("/usr/local/cuda").is_ok_and(|exists| exists) {
                Some(PathBuf::from("/usr/local/cuda"))
            } else if std::fs::exists("/opt/cuda").is_ok_and(|exists| exists) {
                Some(PathBuf::from("/opt/cuda"))
            } else if std::fs::exists("/usr/bin/nvcc").is_ok_and(|exists| exists) {
                // Maybe the compiler was installed within the user path.
                Some(PathBuf::from("/usr"))
            } else {
                None
            };
        }

        #[cfg(target_os = "windows")]
        {
            return Some(PathBuf::from(
                "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/",
            ));
        }

        #[allow(unreachable_code)]
        None
    }
}

#[cfg(test)]
#[allow(unexpected_cfgs)]
mod tests {
    pub type TestRuntime = crate::CudaRuntime;

    pub use half::{bf16, f16};

    cubecl_core::testgen_all!(f32: [f16, bf16, f32, f64], i32: [i8, i16, i32, i64], u32: [u8, u16, u32, u64]);
    cubecl_std::testgen!();
    cubecl_std::testgen_tensor_identity!([f16, bf16, f32, u32]);
    cubecl_std::testgen_quantized_view!(f16);
}

/// Can this device read ordinary host memory (an `mmap`, a `Vec`) directly from
/// a kernel, so a buffer can be *aliased* instead of copied?
///
/// This is `cudaDevAttrPageableMemoryAccess`. It is true on integrated and
/// Grace-class parts, where the GPU walks the host page tables
/// (`…UsesHostPageTables`) and `cudaHostGetDevicePointer` returns the host
/// address unchanged; it is false on ordinary discrete cards, which must copy
/// across PCIe.
///
/// Callers that want a fallback should ask BEFORE calling
/// [`cubecl_runtime::client::ComputeClient::register_external_aliased`], which
/// asserts this rather than degrading quietly.
pub fn supports_zero_copy_host(device_index: usize) -> bool {
    use core::sync::atomic::{AtomicU8, Ordering};
    // 0 = unknown, 1 = no, 2 = yes. Queried once; the answer is a property of
    // the silicon and cannot change during a run.
    static CACHE: [AtomicU8; 16] = [const { AtomicU8::new(0) }; 16];
    let slot = CACHE.get(device_index);
    if let Some(c) = slot {
        match c.load(Ordering::Relaxed) {
            1 => return false,
            2 => return true,
            _ => {}
        }
    }
    // SAFETY: querying a device attribute has no side effects and does not
    // require a current context.
    let ok = unsafe {
        cudarc::driver::result::device::get(device_index as i32)
            .and_then(|dev| {
                cudarc::driver::result::device::get_attribute(
                    dev,
                    cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS,
                )
            })
            .map(|v| v != 0)
            .unwrap_or(false)
    };
    if let Some(c) = slot {
        c.store(if ok { 2 } else { 1 }, Ordering::Relaxed);
    }
    ok
}
