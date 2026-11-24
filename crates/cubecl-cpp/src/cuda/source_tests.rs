//! A kernel's emitted source is a function of the kernel: hosts that cache
//! compiled code key it on the source. No CUDA initialization.
use cubecl_core as cubecl;
use cubecl_core::{
    KernelSettings,
    compute::KernelBuilder,
    cube,
    ir::{AddressType, StorageType, UIntKind},
    prelude::*,
    server::ExecutionMode,
};
use cubecl_runtime::compiler::Compiler;

use crate::{
    cuda::{CudaDialect, mma::CudaWmmaCompiler},
    shared::CppCompiler,
};

/// Several vector types, pooled mutable locals of several types, and shared
/// memories that live together.
#[cube]
fn kernel<A: Size, B: Size>(c0: f32, d0: u32) {
    let mut shared_a = SharedMemory::<Vector<f32, A>>::new(8usize);
    let mut shared_b = SharedMemory::<u32>::new(16usize);
    let mut shared_c = SharedMemory::<Vector<f32, B>>::new(4usize);
    let mut a = Vector::<f32, A>::cast_from(1.0f32);
    let mut b = Vector::<f32, B>::cast_from(2.0f32);
    let mut c = c0;
    let mut d = d0;
    let mut e = false;
    while d < 64u32 {
        e = !e;
        if e {
            a += Vector::<f32, A>::cast_from(c);
            c = c * 2.0f32;
        } else {
            b += Vector::<f32, B>::cast_from(c);
        }
        d += 3u32;
    }
    shared_a[0] = a;
    shared_b[1] = d;
    shared_c[2] = b;
    sync_cube();
    a = shared_a[0];
    b = shared_c[2];
    shared_a[1] = a + Vector::<f32, A>::cast_from(shared_b[1]);
    shared_c[1] = b;
}

fn source() -> String {
    let mut builder = KernelBuilder::default();
    AddressType::U32.register(&mut builder.scope);
    let c0 = builder.scalar(f32::as_type_native_unchecked().storage_type());
    let d0 = builder.scalar(u32::as_type_native_unchecked().storage_type());
    kernel::expand::<Const<4>, Const<2>>(&mut builder.scope, c0.into(), d0.into());
    let definition = builder.build(KernelSettings::default());
    CppCompiler::<CudaDialect<CudaWmmaCompiler>>::default()
        .compile(
            definition,
            &Default::default(),
            ExecutionMode::Unchecked,
            AddressType::U32.unsigned_type(),
        )
        .unwrap()
        .to_string()
}

#[test]
fn the_same_kernel_is_the_same_source() {
    let first = source();
    for name in ["float_4", "float_2", "l_mut_", "shared_memory"] {
        assert!(first.contains(name), "the kernel must use {name}:\n{first}");
    }
    for _ in 0..16 {
        assert_eq!(source(), first);
    }
}
