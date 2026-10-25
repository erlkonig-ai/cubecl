//! Host formatting regressions; no CUDA initialization or numerical oracle.
use super::*;
use crate::cuda::{CudaDialect, mma::CudaWmmaCompiler};
use cubecl_core::ir::ConstantValue;

type Cuda = CudaDialect<CudaWmmaCompiler>;

fn named(elem: Elem<Cuda>, width: usize, name: &'static str) -> Variable<Cuda> {
    Variable::Named { name, item: Item::new(elem, width, false) }
}

fn render(input: Variable<Cuda>, width: usize, kind: FP8Kind) -> String {
    struct Cast(Variable<Cuda>, Variable<Cuda>);
    impl fmt::Display for Cast {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            special_cast(f, &self.0, &self.1)
        }
    }
    // Kernel IR locals participate in optimization; Named variables do not.
    let out = Variable::LocalMut { id: 2, item: Item::new(Elem::FP8(kind), width, false) };
    format!("{}", Cast(input, out))
}

fn conversion_lines(code: &str) -> Vec<&str> {
    code.lines().filter(|line| line.contains("__nv_cvt_")).collect()
}

#[test]
fn scalar_integer_zero_broadcasts_to_packed_fp8() {
    for kind in [FP8Kind::E4M3, FP8Kind::E5M2] {
        for width in [2, 4, 16] {
            let input = Variable::Constant(ConstantValue::Int(0), Item::scalar(Elem::I32, true));
            let code = render(input, width, kind);
            let temporary = code.lines().find(|line| line.starts_with("__nv_bfloat16 _tmp_"))
                .expect("integer conversion must use a BF16 scalar temporary");
            assert!(temporary.ends_with("= __nv_bfloat16(int32(0));"), "{code}");
            let name = temporary.split_whitespace().nth(1).unwrap();
            let expected = format!("__nv_cvt_bfloat16raw2_to_fp8x2(__nv_bfloat162 {{ {name}, {name} }}, __NV_NOSAT, __NV_{kind:?})");
            let lines = conversion_lines(&code);
            assert_eq!(lines.len(), width / 2, "{code}");
            assert!(lines.iter().all(|line| line.contains(&expected)), "{code}");
        }
    }
}

#[test]
fn scalar_half_sources_broadcast_without_changing_value_or_policy() {
    // Named values may contain either signed zero or any NaN payload. Pair
    // constructors take the original half values directly, with no F32 cast,
    // arithmetic, saturation change, or lane swap in the formatter.
    for (elem, pair, intrinsic) in [
        (Elem::F16, "__half2", "halfraw2"),
        (Elem::BF16, "__nv_bfloat162", "bfloat16raw2"),
    ] {
        for name in ["negative_zero", "positive_nan", "negative_nan"] {
            for kind in [FP8Kind::E4M3, FP8Kind::E5M2] {
                for width in [2, 4, 16] {
                    let code = render(named(elem, 1, name), width, kind);
                    let expected = format!("__nv_cvt_{intrinsic}_to_fp8x2({pair} {{ {name}, {name} }}, __NV_NOSAT, __NV_{kind:?})");
                    let lines = conversion_lines(&code);
                    assert_eq!(lines.len(), width / 2, "{code}");
                    assert!(lines.iter().all(|line| line.contains(&expected)), "{code}");
                }
            }
        }
    }
}

#[test]
fn already_vector_half_sources_keep_pair_order() {
    for (elem, pair, intrinsic) in [
        (Elem::F16, "__half2", "halfraw2"),
        (Elem::BF16, "__nv_bfloat162", "bfloat16raw2"),
    ] {
        for kind in [FP8Kind::E4M3, FP8Kind::E5M2] {
            for width in [2, 4, 16] {
                let input = Variable::LocalMut { id: 1, item: Item::new(elem, width, false) };
                let code = render(input, width, kind);
                let lines = conversion_lines(&code);
                assert_eq!(lines.len(), width / 2, "{code}");
                for (i, line) in lines.iter().enumerate() {
                    let value = if width == 2 {
                        format!("reinterpret_cast<{pair} &>(l_mut_1)")
                    } else {
                        format!("(reinterpret_cast<{pair}_{} &>(l_mut_1)).i_{i}", width / 2)
                    };
                    let expected = format!("__nv_cvt_{intrinsic}_to_fp8x2({value}, __NV_NOSAT, __NV_{kind:?})");
                    assert!(line.contains(&expected), "{code}");
                }
                assert!(!code.contains(&format!("{pair} {{")), "must not repack existing pairs: {code}");
            }
        }
    }
}

#[test]
fn f32_sources_keep_broadcast_and_adjacent_lane_order() {
    for kind in [FP8Kind::E4M3, FP8Kind::E5M2] {
        for width in [2, 4, 16] {
            for input_width in [1, width] {
                let input = Variable::LocalMut { id: 1, item: Item::new(Elem::F32, input_width, false) };
                let code = render(input, width, kind);
                let lines = conversion_lines(&code);
                assert_eq!(lines.len(), width / 2, "{code}");
                for (i, line) in lines.iter().enumerate() {
                    let value = if input_width == 1 {
                        "float2 { l_mut_1, l_mut_1 }".to_owned()
                    } else {
                        format!("float2 {{ l_mut_1.i_{}, l_mut_1.i_{} }}", 2 * i, 2 * i + 1)
                    };
                    let expected = format!("__nv_cvt_float2_to_fp8x2({value}, __NV_NOSAT, __NV_{kind:?})");
                    assert!(line.contains(&expected), "{code}");
                }
            }
        }
    }
}

#[test]
fn scalar_fp8_conversions_stay_scalar() {
    for (elem, intrinsic) in [
        (Elem::I32, "bfloat16raw"), (Elem::F16, "halfraw"),
        (Elem::BF16, "bfloat16raw"), (Elem::F32, "float"),
    ] {
        for kind in [FP8Kind::E4M3, FP8Kind::E5M2] {
            let code = render(named(elem, 1, "source"), 1, kind);
            let lines = conversion_lines(&code);
            assert_eq!(lines.len(), 1, "{code}");
            assert!(lines[0].contains(&format!("__nv_cvt_{intrinsic}_to_fp8(")), "{code}");
            assert!(lines[0].contains(&format!(", __NV_NOSAT, __NV_{kind:?})")), "{code}");
            assert!(!code.contains("fp8x2") && !code.contains("__half2") && !code.contains("__nv_bfloat162"), "{code}");
        }
    }
}

#[test]
fn pair_helper_preserves_unpacked_half_lane_order() {
    // The helper is also used by other packed minifloat conversions. Cover
    // actual unpacked lanes directly, as well as the optimized FP8 path above.
    for (elem, pair) in [(Elem::F16, "__half2"), (Elem::BF16, "__nv_bfloat162")] {
        let input = named(elem, 4, "source");
        assert_eq!(float_to_packed(input, 0, 2), format!("{pair} {{ source.i_0, source.i_1 }}"));
        assert_eq!(float_to_packed(input, 1, 2), format!("{pair} {{ source.i_2, source.i_3 }}"));
        assert_eq!(float_to_packed(input, 1, 1), "source.i_1");
    }
}
