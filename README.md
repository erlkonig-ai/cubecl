# MaryML CubeCL release fork

This is a renamed, source-preserving distribution of upstream [CubeCL](https://github.com/tracel-ai/cubecl), not an upstream release. The upstream authors, copyrights, license texts, implementation, and existing Rust library names are retained. Package names carry the `maryml-` prefix; dependency keys remain unchanged so existing Rust imports and generated paths keep their original names.

Fork repository: https://github.com/erlkonig-ai/cubecl. This workspace deliberately combines the exact executed package sources; it is not a checkout of one upstream revision.

| Package | Version | Rust library | Input |
|---|---|---|---|
| `maryml-cubecl` | 0.10.0 | `cubecl` | registry archive |
| `maryml-cubecl-core` | 0.10.0 | `cubecl_core` | registry archive |
| `maryml-cubecl-cpp` | 0.10.0 | `cubecl_cpp` | registry archive |
| `maryml-cubecl-cpu` | 0.10.0 | `cubecl_cpu` | registry archive |
| `maryml-cubecl-cuda` | 0.10.0 | `cubecl_cuda` | 1fc64da1 |
| `maryml-cubecl-hip` | 0.10.0 | `cubecl_hip` | registry archive |
| `maryml-cubecl-macros` | 0.10.0 | `cubecl_macros` | 8b2aa74e |
| `maryml-cubecl-opt` | 0.10.0 | `cubecl_opt` | registry archive |
| `maryml-cubecl-runtime` | 0.10.0 | `cubecl_runtime` | 1fc64da1 |
| `maryml-cubecl-spirv` | 0.10.0 | `cubecl_spirv` | registry archive |
| `maryml-cubecl-std` | 0.10.0 | `cubecl_std` | registry archive |
| `maryml-cubecl-wgpu` | 0.10.0 | `cubecl_wgpu` | 1fc64da1 |

## Source and dependency contract

`SOURCE_PROVENANCE.json` records the exact registry archive checksums or Git commits, original package manifests, and imported Git trees. Registry sources use their normalized manifests. `Cargo.toml.orig`, per-package `Cargo.lock`, and `.cargo_vcs_info.json` are retained upstream evidence, not the fork's resolved dependency graph. No root lockfile has been resolved. Build outputs and cache completion markers are excluded.

The unchanged registry dependencies `cubecl-common`, `cubecl-ir`, `cubecl-macros-internal`, and `cubecl-zspace` remain upstream 0.10.0. The macros from 8b2aa74e generate `cubecl::...`; keep the `cubecl` dependency alias. Arbitrary aliases are not newly supported. The compiler implementations are unchanged, including the registry cubecl-cpp Metal Log1p lowering; no experimental math patch is present.

## Validation status

Source-only preparation: TOML/alias/path/provenance checks only. No Cargo resolution, package verification, build, tests, device run, registry publication, or release tag has been executed. Public crate/version approval and the resolved publish-order check remain separate.
