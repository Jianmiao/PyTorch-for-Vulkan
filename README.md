# PyTorch for Vulkan

A PyTorch fork developing a Vulkan storage-buffer (SSBO) backend to make
existing PyTorch applications usable across GPU vendors. Our goal is to give
users more choice of GPU without tying their applications to one vendor's
software ecosystem.

This project extends upstream PyTorch 2.13.0. Current development and local
validation focus on Windows and Intel integrated/UMA hardware; broader hardware
and operating-system coverage are goals, not established support.

[中文说明](README_ZH.md) | [Roadmap / 路线图](docs/vulkan-roadmap.md)

## Why Vulkan

Vulkan offers a common compute API across GPU vendors. We aim to reduce the
effort needed to run existing PyTorch applications on capable hardware,
including integrated GPUs, new discrete GPUs and older GPUs. Extending the
useful life of older hardware is one benefit of this work.

Vulkan support alone does not guarantee that a GPU, model or operator is
supported. Required device features, driver behavior, tensor semantics and
the actual execution path must be verified. Native execution, hybrid execution,
CPU fallback, unsupported cases and unverified cases must be distinguished.

## Project Goals and Priorities

- First, deliver practical single-device inference compatibility: correct
  tensor/operator behavior, predictable resource lifetimes and visible fallback.
- Reuse the same backend capabilities across ordinary PyTorch programs and
  application adapters such as ComfyUI and GPT-SoVITS.
- Expand the tested GPU-vendor and operating-system matrix incrementally,
  publishing the scope and evidence for each supported configuration.
- Keep model placement, unloading, prefetch and advanced attention strategies
  in application or optional plugin layers. Supported workloads must have a
  usable default path without requiring those plugins.

Full native training needs its own operator, backward and numerical acceptance.
Heterogeneous multi-GPU execution is a separate future research direction;
cross-vendor portability does not imply pooled VRAM or equal performance on
all GPUs. The project does not require applications to abandon working CUDA,
XPU or ROCm paths.

## Current Development Status

The October 2026 source snapshot is experimental compatibility work. It is not
a new wheel release or a declaration of complete PyTorch Vulkan support.
The checks below primarily cover local Windows/Intel UMA configurations; they
do not qualify every Vulkan GPU or other operating systems.

- Small FP32 native-buffer chains and constrained view/copy/stride semantics
  have passed local UMA checks. Full Storage, arbitrary tensor shapes and
  native backward remain incomplete.
- Public Vulkan SDPA currently uses CPU math and returns a Vulkan tensor. Its
  bounded FP32 inference contract supports mask, causal and GQA validation;
  it is not native attention acceleration.
- The MSVC OpenMP runtime selection and native Vulkan shutdown fixes are in
  this source tree. Existing portable installations and the older release
  wheel do not automatically include these changes.
- Companion ComfyUI/GPT-SoVITS adapters use a separate standalone SSBO bridge.
  Their small provider tests and GSV short-speech checks do not establish
  complete native attention, training support or end-to-end speed gains.

See [the current scope and validation record](docs/vulkan-source-status-20261016.md)
before choosing a runtime or interpreting the historical taskbook.

## What We Built

The main compute path uses Vulkan SSBOs instead of the stock texture-only path.
The backend is implemented in `aten/src/ATen/native/vulkan/ops/ssbo/` and wired
into the Vulkan operator registrations under `aten/src/ATen/native/vulkan/ops/`.

Key properties:

- GPU storage-buffer kernels with explicit Vulkan synchronization.
- GPU storage-buffer tensors for large inference workloads without texture
  dimension limits.
- FP16 weights and activations in supported SSBO inference paths; this is not
  a claim of general FP16 native-tensor coverage.
- Embedded SPIR-V shader binaries for the packaged build.
- Fallback to the stock Vulkan implementation where a compatible registered
  path exists; unsupported cases still need explicit handling.

Removing texture-layout constraints does not remove device memory limits or
the bounds of the current native-tensor implementation.

## Languages and Tooling

The custom backend and its supporting tools use the following languages. These
roles describe the current development source, not the language percentages of
the entire upstream-derived repository or the contents of every released wheel.

| Language / Tool | Role in the Vulkan SSBO Work |
| --- | --- |
| C++ | Vulkan runtime, buffers and resource lifetimes, resident caching, synchronization, dispatch, fusion, autotuning, and PyTorch/ATen integration |
| Rust | Memory-budget policy, allocation admission, cache limits, and eviction recommendations based on accounting snapshots |
| GLSL | Vulkan compute shaders, compiled into SPIR-V and embedded for deployment |
| Python | Backend configuration, shader build tools, regression tests, benchmarks, and companion ComfyUI adapters |
| Windows Batch | Windows build scripts, environment checks, and test launchers |
| CMake | Native build configuration and optional Rust static-library integration via Cargo |

Rust is implemented in `aten/src/ATen/native/vulkan/ops/ssbo/memory_policy/`.
The Vulkan build integrates it when Cargo is available; otherwise the C++
memory-policy fallback remains available. Rust receives accounting data only:
it does not own, allocate, or destroy Vulkan handles. C++ retains responsibility
for Vulkan resource lifetimes. Vulkan is the compute API, and SPIR-V is the
compiled shader format, not an additional source language.

## Operator Coverage

The SSBO implementation includes paths for these operator families used by our
ComfyUI workload:

- `linear`, `matmul`, and batched matrix multiplication
- 2D convolution
- `softmax`
- `layer_norm` and `group_norm`
- unary and binary elementwise operators
- copy, concat, shape operations, and upsample

The implementation is designed for GPU-heavy inference chains, keeping Vulkan
weights cached and avoiding the stock image-layout limits that affect large
texture-backed tensors.

This list describes implementation areas, not support for every shape, dtype,
layout or gradient combination. It also does not establish that an arbitrary
PyTorch program runs entirely on Vulkan; consult the verified scope above.

## Source Layout

| Path | Purpose |
| --- | --- |
| `aten/src/ATen/native/vulkan/ops/ssbo/` | SSBO runtime, Vulkan API, memory and dispatch code |
| `aten/src/ATen/native/vulkan/ops/ssbo/memory_policy/` | Rust budget policy, C interface, and Cargo configuration |
| `aten/src/ATen/native/vulkan/ops/ssbo/shaders/` | GLSL compute shader sources |
| `aten/src/ATen/native/vulkan/ops/` | Operator integration and fallback selection |
| `tools/gen_vulkan_spv.py` | Shader compilation and embedding support |
| `torch/vulkan_ssbo_configuration.py` | Python backend configuration |
| `test/*vulkan_ssbo*.py` | Integrated runtime regressions and benchmarks |
| `vulkan-build/` | Windows build scripts for CPython 3.10-3.15 |

The companion ComfyUI adapter and standalone `tools/vulkan_ssbo/` test/package
utilities are maintained in the separate ComfyUI workspace; that directory is
not part of this PyTorch source checkout.

## Prebuilt Wheel

This is an older Windows artifact. It does not contain all changes or
verification described in the October 2026 source snapshot, and should not be
used to infer the support status of the current development source.

The CPython 3.13 Windows x64 build is published as a GitHub Release asset, so
it can be downloaded directly without cloning the repository or using Git LFS:

[`torch-2.13.0a0+gitcf30153-cp313-cp313-win_amd64-SSBO.whl`](https://github.com/Jianmiao/PyTorch-for-Vulkan/releases/download/v2.13.0-vulkan-ssbo-cp313/torch-2.13.0a0+gitcf30153-cp313-cp313-win_amd64-SSBO.whl)

Install it directly with:

```powershell
python -m pip install https://github.com/Jianmiao/PyTorch-for-Vulkan/releases/download/v2.13.0-vulkan-ssbo-cp313/torch-2.13.0a0+gitcf30153-cp313-cp313-win_amd64-SSBO.whl
```

SHA-256: `3428B6703D1BDD20F3A363198FE473E2958467AFCCC607365DDB76E95A315988`

The wheel requires a Vulkan-capable Windows driver. The Vulkan SDK is needed
when rebuilding, but is not required for normal wheel use.

## Build From Source

The build scripts in `vulkan-build/` set the required Vulkan options:

```text
USE_VULKAN=1
USE_VULKAN_SHADERC_RUNTIME=1
USE_VULKAN_WRAPPER=0
USE_VULKAN_FP16_INFERENCE=1
USE_CUDA=0
USE_DISTRIBUTED=0
```

The scripts assume Visual Studio 2022 Build Tools, Ninja, CMake, a Vulkan SDK,
and the matching Python version. Adjust their local tool paths before running
them on another machine. The general build command is:

```powershell
python setup.py bdist_wheel
```

## Background

This fork is based on the upstream [PyTorch](https://github.com/pytorch/pytorch)
2.13.0 source tree; the custom Vulkan SSBO work above is the purpose of this
repository.

### Upstream and Project Contributions

PyTorch's Python API, ATen, dispatcher, autograd, stock backends, and other
framework components remain upstream work. This project's additions focus on
the SSBO backend, its memory policies, kernels, operator integration, and
supporting tools. Third-party dependencies retain their respective authorship
and licenses.

GitHub's contributor list reflects the repository's retained commit history,
including upstream contributors; it is not a roster of the Vulkan SSBO team.
The Languages panel summarizes the files recognized by GitHub Linguist across
the repository, not the implementation share of this custom backend. The table
above documents our technology choices without altering upstream attribution
or hiding upstream source from language statistics.

## License

This project follows the original PyTorch BSD-style license. The original
`LICENSE` and `NOTICE` files are retained unchanged, including the upstream
copyright and attribution notices.
