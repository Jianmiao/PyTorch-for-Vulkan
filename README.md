# PyTorch for Vulkan

A Windows-focused PyTorch fork with our custom Vulkan storage-buffer (SSBO)
backend. Upstream PyTorch 2.13.0 is only the build base; the work in this
repository is the Vulkan implementation and its integration into PyTorch.

## Why Vulkan

If your GPU supports Vulkan, you can use this PyTorch build on it. The project
was created to make local AI tools usable on older GPUs that support Vulkan but
do not support XPU, CUDA, or ROCm, so those workloads do not have to fall back
to the CPU.

## What We Built

The main compute path uses Vulkan SSBOs instead of the stock texture-only path.
The backend is implemented in `aten/src/ATen/native/vulkan/ops/ssbo/` and wired
into the Vulkan operator registrations under `aten/src/ATen/native/vulkan/ops/`.

Key properties:

- GPU storage-buffer kernels with explicit Vulkan synchronization.
- GPU storage-buffer tensors for large inference workloads without texture
  dimension limits.
- FP16 inference support for weights and activations where supported.
- Embedded SPIR-V shader binaries for the packaged build.
- Official Vulkan fallback for operators that are not yet covered by SSBO.

## Operator Coverage

The current SSBO path covers the core operators used by our ComfyUI workload:

- `linear`, `matmul`, and batched matrix multiplication
- 2D convolution
- `softmax`
- `layer_norm` and `group_norm`
- unary and binary elementwise operators
- copy, concat, shape operations, and upsample

The implementation is designed for GPU-heavy inference chains, keeping Vulkan
weights cached and avoiding the stock image-layout limits that affect large
texture-backed tensors.

## Source Layout

| Path | Purpose |
| --- | --- |
| `aten/src/ATen/native/vulkan/ops/ssbo/` | SSBO runtime, Vulkan API, memory and dispatch code |
| `aten/src/ATen/native/vulkan/ops/` | Operator integration and fallback selection |
| `tools/gen_vulkan_spv.py` | Shader compilation and embedding support |
| `vulkan-build/` | Windows build scripts for CPython 3.10-3.13 |

## Prebuilt Wheel

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

## License

This project follows the original PyTorch BSD-style license. The original
`LICENSE` and `NOTICE` files are retained unchanged, including the upstream
copyright and attribution notices.
