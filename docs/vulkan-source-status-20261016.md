# Vulkan source snapshot - 2026-10-16

This update publishes the accumulated Vulkan development source, Rust policy,
shader sources/embedded headers, configuration and regression tools. It does
not publish a wheel, replace an installed runtime or change the supported
scope to full PyTorch compatibility.

## Verified scope

The following are local verification results from 2026-10-11, not a fresh
full release qualification on 2026-10-16:

| Area | Verified behavior | Boundary |
| --- | --- | --- |
| Native buffer B1 | CPU upload, bounded FP32 SiLU/Sigmoid chain, download; no intermediate CPU copies | Small positive-size tensors only |
| B2 views and copies | Selected view/alias/detach, positive strides, transpose/permute/slice offset, gather/scatter and version/lifetime checks | Not arbitrary strides or full Storage |
| B3 capabilities | Capability reporting and safe functional-copy boundaries | Vulkan tensors remain opaque |
| Public SDPA | FP32 rank-4 inference, bool/float masks, causal and GQA checks | CPU math fallback with Vulkan output; no native backward |
| CPU parallel runtime | 204 numerical checks covering softmax/log-softmax/backward, reductions and layer norm | Local MSVC x64; other platforms not qualified |
| Vulkan shutdown | Bounded import/query/empty/pending/cycle/global/error exit checks | Requires the matching source-built runtime |
| Companion provider | Tiny optional standalone UMA attention calls and baseline fallback | Separate application workspaces; not the public Vulkan SDPA implementation |
| GSV short speech | CPU/standalone hybrid pair exited normally; identical semantic tokens; sample correlation 0.999994, SNR 49.40 dB | Attention option disabled; not a performance or perceptual-quality acceptance |

The standalone speech pair exercised 194 Linear and 91 Conv1d calls. Standard
GSV `infer_panel_*` still uses TorchScript T2SBlock, bypassing the optional
eager MHA injection. The narrow provider delegates masks to its baseline.

## Runtime boundaries

The OpenMP fix pairs MSVC `/openmp` or `/openmp:experimental` with VCOMP and
preserves explicit `/openmp:llvm` behavior. Previously mixing VCOMP workers
with LLVM `omp_*` queries produced incorrect CPU numerical results.

The source tree contains the correction. Production portable Torch has not
been migrated. A future migration must keep Python ABI, Torch Python package,
`torch_cpu`, `torch_python`, `torch`, `c10` and extension dependencies compatible;
copying a single DLL is not a supported deployment procedure. In particular,
the GSV Python 3.9 runtime cannot consume CPython 3.13 extension binaries.

## Remaining work

- Complete the general tensor/storage contract and native backward before
  claiming general training support.
- Design and validate the optional GSV T2S attention boundary with masks and
  KV caches, preserving the default TorchScript path.
- Validate complete ComfyUI loading, generation, cancellation and unloading
  with a matched isolated runtime before production migration.
- Perform the deferred real dGPU and 1080P speed/memory acceptance separately.

## Reproducibility and source layout

The source-upload checks on 2026-10-16 passed: 11 Rust unit tests, 204 CPU
parallel numerical checks against NumPy, syntax parsing of 22 Python files,
the configuration contract (52 entries), and staged whitespace checks.
No full PyTorch rebuild, GPU workload or wheel build was performed for this
upload. These focused checks do not constitute a fresh release qualification.

Native code, integrated tests and the Rust policy are included in this
repository. GLSL sources and embedded C++ shader headers are both included;
loose compiler outputs, DLLs, models, datasets and local backup files are not.

The standalone bridge, optional provider and application adapters live in
separate local workspaces and are not included in this source-only upload.
`run_uma_test.bat` and `run_discrete_simulation_test.bat` are historical
integration runners that need those companion scripts via `VKSSBO_TEST_ROOT`.
They are not a self-contained test suite for a clean clone of this repository.
The older `vulkan-build/gen_vulkan_lib.bat` helper also references a local
external `gen_def.py`; that helper is not a portable clean-clone build entry.

Source-tree probes live under `test/test_vulkan_native_buffer_*.py`,
`test/test_vulkan_sdpa_basic.py`, `test/test_vulkan_process_exit.py` and
`test/test_cpu_parallel_runtime.py`. Run with the matching built Python
package; GPU probes require deliberate `--run` selection. The Rust policy can
be tested independently with `cargo test --locked --offline --manifest-path
aten/src/ATen/native/vulkan/ops/ssbo/memory_policy/Cargo.toml`.

Historical development notes contain local machine paths and earlier results.
They do not override the limits in this document or imply the older published
wheel includes these source changes.
