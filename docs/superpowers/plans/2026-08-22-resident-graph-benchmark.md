# Resident Graph Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a JSON-producing resident-chain benchmark that compares eager, resident, and single-submit graph execution on UMA and discrete Vulkan devices.

**Architecture:** Reuse the standalone `vkssbo` Python API and its existing resident handle/graph ABI. Keep the benchmark process-isolated, use one fixed seed and shape for every path, and report wall-clock/error/memory cleanup evidence without changing default dispatch or async policy.

**Tech Stack:** Python 3, PyTorch CPU tensors, `vkssbo.py` ctypes bridge, Vulkan SSBO DLL, PowerShell/BAT test runners.

---

### Task 1: Add a failing benchmark contract

**Files:**
- Create: `D:/ComfyUI Vulkan beat/tools/vulkan_ssbo/test_resident_benchmark_contract.py`
- Test runner: `D:/ComfyUI Vulkan beat/tools/vulkan_ssbo/bench_resident_json.py`

- [ ] Write a subprocess contract test that runs the JSON runner with `--depth 4`, `--warmup 1`, and `--repeats 3`.
- [ ] Require `schema_version`, `device`, `shape`, `depth`, `warmup`, `repeats`, `paths`, and `cleanup` fields; require all three path records to include `median_ms`, `p95_ms`, `max_abs`, and `resident_entries_after`.
- [ ] Run the contract test before creating the runner and verify it fails because the runner is missing.

### Task 2: Implement the minimal JSON runner

**Files:**
- Create: `D:/ComfyUI Vulkan beat/tools/vulkan_ssbo/bench_resident_json.py`

- [ ] Parse depth, warmup, repeats, and positive vector length; initialize a fixed `torch.manual_seed(42)` input.
- [ ] Implement three functions using existing APIs: eager repeated `vk.elementwise(5, ...)`, `vk.resident(...).elementwise(5)`, and `vk.resident_graph().elementwise(...)` with one graph submit.
- [ ] Warm each path independently, then measure repeats with `time.perf_counter()` and compute median/p95.
- [ ] Compare each output with the CPU SiLU reference and record max absolute error.
- [ ] Record `vk.memory_stats()` before and after cleanup and terminate the Windows process after flushing JSON to avoid Vulkan DLL teardown races.

### Task 3: Make the contract green and preserve existing regression coverage

**Files:**
- Modify: `D:/ComfyUI Vulkan beat/tools/vulkan_ssbo/test_resident.py` only if the contract exposes an existing lifecycle defect.

- [ ] Run the contract test and verify it reports PASS with zero resident entries after cleanup.
- [ ] Run `test_resident.py` and `test_vulkan_ssbo_native_runtime.py` using the existing environment.
- [ ] Run `python -m py_compile` on both new Python files.

### Task 4: Add the benchmark to the discrete-GPU runner

**Files:**
- Modify: `D:/PyTorch for Vulkan/pytorch-src/run_discrete_gpu_test.bat`

- [ ] Add configurable `VKSSBO_RESIDENT_DEPTH`, `VKSSBO_RESIDENT_WARMUP`, and `VKSSBO_RESIDENT_REPEATS` defaults.
- [ ] Invoke the resident JSON runner once per process after the existing correctness and transfer benchmarks, writing stdout/stderr to `discrete_gpu_test.log`.
- [ ] Keep a nonzero exit code for contract/lifecycle failures, but do not change the default `VKSSBO_ASYNC_TRANSFER` policy.

### Task 5: Record evidence and stop before wheel work

**Files:**
- Modify: `D:/PyTorch for Vulkan/pytorch-src/VULKAN_SSBO_DEVELOPMENT_TASKBOOK_ZH.md`

- [ ] Record local UMA contract results and, when supplied, real RTX median/p95 and cleanup results.
- [ ] State explicitly whether resident graph reduces wall time and host/device submissions.
- [ ] Run `git diff --check`; do not build wheels.
