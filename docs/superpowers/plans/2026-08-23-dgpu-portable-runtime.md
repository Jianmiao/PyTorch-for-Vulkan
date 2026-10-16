# dGPU Portable Runtime Package Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (or equivalent inline execution) to implement this plan task-by-task.

**Goal:** Turn the existing discrete-GPU one-shot test package into a self-contained Windows x64 package with a relocatable Python 3.13 runtime and the matching custom PyTorch Vulkan environment.

**Architecture:** Keep the existing SSBO scripts, DLL, shaders, tests, and benchmark entry points unchanged in their current package boundaries. Add `runtime\` as a copied Python installation, make BAT files prefer that interpreter while preserving environment-variable overrides, add a package-local preflight, then validate the copied runtime with the existing discrete-policy simulation before producing a static SHA-256 manifest.

**Tech Stack:** Windows x64 CPython 3.13, custom PyTorch Vulkan build, NumPy and Python runtime dependencies, batch entry points, PowerShell file hashing.

---

### Task 1: Make package entry points runtime-relative

**Files:**
- Modify: `E:\PyTorch for Vulkan SSBO 离散型GPU一次性性能测试整合包\run_discrete_gpu_performance_once.bat`
- Modify: `E:\PyTorch for Vulkan SSBO 离散型GPU一次性性能测试整合包\run_local_discrete_simulation_check.bat`

- [ ] Prefer `%ROOT%runtime\python.exe` when `VKSSBO_PYTHON` is unset.
- [ ] Keep `VKSSBO_PYTHON` and `VKSSBO_TORCH_PYTHON` overrides for target-machine diagnostics.
- [ ] Remove the local absolute `D:\ComfyUI Vulkan beat\tools\Python313\python.exe` default from the simulation wrapper.

### Task 2: Add package environment preflight

**Files:**
- Create: `E:\PyTorch for Vulkan SSBO 离散型GPU一次性性能测试整合包\preflight_environment.bat`

- [ ] Verify the package-local Python exists and imports Torch and NumPy.
- [ ] Verify the package-local SSBO DLL and shader directory exist.
- [ ] Print Torch version, Python version, and the package root without requiring a GPU.
- [ ] Exit nonzero on missing runtime components.

### Task 3: Copy the complete runtime

**Files:**
- Create: `E:\PyTorch for Vulkan SSBO 离散型GPU一次性性能测试整合包\runtime\...`

- [ ] Copy the complete `D:\ComfyUI Vulkan beat\tools\Python313` tree, including Python, site-packages, Torch DLLs, and dependencies.
- [ ] Confirm the copied interpreter works from the package path.

### Task 4: Refresh documentation and manifest

**Files:**
- Modify: `E:\PyTorch for Vulkan SSBO 离散型GPU一次性性能测试整合包\README_ZH.md`
- Replace: `E:\PyTorch for Vulkan SSBO 离散型GPU一次性性能测试整合包\MANIFEST_SHA256.txt`

- [ ] Document that the package includes Python 3.13 and custom PyTorch Vulkan, but still requires target-machine Vulkan/NVIDIA drivers.
- [ ] State expected size and that generated logs are excluded from the static manifest.
- [ ] Hash all static files after copying the runtime.

### Task 5: Verify the relocated package

- [ ] Run `preflight_environment.bat` using only `runtime\python.exe`.
- [ ] Run `run_local_discrete_simulation_check.bat` with a deliberately pre-set async variable and confirm the wrapper resets it.
- [ ] Confirm final `RESULT: PASS`, 14/14 Torch, INT4/FP8/FP4 contract passes, and benchmark exit codes of zero.
- [ ] Recompute manifest and report the uncompressed package size.
