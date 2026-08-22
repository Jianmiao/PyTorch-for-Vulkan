# FP4 Weight-Only Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a CPU-reference FP4 E2M1 weight-only contract with deterministic nibble packing, explicit FP32 fallback, and UMA/discrete-simulation compatibility coverage.

**Architecture:** Keep FP4 isolated in `fp4_weight_only.py`; it owns the fixed E2M1 codebook, per-32-element scales, packing, validation, decode, and fallback. A subprocess-isolated runtime test exercises CPU Linear accuracy, odd-column padding, SSBO bridge behavior, invalid metadata, and resident cleanup. Existing BAT runners call this contract test without changing Vulkan capabilities, shaders, resident storage, async policy, or performance paths.

**Tech Stack:** Python 3.13, PyTorch CPU tensors, existing `vkssbo.py` ctypes bridge, Windows BAT runners, Markdown taskbook.

---

## File Map

- Create `D:/ComfyUI Vulkan beat/tools/vulkan_ssbo/fp4_weight_only.py`: frozen `Fp4WeightOnly` payload, E2M1 codebook, pack/decode/validation/fallback.
- Create `D:/ComfyUI Vulkan beat/tools/vulkan_ssbo/test_fp4_weight_only_contract_runtime.py`: child-process contract and SSBO bridge regression.
- Modify `D:/PyTorch for Vulkan/pytorch-src/run_uma_test.bat`: call the FP4 contract after FP8.
- Modify `D:/PyTorch for Vulkan/pytorch-src/run_discrete_simulation_test.bat`: call the FP4 simulation after FP8.
- Modify `D:/PyTorch for Vulkan/pytorch-src/VULKAN_SSBO_DEVELOPMENT_TASKBOOK_ZH.md`: record the FP4 contract and actual results.

Do not modify Vulkan C++, Rust, device capability fields, shaders, pipeline setup, resident storage, async defaults, or wheel build scripts.

### Task 1: Write the failing FP4 contract test

**Files:** Create `D:/ComfyUI Vulkan beat/tools/vulkan_ssbo/test_fp4_weight_only_contract_runtime.py`.

- [ ] **Step 1: Add isolated parent/child entry points.**

Copy the established INT4/FP8 runtime-test structure: `main()` launches `--child` with `subprocess.run`, the child imports `vkssbo` and `fp4_weight_only`, and the child ends through the existing Windows `TerminateProcess` helper after flushing the final JSON line. Set `VKSSBO_ACT_FP16=0` in the parent environment.

- [ ] **Step 2: Add payload shape and metadata assertions.**

With `torch.manual_seed(47)` and a finite `[7,35]` FP32 weight, call `pack_weight(weight)`. Assert `format_version=1`, `scheme="e2m1"`, `scale_granularity="per_output_channel_block"`, `scale_mode="max_abs_to_finite"`, `nibble_order="low_even_high_odd"`, `block_size=32`, `in_features=35`, qweight `torch.uint8` shape `(7,18)`, scale `torch.float32` shape `(7,2)`, and both tensors contiguous.

- [ ] **Step 3: Add the fixed E2M1 decode and packing tests.**

Use a one-row manual payload with qweight `[[0x10, 0x32, 0x54, 0x76]]`, scale `[[1.0]]`, and `in_features=8`; require decoded values `[0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]`, mode `fp4_weight_only`, and reason `supported_e2m1`. Add a negative-code case, an odd `in_features=3` case, and a zero block case requiring scale `1.0` and decoded zero values. Verify the final byte of a packed odd-column payload has a zero high nibble.

- [ ] **Step 4: Add CPU Linear error gates.**

For shapes `(128,256)`, `(1024,1280)`, and `(4096,4096)`, use fixed random inputs and weights, resolve the packed payload, and calculate the relative L2 error between `torch.nn.functional.linear(input, weight)` and the decoded-weight result. Require every row to be `< 0.12`; append `{shape, relative_l2}` rows for the final JSON. Include a second deterministic large-range weight sample to prove scaling remains finite.

- [ ] **Step 5: Add SSBO bridge and fallback coverage.**

Pack a finite `[7,11]` bridge weight, resolve it, and compare `vk.linear` against `torch.nn.functional.linear` with `max_abs < 3e-2`. Create invalid payloads for metadata, dtype, shape, scale, and odd-column padding; resolve each to `mode="fp32_fallback"`, require the stable reason and contiguous FP32 output, send the fallback weight through `vk.linear`, and require the same `3e-2` bridge gate. End with `vk.memory_stats()` and require `resident_entries == 0`.

- [ ] **Step 6: Add invalid-source and JSON assertions.**

For both NaN and Inf source weights, require `pack_weight` to raise `ValueError("nonfinite_weight")`. The child JSON must contain `schema_version=1`, `format_version=1`, `scheme="e2m1"`, `error_rows`, `supported_max_abs`, `fallback_reasons`, and `resident_entries`. The parent must parse only the last JSON line and print `FP4 weight-only format/fallback regression: PASS` with the maximum error and `fallback=fp32`.

- [ ] **Step 7: Run the new test before implementation.**

Run:

```powershell
& 'D:/ComfyUI Vulkan beat/tools/Python313/python.exe' 'D:/ComfyUI Vulkan beat/tools/vulkan_ssbo/test_fp4_weight_only_contract_runtime.py'
```

Expected result: a child failure caused by the missing `fp4_weight_only` module, not a Vulkan initialization or Python-environment failure.

- [ ] **Step 8: Preserve the red-test evidence.**

```powershell
Get-FileHash 'D:/ComfyUI Vulkan beat/tools/vulkan_ssbo/test_fp4_weight_only_contract_runtime.py' -Algorithm SHA256
```

The runtime test root is a separate non-Git workspace, so do not stage it with the PyTorch source repository. Keep the test in place and preserve the required timestamped backup.

### Task 2: Implement the CPU E2M1 contract

**Files:** Create `D:/ComfyUI Vulkan beat/tools/vulkan_ssbo/fp4_weight_only.py`.

- [ ] **Step 1: Define the frozen payload and constants.**

Define `BLOCK_SIZE=32`, `FORMAT_VERSION=1`, `SCHEME="e2m1"`, `SCALE_GRANULARITY="per_output_channel_block"`, `SCALE_MODE="max_abs_to_finite"`, `NIBBLE_ORDER="low_even_high_odd"`, and `MAX_FINITE=6.0`. Define frozen dataclass `Fp4WeightOnly(qweight, scale, in_features, block_size=32, format_version=1, scheme="e2m1", scale_granularity="per_output_channel_block", scale_mode="max_abs_to_finite", nibble_order="low_even_high_odd")`.

- [ ] **Step 2: Build the fixed 16-code decode table.**

Decode sign, two-bit exponent, and one-bit mantissa using E2M1 bias 1. Produce a float32 table where codes `0x0..0x7` are `[0, 0.5, 1, 1.5, 2, 3, 4, 6]` and codes `0x8..0xf` are their signed counterparts. Mark every code finite; keep a parallel mantissa-bit/code table for deterministic tie resolution.

- [ ] **Step 3: Implement deterministic nearest-value encoding.**

Reject non-finite input with `ValueError("nonfinite_weight")`. For each finite normalized value, compare its absolute value with the eight non-negative finite levels; choose the minimum absolute distance, then prefer a candidate whose mantissa bit is zero, then the smaller code. Apply the sign bit, emit code `0x00` for zero, and return uint8 codes.

- [ ] **Step 4: Implement per-block packing.**

Accept only a two-dimensional floating-point tensor, convert to contiguous FP32, and reject non-finite values. For each output row and 32-column block compute `max_abs / 6.0`, use `1.0` for an all-zero block, encode the normalized block, and pack even columns into low nibbles and odd columns into high nibbles. Initialize the unused high nibble of an odd-column final byte to zero. Return contiguous qweight shape `[rows, ceil(cols/2)]` and scale shape `[rows, ceil(cols/32)]`.

- [ ] **Step 5: Implement strict validation and decode.**

Validate payload type, version, scheme, metadata, positive integer `in_features`, original weight type/rank/dtype/shape, qweight type/rank/dtype/contiguity/shape, scale type/rank/dtype/contiguity/shape, finite positive scale, and odd-column padding nibble. Return stable reasons such as `payload_type`, `format_version`, `scheme`, `scale_granularity`, `scale_mode`, `nibble_order`, `block_size`, `qweight_dtype`, `qweight_shape`, `scale_dtype`, `scale_shape`, `nonfinite_scale`, `nonpositive_scale`, `padding_nibble`, and `original_weight_shape`. On success, unpack low/high nibbles, decode through the table, multiply per-block scale, and return contiguous FP32 with `("fp4_weight_only", "supported_e2m1")`. On failure return the original weight as contiguous FP32 with `("fp32_fallback", reason)`.

- [ ] **Step 6: Run compile and contract verification.**

Run:

```powershell
& 'D:/ComfyUI Vulkan beat/tools/Python313/python.exe' -m py_compile 'D:/ComfyUI Vulkan beat/tools/vulkan_ssbo/fp4_weight_only.py' 'D:/ComfyUI Vulkan beat/tools/vulkan_ssbo/test_fp4_weight_only_contract_runtime.py'
& 'D:/ComfyUI Vulkan beat/tools/Python313/python.exe' 'D:/ComfyUI Vulkan beat/tools/vulkan_ssbo/test_fp4_weight_only_contract_runtime.py'
```

Expected result: compile succeeds and the parent prints `FP4 weight-only format/fallback regression: PASS` with maximum relative L2 below `0.12`, bridge error below `0.03`, and `resident_entries=0`.

- [ ] **Step 7: Record the green contract files.**

```powershell
Get-FileHash 'D:/ComfyUI Vulkan beat/tools/vulkan_ssbo/fp4_weight_only.py','D:/ComfyUI Vulkan beat/tools/vulkan_ssbo/test_fp4_weight_only_contract_runtime.py' -Algorithm SHA256
```

These runtime files are outside the PyTorch source Git repository; do not stage unrelated source files.

### Task 3: Add the compatibility entry points

**Files:** Modify `D:/PyTorch for Vulkan/pytorch-src/run_uma_test.bat` and `D:/PyTorch for Vulkan/pytorch-src/run_discrete_simulation_test.bat`.

- [ ] **Step 1: Back up both BAT files before editing.**

Create `D:/ComfyUI Vulkan beat/backups/fp4-integration-YYYYMMDD-HHmmss/` and copy both BAT files into it.

- [ ] **Step 2: Add the UMA call after FP8.**

Insert:

```bat
if exist "%VKSSBO_TEST_ROOT%\test_fp4_weight_only_contract_runtime.py" call :run_required "FP4 weight-only contract" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_fp4_weight_only_contract_runtime.py"
```

Do not alter device selection, `VKSSBO_SIMULATE_DISCRETE`, async defaults, or benchmark sections.

- [ ] **Step 3: Add the discrete simulation call after FP8.**

Insert the equivalent line with label `FP4 weight-only simulation`. Preserve forced discrete policy, staging, virtual budget, timeline, and the no-performance-benchmark boundary.

- [ ] **Step 4: Run both focused entry points.**

Run:

```powershell
& 'D:/PyTorch for Vulkan/pytorch-src/run_uma_test.bat'
& 'D:/PyTorch for Vulkan/pytorch-src/run_discrete_simulation_test.bat'
```

Expected result: both exit with code 0, both logs include the FP4 PASS line, and both end with `RESULT: PASS`.

### Task 4: Record the evidence in the Chinese taskbook

**File:** Modify `D:/PyTorch for Vulkan/pytorch-src/VULKAN_SSBO_DEVELOPMENT_TASKBOOK_ZH.md`.

- [ ] **Step 1: Back up the taskbook.**

Copy it into the same timestamped FP4 integration backup directory before editing.

- [ ] **Step 2: Add the FP4 status entry.**

Record E2M1, the codebook levels, 4-bit low/high nibble order, 32-element scale, zero padding rule, `<0.12` Linear gate, actual bridge error, fallback reasons, and `resident_entries=0`. State that UMA is real compatibility evidence, discrete is policy simulation only, and FP4 Vulkan feature/storage/shader/pipeline/resident/performance remain disabled.

- [ ] **Step 3: Update the current execution instruction.**

Change the final low-precision status sentence so FP4 is listed as a completed CPU contract while all Vulkan FP4 paths remain disabled. Do not rewrite unrelated historical benchmark values.

### Task 5: Full acceptance and final checks

- [ ] **Step 1: Re-run the focused FP4 test and compile check.**

Confirm fresh exit code 0 and capture the JSON-derived max errors.

- [ ] **Step 2: Re-run UMA and discrete simulation logs.**

Confirm `uma_test.log` and `discrete_simulation_test.log` each contain FP4 `rc=0` and final `RESULT: PASS`. Do not run `run_discrete_gpu_test.bat` and do not build a wheel.

- [ ] **Step 3: Check repository hygiene.**

Run:

```powershell
git -C 'D:/PyTorch for Vulkan/pytorch-src' diff --check
git -C 'D:/PyTorch for Vulkan/pytorch-src' status --short
Get-ChildItem -Path 'D:/PyTorch for Vulkan/pytorch-src' -Recurse -File -Include '*.whl'
```

The diff check must have no whitespace errors, the wheel query must be empty, and existing unrelated dirty changes must remain untouched.

- [ ] **Step 4: Commit the entry-point and documentation changes.**

```powershell
git -C 'D:/PyTorch for Vulkan/pytorch-src' add -- 'run_uma_test.bat' 'run_discrete_simulation_test.bat' 'VULKAN_SSBO_DEVELOPMENT_TASKBOOK_ZH.md'
git -C 'D:/PyTorch for Vulkan/pytorch-src' commit -m "test: integrate FP4 compatibility contract"
```

- [ ] **Step 5: Report the result with scope boundaries.**

Report the FP4 error and bridge measurements, UMA real compatibility status, discrete simulation compatibility status, no real dGPU performance claim, no Vulkan FP4 enablement, no wheel build, and the exact log paths.
