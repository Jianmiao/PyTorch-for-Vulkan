# INT4 Weight-Only Contract Design

Date: 2026-08-22

## Goal

Add a versioned CPU reference contract for weight-only INT4 data so the
Vulkan SSBO backend can validate, dequantize, and safely fall back to the
existing FP32 weight path. This phase is a compatibility and format phase;
it does not enable an INT4 Vulkan feature, storage path, shader, pipeline, or
performance claim.

## Scope

The implementation will live beside the existing INT8 contract in
`tools/vulkan_ssbo/int4_weight_only.py` and will have an independent runtime
regression in `test_int4_weight_only_contract_runtime.py`. UMA and software
discrete-policy simulation must run the same test. The normal UMA and
discrete-simulation entry points will invoke this regression after the INT8
contract checks.

The contract has two explicit outcomes:

1. `int4_weight_only`: the payload is valid and is decoded to a contiguous
   FP32 weight for the existing operator path.
2. `fp32_fallback`: the payload is malformed, unsupported, or has a shape
   mismatch, so the original weight is returned as contiguous FP32.

No invalid payload may reach a future INT4 dispatch implicitly.

## Format v1

The payload is a frozen dataclass with these fields:

```text
qweight          uint8 tensor [out_features, ceil(in_features / 2)]
scale            float32 tensor [out_features, ceil(in_features / 32)]
zero_point       int16 tensor [out_features, ceil(in_features / 32)]
in_features      positive integer
block_size       integer 32
format_version   integer 1
scheme           "affine"
scale_granularity "per_output_channel_block"
nibble_order     "low_even_high_odd"
```

Each stored byte contains two unsigned 4-bit codes. The value for input
column `2*j` is the low nibble and the value for `2*j+1` is the high nibble.
The last high nibble is ignored when `in_features` is odd. Codes are in
`[0, 15]`. For block `b`, the decoded value is:

```text
decoded = (code - zero_point[row, b]) * scale[row, b]
```

The packer computes affine min/max parameters per output-channel block. An
all-zero block uses `scale=1` and `zero_point=0`, preserving an exact zero
representation. The v1 packer always emits contiguous tensors and does not
accept a caller-supplied zero point.

## Validation and fallback

Validation rejects, with a stable reason string, any of the following:

- wrong version, scheme, granularity, nibble order, or block size;
- non-tensor fields, wrong ranks, dtypes, shapes, or non-contiguous storage;
- `in_features` inconsistent with the original weight or packed byte width;
- non-finite or non-positive scales;
- zero points outside `[0, 15]` or with a shape different from `scale`;
- qweight values outside the uint8/nibble representation.

`resolve_weight(original_weight, payload)` returns
`(weight, mode, reason)`. A valid payload returns a contiguous FP32
dequantization and `mode="int4_weight_only"`. Every validation failure and
shape mismatch returns the original weight converted to contiguous FP32 with
`mode="fp32_fallback"`. The original tensor is never mutated.

## Numerical gates

The regression uses fixed seeds and representative linear shapes
`128x256`, `1024x1280`, and `4096x4096`. It records the relative L2 error of
`F.linear(input, dequantized_weight)` against the original FP32 reference.
The v1 candidate gate is `< 0.10` relative L2 for every representative shape.
The test also checks a small SSBO linear call with the dequantized weight and
requires the existing FP16-storage tolerance (`max_abs < 3e-2`). This is a
bridge/fallback check, not an INT4 dispatch benchmark.

Invalid payload cases must resolve to FP32 fallback and reproduce the
original-weight reference within the same existing SSBO tolerance. The test
must confirm that no resident entries remain after cleanup.

## Capability and policy boundaries

- `int4_weight_only_enabled` is not added as an enabled runtime capability in
  this phase; the format module is CPU-only.
- Existing `shaderInt8` and 8-bit storage fields remain diagnostic only.
- No INT4 shader compilation probe is required until a kernel design is
  approved separately.
- `VKSSBO_SIMULATE_DISCRETE=1` changes only policy/topology behavior. It must
  not turn INT4 into a GPU dispatch or convert simulated timings into a
  performance result.
- INT8, BF16, FP16, native SSBO, torch 14/14, and existing fusion regressions
  remain mandatory gates.

## Failure handling and reporting

The test runs its Vulkan child in a fresh process, emits one final JSON record,
and uses the existing controlled Windows termination pattern to avoid native
Vulkan teardown crashes. The JSON records format metadata, per-shape relative
L2 errors, SSBO bridge error, fallback reasons, and resident entry count.
The parent prints `INT4 weight-only format/fallback regression: PASS` only
when every assertion and the child exit code succeed.

## Deliverables

- `int4_weight_only.py` format, pack, validation, and FP32 fallback module.
- `test_int4_weight_only_contract_runtime.py` contract and bridge regression.
- UMA and discrete-simulation entry-point integration.
- A taskbook result entry with exact test status and explicit statement that
  INT4 Vulkan storage/kernel/performance remain disabled.

## Non-goals

- No INT4 Vulkan shader, feature enablement, or pipeline.
- No INT4 resident storage or async-transfer policy change.
- No wheel build or external package creation.
- No performance conclusion from CPU dequantization or discrete simulation.
