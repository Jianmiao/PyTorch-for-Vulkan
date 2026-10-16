# Resident Graph Benchmark Design

Date: 2026-08-22

## Goal

Measure whether keeping intermediate tensors resident on the SSBO device is
faster and materially reduces host/device traffic on UMA and discrete GPUs.
This phase is diagnostic only; it must not change default dispatch or async
policy.

## Comparisons

Use one fixed seed, shape, dtype, warmup count, and repeat count for three
paths:

1. **Eager**: each elementwise operation uploads and downloads at the Python
   boundary.
2. **Resident chain**: upload once, submit each resident operation, download
   once at the end.
3. **Resident graph**: upload once, record all operations, submit once, and
   download once.

The first implementation uses the existing standalone `vkssbo` resident API.
It reports wall-clock median/p95, output max absolute error, resident bytes
before/after cleanup, and the number of logical operations. C++ profiling
counters are added only if the standalone DLL can expose them without changing
the existing ABI.

## Correctness and failure policy

The eager result is the reference for the same fp16-storage path. A run fails
if the resident or graph result exceeds the established elementwise tolerance,
if resident entries remain after cleanup, or if a graph abort leaks handles.
Unsupported shapes and allocation failures remain explicit fallback results;
they are not silently counted as speedups.

## Hardware matrix

- UMA: local Arc/iGPU run for correctness, memory lifetime, and fallback.
- Discrete/mixed: RTX 4060 run for resident traffic and queue behavior.

Async transfer is not toggled inside a single process. Sync and async remain
separate processes, and async is considered beneficial only when repeated
resident-chain measurements improve median and p95 without increasing error or
leaking memory.

## Deliverables

- A JSON-producing benchmark runner and contract test.
- A short result entry in `VULKAN_SSBO_DEVELOPMENT_TASKBOOK_ZH.md`.
- No wheel build and no default-policy change in this phase.
