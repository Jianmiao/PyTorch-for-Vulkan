"""Measure the stock-Vulkan/SSBO bridge on a representative operator chain.

The process is intentionally isolated so VKSSBO_HOST_LEASE is fixed before
the runtime is initialized. Results are diagnostic and are not dGPU claims.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import statistics
import sys
import time
from typing import Any


def _p95(samples: list[float]) -> float:
    ordered = sorted(samples)
    index = max(0, min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1))
    return ordered[index]


def _plain_dict(value: Any) -> dict[str, Any]:
    return {str(key): value for key, value in dict(value).items()}


def _terminate_successfully() -> None:
    sys.stdout.flush()
    sys.stderr.flush()
    if os.name == "nt":
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.TerminateProcess.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        kernel32.TerminateProcess.restype = ctypes.c_int
        kernel32.TerminateProcess(kernel32.GetCurrentProcess(), 0)
    raise SystemExit(0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.warmup < 0 or args.repeats < 1:
        parser.error("warmup must be >= 0 and repeats must be >= 1")

    import torch

    try:
        bridge_stats = torch.ops.vulkan_ssbo.bridge_stats
        bridge_reset = torch.ops.vulkan_ssbo.bridge_reset
    except (AttributeError, RuntimeError):
        print(json.dumps({"skip": "bridge API is not present in this runtime"}))
        _terminate_successfully()

    torch.manual_seed(42)
    x = torch.randn(1, 8, 32, 32, dtype=torch.float32).to("vulkan")
    w1 = torch.randn(8, 8, 3, 3, dtype=torch.float32).to("vulkan")
    b1 = torch.randn(8, dtype=torch.float32).to("vulkan")
    w2 = torch.randn(6, 8, 3, 3, dtype=torch.float32).to("vulkan")
    b2 = torch.randn(6, dtype=torch.float32).to("vulkan")

    def run_once() -> torch.Tensor:
        first = torch.conv2d(x, w1, b1, padding=1)
        activated = torch.nn.functional.silu(first)
        second = torch.conv2d(activated, w2, b2, padding=1)
        return second.cpu()

    with torch.inference_mode():
        for _ in range(args.warmup):
            run_once()
        bridge_reset()
        torch.ops.vulkan_ssbo.profile_reset()
        samples: list[float] = []
        output = None
        for _ in range(args.repeats):
            start = time.perf_counter()
            output = run_once()
            samples.append((time.perf_counter() - start) * 1000.0)

    assert output is not None
    bridge = _plain_dict(torch.ops.vulkan_ssbo.bridge_stats())
    profile = _plain_dict(torch.ops.vulkan_ssbo.profile_stats())
    result = {
        "schema_version": 1,
        "mode": "lease_enabled" if os.environ.get("VKSSBO_HOST_LEASE", "1") != "0" else "lease_disabled",
        "chain": "conv2d_silu_conv2d",
        "shape": [1, 8, 32, 32],
        "warmup": args.warmup,
        "repeats": args.repeats,
        "samples_ms": [round(sample, 6) for sample in samples],
        "median_ms": statistics.median(samples),
        "p95_ms": _p95(samples),
        "bridge_stats": bridge,
        "profile_stats": profile,
        "output_sum": float(output.sum()),
        "output_std": float(output.std()),
        "finite": bool(torch.isfinite(output).all()),
    }
    print(json.dumps(result, sort_keys=True))
    _terminate_successfully()


if __name__ == "__main__":
    main()
