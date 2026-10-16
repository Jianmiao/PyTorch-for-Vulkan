"""Repeatable upload/compute/download benchmark for the Vulkan SSBO backend.

Run this script in a fresh process for each transfer mode.  The process
boundary is deliberate: Vulkan device selection, async queue resources and
the weight cache are all initialized from environment variables at startup.
The final line is a JSON object so callers can safely ignore diagnostic lines
printed by the backend.
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


def _parse_shape(value: str) -> tuple[int, int, int]:
    try:
        dims = tuple(int(part) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("shape must be M,K,N") from exc
    if len(dims) != 3 or any(dim <= 0 for dim in dims):
        raise argparse.ArgumentTypeError("shape must contain three positive integers M,K,N")
    return dims


def _percentile95(samples: list[float]) -> float:
    ordered = sorted(samples)
    index = max(0, min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1))
    return ordered[index]


def _plain_dict(value: Any) -> dict[str, Any]:
    return {str(key): value for key, value in dict(value).items()}


def _terminate_successfully() -> None:
    """Avoid Windows Vulkan DLL teardown races after the JSON is flushed."""
    if os.name == "nt":
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.TerminateProcess.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        kernel32.TerminateProcess.restype = ctypes.c_int
        sys.stdout.flush()
        sys.stderr.flush()
        kernel32.TerminateProcess(kernel32.GetCurrentProcess(), 0)
    raise SystemExit(0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--shape", type=_parse_shape, default=(256, 256, 256))
    args = parser.parse_args()
    if args.warmup < 0 or args.repeats < 1:
        parser.error("warmup must be >= 0 and repeats must be >= 1")

    import torch
    from torch.vulkan_ssbo_configuration import configuration_snapshot

    torch.manual_seed(42)
    m, k, n = args.shape
    left = torch.randn(m, k, dtype=torch.float32)
    right = torch.randn(k, n, dtype=torch.float32)

    # Initializing device_info before the timed region records the exact device
    # and queue topology used by this process, without polluting profile stats.
    device_info = _plain_dict(torch.ops.vulkan_ssbo.device_info())
    mode = "async" if os.environ.get("VKSSBO_ASYNC_TRANSFER") == "1" else "sync"

    def run_once() -> None:
        left_vk = left.to("vulkan")
        right_vk = right.to("vulkan")
        output = torch.mm(left_vk, right_vk)
        # A CPU materialization is the benchmark boundary and forces the
        # download path to complete before the wall-clock sample ends.
        output.cpu()

    with torch.inference_mode():
        for _ in range(args.warmup):
            run_once()
        torch.ops.vulkan_ssbo.profile_reset()
        samples: list[float] = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            run_once()
            samples.append((time.perf_counter() - start) * 1000.0)

    profile = {str(key): int(value) for key, value in dict(torch.ops.vulkan_ssbo.profile_stats()).items()}
    async_keys = (
        "async_transfer_requested",
        "async_transfer_eligible",
        "async_transfer_enabled",
        "async_transfer_resources_ready",
        "dedicated_transfer_queue_available",
        "timeline_semaphore_supported",
    )
    async_capabilities = {key: device_info.get(key, "0") for key in async_keys}
    result = {
        "schema_version": 1,
        "mode": mode,
        "device": device_info,
        "shape": {"m": m, "k": k, "n": n},
        "dtype": "float32",
        "warmup": args.warmup,
        "repeats": args.repeats,
        "samples_ms": [round(sample, 6) for sample in samples],
        "median_ms": statistics.median(samples),
        "p95_ms": _percentile95(samples),
        "profile_stats": profile,
        "bytes_uploaded": profile.get("upload_bytes", 0),
        "bytes_downloaded": profile.get("download_bytes", 0),
        "compute_submissions": profile.get("compute_submissions", 0),
        "async_capabilities": async_capabilities,
        "configuration": configuration_snapshot(),
    }
    print(json.dumps(result, sort_keys=True))
    _terminate_successfully()


if __name__ == "__main__":
    main()
