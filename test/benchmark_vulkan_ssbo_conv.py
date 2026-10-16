"""A/B benchmark for native Vulkan SSBO Conv2D kernels.

The benchmark keeps tensors and the weight cache alive while switching only
the input-channel chunk size. Weights stay on CPU so the SSBO cache can upload
them directly without the legacy Vulkan texture-packing limits. Each measured
call includes the native PyTorch bridge and a CPU synchronization of the result.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch


SHAPES = {
    "small": (1, 32, 64, 64, 64),
    "medium": (1, 64, 64, 128, 128),
    "sdxl": (2, 320, 320, 64, 64),
    "vae": (1, 256, 256, 512, 512),
}


def run_kernel(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    variant: str,
    warmup: int,
    repeats: int,
) -> tuple[list[float], torch.Tensor]:
    os.environ["VKSSBO_CONV_X32"] = "1"
    os.environ["VKSSBO_CONV_C32"] = "1" if variant == "c32" else "0"
    os.environ["VKSSBO_CONV_V4"] = "1" if variant in {"v4", "v4acc"} else "0"
    os.environ["VKSSBO_CONV_V4ACC"] = "1" if variant == "v4acc" else "0"

    result = torch.empty(0)
    for _ in range(warmup):
        result = torch.conv2d(x, weight, bias, padding=1).cpu()

    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = torch.conv2d(x, weight, bias, padding=1).cpu()
        samples.append((time.perf_counter() - start) * 1000.0)
    return samples, result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("shapes", nargs="*", choices=SHAPES, default=["small"])
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()

    torch.manual_seed(42)
    with torch.inference_mode():
        for name in args.shapes:
            batch, channels, outputs, height, width = SHAPES[name]
            x = torch.rand(batch, channels, height, width).to("vulkan")
            weight = torch.rand(outputs, channels, 3, 3)
            bias = torch.rand(outputs)

            c16_samples, c16_result = run_kernel(
                x, weight, bias, variant="scalar", warmup=args.warmup, repeats=args.repeats
            )
            c32_samples, c32_result = run_kernel(
                x, weight, bias, variant="c32", warmup=args.warmup, repeats=args.repeats
            )
            v4_samples, v4_result = run_kernel(
                x, weight, bias, variant="v4", warmup=args.warmup, repeats=args.repeats
            )
            v4acc_samples, v4acc_result = run_kernel(
                x, weight, bias, variant="v4acc", warmup=args.warmup, repeats=args.repeats
            )
            c16_ms = statistics.median(c16_samples)
            c32_ms = statistics.median(c32_samples)
            v4_ms = statistics.median(v4_samples)
            v4acc_ms = statistics.median(v4acc_samples)
            c32_max_abs = float((c16_result - c32_result).abs().max())
            v4_max_abs = float((c16_result - v4_result).abs().max())
            v4acc_max_abs = float((c16_result - v4acc_result).abs().max())
            print(
                f"{name}: scalar={c16_ms:.3f} ms "
                f"v4={v4_ms:.3f} ms ({c16_ms / v4_ms:.3f}x) "
                f"v4acc={v4acc_ms:.3f} ms ({v4_ms / v4acc_ms:.3f}x-v4) "
                f"C32={c32_ms:.3f} ms ({c16_ms / c32_ms:.3f}x) "
                f"max_abs_v4={v4_max_abs:.6f} "
                f"max_abs_v4acc={v4acc_max_abs:.6f} max_abs_C32={c32_max_abs:.6f}"
            )


if __name__ == "__main__":
    main()
