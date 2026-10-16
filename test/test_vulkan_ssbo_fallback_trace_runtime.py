"""Runtime contract for opt-in Vulkan SSBO fallback diagnostics."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap


MARKER = "[vkssbo] fallback op=matmul_softmax"


CHILD = textwrap.dedent(
    r"""
    import ctypes
    import os
    import sys

    import torch

    case = sys.argv[1]
    torch.manual_seed(42)
    if case == "input_rank":
        query = torch.randn(1, 2, 3, 4, dtype=torch.float32).to("vulkan")
        key = torch.randn(1, 2, 3, 4, dtype=torch.float32).to("vulkan")
        result = torch.ops.vulkan_ssbo.matmul_softmax(query, key, 0.5, True)
    elif case == "invalid_scale":
        query = torch.randn(2, 8, 16, dtype=torch.float32).to("vulkan")
        key = torch.randn(2, 12, 16, dtype=torch.float32).to("vulkan")
        result = torch.ops.vulkan_ssbo.matmul_softmax(
            query, key, float("nan"), True)
    else:
        raise AssertionError(f"unknown case: {case}")

    assert result is None, repr(result)
    print(f"child PASS case={case}", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    if os.name == "nt":
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.GetCurrentProcess.argtypes = []
        kernel32.TerminateProcess.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        kernel32.TerminateProcess.restype = ctypes.c_int
        kernel32.TerminateProcess(kernel32.GetCurrentProcess(), 0)
    os._exit(0)
    """
)


def run_child(case: str, trace: bool) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    if trace:
        env["VKSSBO_TRACE_FALLBACK"] = "1"
    else:
        env.pop("VKSSBO_TRACE_FALLBACK", None)
    return subprocess.run(
        [sys.executable, "-c", CHILD, case],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def require_success(result: subprocess.CompletedProcess[str], label: str) -> None:
    if result.returncode != 0:
        raise AssertionError(
            f"{label} failed rc={result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )


def main() -> None:
    quiet = run_child("input_rank", trace=False)
    require_success(quiet, "default-off child")
    assert MARKER not in quiet.stderr, quiet.stderr

    rank = run_child("input_rank", trace=True)
    require_success(rank, "rank trace child")
    assert f"{MARKER} reason=input_guard" in rank.stderr, rank.stderr
    assert "q_shape=[1,2,3,4] k_shape=[1,2,3,4] dtype=f32" in rank.stderr, rank.stderr

    scale = run_child("invalid_scale", trace=True)
    require_success(scale, "scale trace child")
    assert f"{MARKER} reason=scale_guard" in scale.stderr, scale.stderr
    assert "q_shape=[2,8,16] k_shape=[2,12,16] dtype=f32" in scale.stderr, scale.stderr

    print(
        "Vulkan SSBO fallback trace regression: PASS",
        "default_silent=1",
        "input_guard_reason=1",
        "scale_guard_reason=1",
        flush=True,
    )


if __name__ == "__main__":
    main()
