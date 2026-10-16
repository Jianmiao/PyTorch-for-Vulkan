"""Runtime contract for opt-in Vulkan SSBO profiling statistics.

The profiler records host-visible upload, submission/wait, and download
segments plus Vulkan GPU command timestamps. It is diagnostic-only and must
not be required for normal execution.
Run this file with the Python environment containing the freshly built wheel.
"""

from __future__ import annotations

import ctypes
import json
import os
import subprocess
import sys


REQUIRED_KEYS = {
    "schema_version",
    "enabled",
    "upload_host_ns",
    "upload_bytes",
    "upload_calls",
    "submit_wait_ns",
    "compute_submissions",
    "transfer_submissions",
    "copy_submissions",
    "fence_waits",
    "queue_wait_idle_count",
    "gpu_command_ns",
    "gpu_timestamp_submissions",
    "gpu_timestamp_failures",
    "dispatches",
    "download_host_ns",
    "download_transfer_wait_ns",
    "download_bytes",
    "download_submissions",
    "download_calls",
}


def terminate_successfully() -> None:
    sys.stdout.flush()
    sys.stderr.flush()
    if os.name == "nt":
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.TerminateProcess.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        kernel32.TerminateProcess.restype = ctypes.c_int
        kernel32.TerminateProcess(kernel32.GetCurrentProcess(), 0)
    os._exit(0)


def child_main() -> None:
    import torch

    torch.manual_seed(42)
    torch.ops.vulkan_ssbo.profile_reset()
    empty = dict(torch.ops.vulkan_ssbo.profile_stats())
    assert set(empty) >= REQUIRED_KEYS, empty
    assert empty["schema_version"] == 2, empty
    assert empty["enabled"] == 1, empty
    for key in REQUIRED_KEYS - {"schema_version", "enabled"}:
        assert empty[key] == 0, (key, empty)

    with torch.inference_mode():
        left = torch.randn(96, 160).to("vulkan")
        right = torch.randn(160, 128).to("vulkan")
        result = torch.mm(left, right).cpu()
    assert result.shape == (96, 128)

    stats = dict(torch.ops.vulkan_ssbo.profile_stats())
    assert set(stats) >= REQUIRED_KEYS, stats
    assert stats["schema_version"] == 2, stats
    assert stats["enabled"] == 1, stats
    assert stats["upload_bytes"] > 0, stats
    assert stats["upload_calls"] >= 2, stats
    assert stats["submit_wait_ns"] > 0, stats
    assert stats["compute_submissions"] >= 1, stats
    assert stats["fence_waits"] >= stats["compute_submissions"], stats
    assert stats["queue_wait_idle_count"] == 0, stats
    assert stats["dispatches"] >= 1, stats
    assert (
        stats["gpu_timestamp_submissions"]
        + stats["gpu_timestamp_failures"]
        >= stats["compute_submissions"]
    ), stats
    if stats["gpu_timestamp_submissions"]:
        assert stats["gpu_command_ns"] > 0, stats
    assert stats["download_host_ns"] > 0, stats
    assert stats["download_bytes"] > 0, stats
    assert stats["download_submissions"] >= 1, stats
    assert stats["download_calls"] >= 1, stats
    assert stats["download_transfer_wait_ns"] >= 0, stats
    print(json.dumps(stats, sort_keys=True))
    terminate_successfully()


def main() -> None:
    env = os.environ.copy()
    env["VKSSBO_PROFILE"] = "1"
    result = subprocess.run(
        [sys.executable, __file__, "--child"],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(
            "profile child failed\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    stats = json.loads(result.stdout.strip().splitlines()[-1])
    print(
        "Vulkan SSBO profile runtime regression: PASS",
        f"upload_bytes={stats['upload_bytes']}",
        f"compute_submissions={stats['compute_submissions']}",
        f"gpu_command_ns={stats['gpu_command_ns']}",
        f"download_bytes={stats['download_bytes']}",
    )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        child_main()
    else:
        main()
