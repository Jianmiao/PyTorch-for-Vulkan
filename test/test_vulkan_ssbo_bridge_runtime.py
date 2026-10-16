"""Runtime contract for the SSBO/stock Vulkan materialization boundary."""

from __future__ import annotations

import ctypes
import json
import os
import subprocess
import sys


REQUIRED_KEYS = {
    "schema_version",
    "host_lease_enabled",
    "direct_copy_eligible",
    "direct_copy_enabled",
    "host_lease_hits",
    "host_lease_misses",
    "host_lease_hit_bytes",
    "host_lease_miss_bytes",
    "ssbo_to_cpu_materializations",
    "ssbo_to_cpu_bytes",
    "stock_vulkan_to_cpu_materializations",
    "stock_vulkan_to_cpu_bytes",
    "cpu_to_stock_vulkan_materializations",
    "cpu_to_stock_vulkan_bytes",
    "stock_vulkan_copy_calls",
    "stock_vulkan_copy_from_ssbo",
}


def terminate_successfully() -> None:
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


def child_main() -> None:
    import torch

    mode = sys.argv[2]
    assert mode in {"enabled", "disabled"}, mode
    expected_lease = int(mode == "enabled")
    torch.manual_seed(42)

    try:
        bridge_stats = torch.ops.vulkan_ssbo.bridge_stats
        bridge_reset = torch.ops.vulkan_ssbo.bridge_reset
    except (AttributeError, RuntimeError):
        print(json.dumps({"skip": "bridge API is not present in this runtime"}))
        terminate_successfully()

    bridge_reset()
    empty = dict(bridge_stats())
    assert REQUIRED_KEYS.issubset(empty), empty
    assert empty["schema_version"] == 2, empty
    assert empty["host_lease_enabled"] == expected_lease, empty
    assert empty["direct_copy_eligible"] == 0, empty
    assert empty["direct_copy_enabled"] == 0, empty
    assert empty["ssbo_to_cpu_materializations"] == 0, empty
    assert empty["stock_vulkan_to_cpu_materializations"] == 0, empty
    assert empty["cpu_to_stock_vulkan_materializations"] == 0, empty

    with torch.inference_mode():
        x = torch.randn(1, 4, 8, 8, dtype=torch.float32).to("vulkan")
        # The SSBO result is first downloaded to a host mirror and then
        # materialized as stock Vulkan storage. A second SSBO consumer should
        # reuse that mirror instead of downloading the stock tensor again.
        y = torch.nn.functional.silu(x)
        z = torch.nn.functional.silu(y)
        assert tuple(z.shape) == tuple(y.shape)
        z_cpu = z.cpu()
        assert torch.isfinite(z_cpu).all(), "bridge produced non-finite output"

        copied = torch.empty_like(y)
        copied.copy_(y)

    stats = dict(bridge_stats())
    assert stats["ssbo_to_cpu_materializations"] >= 2, stats
    assert stats["ssbo_to_cpu_bytes"] > 0, stats
    assert stats["stock_vulkan_to_cpu_materializations"] >= 1, stats
    assert stats["stock_vulkan_to_cpu_bytes"] > 0, stats
    assert stats["cpu_to_stock_vulkan_materializations"] >= 2, stats
    assert stats["cpu_to_stock_vulkan_bytes"] > 0, stats
    if expected_lease:
        assert stats["host_lease_hits"] >= 1, stats
        assert stats["host_lease_hit_bytes"] > 0, stats
    else:
        assert stats["host_lease_hits"] == 0, stats
        assert stats["host_lease_hit_bytes"] == 0, stats
        assert stats["stock_vulkan_to_cpu_materializations"] >= 2, stats
    assert stats["stock_vulkan_copy_calls"] >= 1, stats
    assert stats["stock_vulkan_copy_from_ssbo"] >= 1, stats
    assert stats["direct_copy_eligible"] == 0, stats
    assert stats["direct_copy_enabled"] == 0, stats
    stats["mode"] = mode
    stats["output_sum"] = float(z_cpu.sum())
    stats["output_std"] = float(z_cpu.std())
    print(json.dumps(stats, sort_keys=True))
    terminate_successfully()


def run_child(mode: str) -> dict[str, int | float | str]:
    env = os.environ.copy()
    env["VKSSBO_HOST_LEASE"] = "1" if mode == "enabled" else "0"
    result = subprocess.run(
        [sys.executable, __file__, "--child", mode],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(
            "bridge runtime child failed\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return json.loads(result.stdout.strip().splitlines()[-1])


def main() -> None:
    enabled = run_child("enabled")
    if "skip" in enabled:
        print(f"Vulkan SSBO bridge runtime regression: SKIP ({enabled['skip']})")
        return
    disabled = run_child("disabled")
    assert "skip" not in disabled, disabled
    assert abs(float(enabled["output_sum"]) - float(disabled["output_sum"])) < 1e-6
    assert abs(float(enabled["output_std"]) - float(disabled["output_std"])) < 1e-6
    print(
        "Vulkan SSBO bridge runtime regression: PASS",
        f"lease_hits={enabled['host_lease_hits']}",
        f"fallback_enabled={enabled['stock_vulkan_to_cpu_materializations']}",
        f"fallback_disabled={disabled['stock_vulkan_to_cpu_materializations']}",
        f"ssbo_to_cpu={enabled['ssbo_to_cpu_materializations']}",
        f"cpu_to_stock={enabled['cpu_to_stock_vulkan_materializations']}",
        f"stock_copy_from_ssbo={enabled['stock_vulkan_copy_from_ssbo']}",
    )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        child_main()
    else:
        main()
