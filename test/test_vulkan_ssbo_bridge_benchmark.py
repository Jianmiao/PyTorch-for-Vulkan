"""Contract test for the bridge chain benchmark."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


RUNNER = Path(__file__).with_name("bench_vulkan_ssbo_bridge.py")


def run_child(lease: bool) -> dict[str, object]:
    env = os.environ.copy()
    env["VKSSBO_HOST_LEASE"] = "1" if lease else "0"
    env["VKSSBO_PROFILE"] = "1"
    result = subprocess.run(
        [sys.executable, str(RUNNER), "--warmup", "1", "--repeats", "3"],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"bridge benchmark failed lease={lease} rc={result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return json.loads(result.stdout.strip().splitlines()[-1])


def main() -> None:
    enabled = run_child(True)
    if "skip" in enabled:
        print(f"Vulkan SSBO bridge benchmark: SKIP ({enabled['skip']})")
        return
    disabled = run_child(False)
    assert enabled["schema_version"] == 1 and disabled["schema_version"] == 1
    assert enabled["chain"] == disabled["chain"] == "conv2d_silu_conv2d"
    assert enabled["finite"] and disabled["finite"]
    assert abs(float(enabled["output_sum"]) - float(disabled["output_sum"])) < 1e-6
    assert abs(float(enabled["output_std"]) - float(disabled["output_std"])) < 1e-6
    enabled_bridge = enabled["bridge_stats"]
    disabled_bridge = disabled["bridge_stats"]
    assert enabled_bridge["host_lease_enabled"] == 1, enabled_bridge
    assert disabled_bridge["host_lease_enabled"] == 0, disabled_bridge
    assert enabled_bridge["host_lease_hits"] >= 1, enabled_bridge
    assert disabled_bridge["host_lease_hits"] == 0, disabled_bridge
    assert enabled_bridge["direct_copy_enabled"] == 0
    assert disabled_bridge["direct_copy_enabled"] == 0
    assert disabled_bridge["stock_vulkan_to_cpu_materializations"] > enabled_bridge[
        "stock_vulkan_to_cpu_materializations"
    ], (enabled_bridge, disabled_bridge)
    print(
        "Vulkan SSBO bridge benchmark contract: PASS",
        f"enabled_median_ms={enabled['median_ms']:.3f}",
        f"disabled_median_ms={disabled['median_ms']:.3f}",
        f"fallback_enabled={enabled_bridge['stock_vulkan_to_cpu_materializations']}",
        f"fallback_disabled={disabled_bridge['stock_vulkan_to_cpu_materializations']}",
    )


if __name__ == "__main__":
    main()
