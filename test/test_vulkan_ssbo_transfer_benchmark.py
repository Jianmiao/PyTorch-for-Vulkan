"""Contract test for the repeatable Vulkan SSBO transfer benchmark.

The benchmark is intentionally process-isolated: sync and async runs must use
the same tensors, warmup, repeats, and selected device.  This test exercises
the JSON contract on the local Vulkan device without requiring a discrete GPU.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


RUNNER = Path(__file__).with_name("bench_vulkan_ssbo_transfer.py")
REQUIRED = {
    "schema_version",
    "mode",
    "device",
    "shape",
    "dtype",
    "warmup",
    "repeats",
    "samples_ms",
    "median_ms",
    "p95_ms",
    "profile_stats",
    "bytes_uploaded",
    "bytes_downloaded",
    "compute_submissions",
    "async_capabilities",
    "configuration",
}


def main() -> None:
    env = os.environ.copy()
    env.pop("VKSSBO_DEVICE_NAME", None)
    env.pop("VKSSBO_DEVICE_INDEX", None)
    env["VKSSBO_PROFILE"] = "1"
    env["VKSSBO_ASYNC_TRANSFER"] = "0"
    result = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--warmup",
            "1",
            "--repeats",
            "3",
            "--shape",
            "16,64,64",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"benchmark failed rc={result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    missing = REQUIRED - set(payload)
    assert not missing, f"missing={sorted(missing)} payload={payload}"
    assert payload["schema_version"] == 1, payload
    assert payload["mode"] == "sync", payload
    assert payload["warmup"] == 1 and payload["repeats"] == 3, payload
    assert len(payload["samples_ms"]) == 3, payload
    assert payload["median_ms"] > 0 and payload["p95_ms"] > 0, payload
    assert payload["bytes_uploaded"] > 0 and payload["bytes_downloaded"] > 0, payload
    assert payload["compute_submissions"] >= 1, payload
    assert payload["profile_stats"]["enabled"] == 1, payload
    profile = payload["profile_stats"]
    assert profile["schema_version"] == 2, profile
    assert profile["dispatches"] >= profile["compute_submissions"], profile
    assert profile["queue_wait_idle_count"] == 0, profile
    assert (
        profile["gpu_timestamp_submissions"]
        + profile["gpu_timestamp_failures"]
        >= profile["compute_submissions"]
    ), profile
    assert payload["async_capabilities"]["async_transfer_requested"] == "0", payload
    assert payload["configuration"]["schema_version"] == 1, payload
    settings = {item["name"]: item for item in payload["configuration"]["settings"]}
    assert settings["VKSSBO_PROFILE"]["current"] == "1", settings
    print(
        "Vulkan SSBO transfer benchmark contract: PASS",
        f"median_ms={payload['median_ms']:.3f}",
        f"compute_submissions={payload['compute_submissions']}",
    )


if __name__ == "__main__":
    main()
