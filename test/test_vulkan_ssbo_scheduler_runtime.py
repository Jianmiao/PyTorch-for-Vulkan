"""PyTorch SSBO scheduler lifecycle contract.

This test is also copied into the portable tools test root so the UMA and
discrete-policy runners exercise the same integration contract.
"""

from __future__ import annotations

import ctypes
import json
import os
import subprocess
import sys


REQUIRED_KEYS = {
    "last_submitted",
    "completed",
    "in_flight",
    "peak_in_flight",
    "submits",
    "polls",
    "waits",
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

    torch.manual_seed(42)
    before = dict(torch.ops.vulkan_ssbo.scheduler_stats())
    assert REQUIRED_KEYS.issubset(before), before
    with torch.inference_mode():
        query = torch.randn(2, 8, 16, dtype=torch.float32).to("vulkan")
        key = torch.randn(2, 12, 16, dtype=torch.float32).to("vulkan")
        result = torch.ops.vulkan_ssbo.matmul_softmax(query, key, 0.25, True)
        assert tuple(result.shape) == (2, 8, 12)
        after = dict(torch.ops.vulkan_ssbo.scheduler_stats())
        result_cpu = result.cpu()

    assert torch.isfinite(result_cpu).all(), "scheduler test produced non-finite output"
    assert after["last_submitted"] > before["last_submitted"], after
    assert after["completed"] >= after["last_submitted"], after
    assert after["in_flight"] == 0, after
    assert after["peak_in_flight"] >= 1, after
    assert after["submits"] > before["submits"], after
    assert after["waits"] > before["waits"], after
    assert after["polls"] > before["polls"], after
    print(json.dumps({"before": before, "after": after}, sort_keys=True))
    terminate_successfully()


def main() -> None:
    env = os.environ.copy()
    result = subprocess.run(
        [sys.executable, __file__, "--child"],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(
            "scheduler lifecycle child failed\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    before = payload["before"]
    after = payload["after"]
    print(
        "PyTorch scheduler lifecycle regression: PASS",
        f"submitted_delta={after['last_submitted'] - before['last_submitted']}",
        f"submit_delta={after['submits'] - before['submits']}",
        f"wait_delta={after['waits'] - before['waits']}",
        f"poll_delta={after['polls'] - before['polls']}",
        f"completed={after['completed']}",
        f"in_flight={after['in_flight']}",
        flush=True,
    )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        child_main()
    else:
        main()
