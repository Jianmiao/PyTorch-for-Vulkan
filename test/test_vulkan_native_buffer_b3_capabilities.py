"""B3 native-buffer capability declaration.

The query is metadata-only and is intentionally exercised before any Vulkan
tensor is created.  It describes the opaque Storage boundary explicitly so
applications can select a supported path without probing unsupported APIs.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


EXPECTED = {
    "device": "vulkan",
    "dtype": "float32",
    "rank": "1..4",
    "max_numel": "65535",
    "strides": "positive",
    "storage": "opaque",
    "storage_api": "unsupported",
    "untyped_storage_api": "unsupported",
    "data_ptr": "unsupported",
    "requires_grad": "unsupported",
    "backward": "unsupported",
    "copy": "fp32_positive_stride",
    "clone": "contiguous_only_for_strided",
    "as_strided": "unsupported",
}


def child(enabled: bool):
    os.environ["VK_NATIVE_BUFFER"] = "1" if enabled else "0"
    sys.path.insert(0, str(ROOT))
    import torch

    capabilities = dict(torch.ops.vulkan_ssbo.native_buffer_capabilities())
    assert capabilities["enabled"] == ("1" if enabled else "0")
    assert capabilities["representation"] == (
        "linear_fp32_buffer" if enabled else "legacy_texture"
    )
    for key, value in EXPECTED.items():
        assert capabilities[key] == value, (key, capabilities[key])
    return {
        "status": "PASS",
        "enabled": enabled,
        "capabilities": capabilities,
        "torch": torch.__version__,
        "torch_file": torch.__file__,
        "tensor_created": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--enabled", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.child:
        print(json.dumps(child(args.enabled)), flush=True)
        return 0
    if not args.run:
        print(json.dumps({
            "status": "READY",
            "gpu_workload_started": False,
            "scope": "metadata-only native buffer capability declaration",
        }))
        return 0
    if args.output and args.output.exists():
        parser.error("output already exists; choose a new evidence path")

    records = []
    for enabled in (True, False):
        env = os.environ.copy()
        env.update(
            VK_NATIVE_BUFFER="1" if enabled else "0",
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
        )
        proc = subprocess.run(
            [
                sys.executable,
                "-B",
                str(Path(__file__).resolve()),
                "--child",
                *( ["--enabled"] if enabled else [] ),
            ],
            cwd=ROOT.parent,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
        )
        try:
            record = json.loads(proc.stdout.strip().splitlines()[-1])
        except (IndexError, ValueError):
            record = {"status": "FAIL", "stdout": proc.stdout[-4000:]}
        record.update(enabled=enabled, exit_code=proc.returncode,
                      normal_shutdown=proc.returncode == 0,
                      stderr_tail=proc.stderr[-4000:])
        records.append(record)

    passed = all(
        item.get("status") == "PASS" and item.get("normal_shutdown")
        for item in records
    )
    report = {
        "status": "PASS" if passed else "FAIL",
        "runs": records,
        "gpu_workload_started": False,
        "vulkan_tensor_created": False,
        "storage_api_claim": "unsupported",
        "production_runtime_modified": False,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2)
    print(json.dumps(report))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
