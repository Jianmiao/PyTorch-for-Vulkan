"""B3 opaque-storage boundary and functional copy regression.

Vulkan tensors use OpaqueTensorImpl and therefore intentionally have no
``c10::Storage``.  This test makes that contract explicit and exercises the
functional ``aten.copy`` guard which must not dereference an empty Storage.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def _reject(name, fn, checks):
    try:
        fn()
    except (RuntimeError, NotImplementedError) as exc:
        checks.append({"name": name, "status": "REJECTED",
                       "message": str(exc)[:500]})
    else:
        raise AssertionError(name + " unexpectedly succeeded")


def child():
    os.environ["VK_NATIVE_BUFFER"] = "1"
    sys.path.insert(0, str(ROOT))
    import torch

    host = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    x = host.to("vulkan")
    assert torch._C._has_storage(x) is False
    checks = []
    _reject("storage", x.storage, checks)
    _reject("untyped_storage", x.untyped_storage, checks)
    _reject("data_ptr", x.data_ptr, checks)

    # The functional copy path used to dereference unsafe_storage() before
    # dispatch.  It must now either return a correct copy or reject cleanly;
    # a process crash is a regression.  Native FP32 contiguous tensors should
    # take the former path.
    copied = torch.ops.aten.copy.default(x, x, False)
    assert copied.device.type == "vulkan"
    torch.testing.assert_close(copied.cpu(), host, rtol=0, atol=0)
    copied_from_cpu = torch.ops.aten.copy.default(x, host, False)
    assert copied_from_cpu.device.type == "vulkan"
    torch.testing.assert_close(copied_from_cpu.cpu(), host, rtol=0, atol=0)
    return {
        "status": "PASS",
        "torch": torch.__version__,
        "has_storage": False,
        "rejections": checks,
        "functional_copy": "PASS",
        "functional_copy_from_cpu": "PASS",
        "native_backward": False,
        "performance_claim": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.child:
        print(json.dumps(child()), flush=True)
        return 0
    if not args.run:
        print(json.dumps({"status": "READY", "gpu_workload_started": False,
                          "scope": "opaque storage and functional copy"}))
        return 0
    if args.output and args.output.exists():
        parser.error("output already exists; choose a new evidence path")
    env = os.environ.copy()
    env.update(VK_NATIVE_BUFFER="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    proc = subprocess.run(
        [sys.executable, "-B", str(Path(__file__).resolve()), "--child"],
        cwd=ROOT.parent, env=env, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=90,
    )
    try:
        result = json.loads(proc.stdout.strip().splitlines()[-1])
    except (IndexError, ValueError):
        result = {"status": "FAIL", "stdout": proc.stdout[-4000:]}
    result.update(exit_code=proc.returncode,
                  normal_shutdown=proc.returncode == 0,
                  stderr_tail=proc.stderr[-4000:])
    report = {
        "status": "PASS" if result.get("status") == "PASS" and proc.returncode == 0 else "FAIL",
        "run": result,
        "gpu_workload_started": True,
        "scope": "bounded FP32 UMA functional-copy regression",
        "production_runtime_modified": False,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2)
    print(json.dumps(report))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
