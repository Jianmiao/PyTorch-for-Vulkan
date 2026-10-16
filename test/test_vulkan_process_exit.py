"""Regression: normal Python finalization must release Vulkan before DLL exit.

Default is host-only readiness. --run uses isolated, sequential child processes
and a tiny Intel UMA workload; there is no forced exit or crash suppression.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def run():
    prelude = f"import sys; sys.path.insert(0, {str(ROOT)!r}); import torch; "
    cases = {
        "import_only": "print('IMPORTED', flush=True)",
        "query_only": "assert torch.is_vulkan_available(); print('AVAILABLE', flush=True)",
        "empty_tensor": "x=torch.empty((1,), device='vulkan'); print('ALLOCATED', flush=True)",
        "pending_commands": (
            "x=torch.tensor([0.25,-0.5]).to('vulkan'); "
            "y=torch.sigmoid(x); print('ENQUEUED_WITHOUT_DOWNLOAD',flush=True)"
        ),
        "cyclic_tensor": (
            "x=torch.tensor([0.25,-0.5]).to('vulkan'); y=torch.sigmoid(x); "
            "holder=[y]; holder.append(holder); del x,y; "
            "print('CYCLE_WITH_PENDING_WORK',flush=True)"
        ),
        "global_tensor": (
            "x=torch.tensor([0.25,-0.5]).to('vulkan'); "
            "y=torch.sigmoid(torch.nn.functional.silu(x)); "
            "torch.testing.assert_close(y.cpu(),torch.sigmoid(torch.nn.functional.silu(torch.tensor([0.25,-0.5]))),atol=2e-6,rtol=2e-6); "
            "print('COMPUTED', flush=True)"
        ),
        "exception_after_tensor": (
            "x=torch.empty((1,),device='vulkan'); print('EXPECTED_EXCEPTION',flush=True); "
            "raise RuntimeError('intentional finalization regression')"
        ),
    }
    records = []
    env = os.environ.copy()
    env.update(VK_NATIVE_BUFFER="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    env.pop("VK_TRACE", None)
    for name, code in cases.items():
        proc = subprocess.run([sys.executable, "-B", "-c", prelude + code],
                              env=env, cwd=ROOT.parent, capture_output=True,
                              text=True, encoding="utf-8", errors="replace", timeout=45)
        expected = 1 if name == "exception_after_tensor" else 0
        passed = proc.returncode == expected and bool(proc.stdout.strip())
        records.append({"case": name, "exit_code": proc.returncode,
                        "expected_exit_code": expected, "passed": passed,
                        "stdout": proc.stdout.strip(), "stderr": proc.stderr[-3000:]})
    hashes = {name: hashlib.sha256((ROOT / "torch/lib" / name).read_bytes()).hexdigest()
              for name in ("torch_cpu.dll", "torch_python.dll")}
    return {"status": "PASS" if all(r["passed"] for r in records) else "FAIL",
            "cases": records, "binary_sha256": hashes,
            "scope": "normal Python exit, native Vulkan, bounded UMA",
            "forced_exit": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.run:
        print(json.dumps({"status": "READY", "gpu_workload_started": False}))
        return 0
    if args.output and args.output.exists():
        parser.error("output already exists; choose a new evidence path")
    report = run()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2)
    print(json.dumps(report))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
