"""Bounded native Tensor smoke; run outside application patch environments.

No device is initialized without --run. Uses only the source-tree PyTorch
package selected by the caller. This is an FP32 inference contract, not a
performance benchmark or a claim of full native Tensor/autograd support.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def child():
    sys.path.insert(0, str(ROOT))
    import torch

    torch.set_num_threads(1)
    assert Path(torch.__file__).resolve().parent == ROOT / "torch"
    assert not any(name.startswith(("comfy", "vulkan_gsv")) for name in sys.modules)
    import inspect
    assert inspect.ismethoddescriptor(torch.Tensor.to)
    assert torch.nn.functional.scaled_dot_product_attention.__module__ == "torch._C._nn"
    assert torch.is_vulkan_available()

    ops = torch.ops.vulkan_ssbo
    enabled = os.environ.get("VK_NATIVE_BUFFER") == "1"
    checks = []
    errors = []

    def reject(name, fn, message):
        try:
            fn()
        except (RuntimeError, NotImplementedError) as exc:
            assert message in str(exc), (name, str(exc))
            errors.append(name)
        else:
            raise AssertionError(name + " did not reject an unsupported request")

    shapes = [(17,), (3, 7), (2, 3, 5), (1, 2, 3, 7)] if enabled else [(2, 3)]
    for shape in shapes:
        count = 1
        for size in shape:
            count *= size
        host = torch.linspace(-2, 2, count, dtype=torch.float32).reshape(shape)
        before = dict(ops.native_buffer_stats())
        x = host.to("vulkan")
        assert x.device.type == "vulkan" and x.dtype == torch.float32
        assert bool(ops.is_native_buffer(x)) == enabled
        if not enabled:
            torch.testing.assert_close(x.cpu(), host, atol=2e-3, rtol=2e-3)
            checks.append({"shape": shape, "storage": "legacy_texture"})
            del x
            continue
        assert x.stride() == host.stride()
        assert x.is_contiguous()
        if len(shape) == 4:
            assert not x.is_contiguous(memory_format=torch.channels_last)
        y = torch.nn.functional.silu(x)
        z = torch.sigmoid(y)
        assert bool(ops.is_native_buffer(y)) and bool(ops.is_native_buffer(z))
        chain = dict(ops.native_buffer_stats())
        assert chain["uploads"] - before["uploads"] == 1
        assert chain["downloads"] - before["downloads"] == 0
        assert chain["unary_dispatches"] - before["unary_dispatches"] == 2
        # Inputs can die before the final download: their buffers must remain
        # safe until queued reads complete.
        del x, y
        gc.collect()
        result = z.cpu()
        expected = torch.sigmoid(torch.nn.functional.silu(host))
        torch.testing.assert_close(result, expected, atol=2e-6, rtol=2e-6)
        after = dict(ops.native_buffer_stats())
        assert after["downloads"] - before["downloads"] == 1
        checks.append({"shape": shape,
                       "max_error": float((result - expected).abs().max()),
                       "chain_uploads": chain["uploads"] - before["uploads"],
                       "chain_downloads": chain["downloads"] - before["downloads"],
                       "final_downloads": after["downloads"] - before["downloads"]})
        del z

    if enabled:
        host = torch.arange(21, dtype=torch.float32).reshape(3, 7)
        x = host.to("vulkan")
        # Minimal regression for the initial B1 failure: odd-width rows must
        # survive a boundary transfer before any compute shader is involved.
        torch.testing.assert_close(x.cpu(), host, rtol=0, atol=0)
        cloned = x.clone()
        x.copy_(torch.zeros_like(host))
        torch.testing.assert_close(cloned.cpu(), host, rtol=0, atol=0)
        target = torch.empty((7, 3), dtype=torch.float32).t()
        target.copy_(cloned)
        torch.testing.assert_close(target, host, rtol=0, atol=0)
        torch.testing.assert_close(x.cpu(), torch.zeros_like(host), rtol=0, atol=0)
        for host_source in (host.t(), torch.arange(42, dtype=torch.float32)
                            .reshape(1, 2, 3, 7).contiguous(memory_format=torch.channels_last)):
            uploaded = torch.empty(host_source.shape, dtype=torch.float32, device="vulkan")
            uploaded.copy_(host_source)
            torch.testing.assert_close(uploaded.cpu(), host_source, rtol=0, atol=0)
            del uploaded
        reject("noncontiguous_factory", lambda: torch.empty_strided(
            (2, 3), (1, 2), device="vulkan"), "non-contiguous")
        reject("unsupported_half", lambda: torch.empty(
            (2, 3), dtype=torch.float16, device="vulkan"), "float32")
        reject("oversized", lambda: torch.empty((65536,), device="vulkan"), "65535")
        reject("invalid_view_size", lambda: cloned.view(20), "invalid")
        reject("texture_operator", lambda: torch.log(cloned), "texture-only")
        reject("dtype_copy", lambda: cloned.copy_(host.double()), "float32")
        del cloned, x
        gc.collect()

    return {"compute_checks": "PASS", "enabled": enabled, "torch": torch.__version__,
            "torch_file": torch.__file__, "checks": checks,
            "rejected": errors, "stats": dict(ops.native_buffer_stats()),
            "native_backward": False, "performance_claim": False,
            "general_view_alias": "contiguous only; separately tested by B2"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.child:
        result = child()
        print(json.dumps(result), flush=True)
        return 0
    if not args.run:
        print(json.dumps({"status": "READY", "gpu_workload_started": False,
                          "scope": "bounded native FP32 chain"}))
        return 0
    if args.output and args.output.exists():
        parser.error("output already exists; choose a new evidence path")
    results = []
    for enabled in (True, False):
        env = os.environ.copy()
        env.update(VK_NATIVE_BUFFER="1" if enabled else "0",
                   OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        proc = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()), "--child"],
                              cwd=str(ROOT.parent), env=env, capture_output=True,
                              text=True, encoding="utf-8", errors="replace", timeout=90)
        try:
            result = json.loads(proc.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            result = {"compute_checks": "FAIL", "enabled": enabled}
        result["process_exit_code"] = proc.returncode
        result["normal_shutdown"] = proc.returncode == 0
        result["stderr_tail"] = proc.stderr[-5000:]
        results.append(result)
        if result["compute_checks"] != "PASS":
            break
    passed = len(results) == 2 and all(
        r["compute_checks"] == "PASS" and r["normal_shutdown"] for r in results)
    report = {"status": "PASS" if passed else "PARTIAL", "scope": "B1 isolated source-tree runtime",
              "runs": results, "performance_claim": False,
              "binary_sha256": {name: hashlib.sha256(
                  (ROOT / "torch/lib" / name).read_bytes()).hexdigest()
                  for name in ("torch_cpu.dll", "torch_python.dll")}}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2)
    print(json.dumps(report))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
