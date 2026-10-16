"""B2 contiguous view/detach sharing through ordinary torch operations.

Uses small FP32 tensors in isolated source-tree Python, with natural exit.
Default mode does not import torch or initialize a device.
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


def child(case):
    sys.path.insert(0, str(ROOT))
    import torch
    torch.set_num_threads(1)
    assert Path(torch.__file__).resolve().parent == ROOT / "torch"
    assert not any(n.startswith(("comfy", "vulkan_gsv")) for n in sys.modules)
    host = torch.arange(21, dtype=torch.float32).reshape(3, 7)
    x = host.to("vulkan")
    ops = torch.ops.vulkan_ssbo
    versions = []
    checks = []

    def equal(actual, expected):
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)

    if case == "view":
        before = dict(ops.native_buffer_stats())
        y = x.view(7, 3)
        z = y.view(1, -1)
        a = torch.ops.aten.alias.default(x)
        assert x.shape == (3, 7) and y.shape == (7, 3) and z.shape == (1, 21)
        assert y.stride() == (3, 1) and z.stride() == (21, 1)
        assert dict(ops.native_buffer_stats()) == before, "view must not copy or submit"
        assert y._base is x and z._base is x
        base_version = x._version
        replacement = torch.arange(21, dtype=torch.float32).reshape(7, 3) + 10
        y.copy_(replacement)
        assert x._version == y._version == z._version == a._version == base_version + 1
        equal(a, replacement.reshape(3, 7))
        equal(x, replacement.reshape(3, 7))
        equal(z, replacement.reshape(1, 21))
        versions.append(x._version)
        saved = z.clone()
        saved_version = saved._version
        x.copy_(torch.zeros_like(host))
        assert saved._version == saved_version
        equal(saved, replacement.reshape(1, 21))
        equal(y, torch.zeros(7, 3))
        versions.append(x._version)
        assert x.shape == (3, 7) and x.stride() == host.stride()
        before = dict(ops.native_buffer_stats())
        version = x._version
        x.copy_(x.detach())
        assert dict(ops.native_buffer_stats()) == before
        assert x._version == y._version == version + 1
        checks += ["view_metadata_only", "writes_visible_in_both_shapes",
                   "shared_version", "clone_independent", "alias_shares",
                   "self_alias_copy_no_dispatch"]
    elif case == "detach":
        before = dict(ops.native_buffer_stats())
        d = x.detach()
        assert d is not x and d._base is None
        assert d.shape == x.shape and d.stride() == x.stride()
        assert ops.is_native_buffer(d) and not d.requires_grad
        assert dict(ops.native_buffer_stats()) == before
        original_version = x._version
        d.copy_(torch.ones_like(host))
        assert d._version == x._version == original_version + 1
        equal(x, torch.ones_like(host))
        del x
        gc.collect()
        equal(d, torch.ones_like(host))
        computed = torch.sigmoid(torch.nn.functional.silu(d))
        torch.testing.assert_close(computed.cpu(),
                                   torch.sigmoid(torch.nn.functional.silu(torch.ones_like(host))),
                                   rtol=2e-6, atol=2e-6)
        checks += ["detach_preserves_type_stride", "detach_shares_version",
                   "detach_survives_base_deletion", "compute_from_detach"]
    elif case == "lifetime":
        d = x.detach().view(1, 3, 7, 1)
        del x
        gc.collect()
        before = dict(ops.native_buffer_stats())
        y = torch.sigmoid(torch.nn.functional.silu(d))
        after = dict(ops.native_buffer_stats())
        assert after["downloads"] == before["downloads"]
        assert after["uploads"] == before["uploads"]
        del d
        gc.collect()
        torch.testing.assert_close(y.cpu(), torch.sigmoid(torch.nn.functional.silu(host)).reshape(1, 3, 7, 1),
                                   rtol=2e-6, atol=2e-6)
        checks += ["view_survives_base_deletion", "no_intermediate_host_copy", "pending_read_lifetime"]
    elif case == "reshape":
        y = x.reshape(1, 3, 7).squeeze(0)
        assert y.shape == (3, 7)
        same = x.squeeze(0)
        assert same is not x and same.shape == x.shape
        y.copy_(torch.ones_like(host))
        equal(x, torch.ones_like(host))
        alias = torch.ops.aten._reshape_alias.default(x, [7, 3], [3, 1])
        alias.copy_(torch.full((7, 3), 2.0))
        equal(x, torch.full((3, 7), 2.0))
        checks += ["reshape_and_squeeze_share", "reshape_alias_contiguous", "noop_squeeze_alias"]
        for name, fn in [
            ("invalid_numel", lambda: x.view(20)),
            ("noncontiguous_alias", lambda: torch.ops.aten._reshape_alias.default(x, [7, 3], [1, 7])),
            ("unsupported_rank", lambda: x.view(1, 1, 1, 3, 7)),
            ("two_inferred_dims", lambda: x.view(-1, -1)),
            ("unsupported_scalar", lambda: torch.empty((1,), device="vulkan").view(())),
        ]:
            try:
                fn()
            except RuntimeError:
                checks.append(name + "_rejected")
            else:
                raise AssertionError(name + " silently accepted")
        equal(x, torch.full((3, 7), 2.0))
    elif case == "data":
        raw = x.data
        base_version = x._version
        raw_version = raw._version
        raw.copy_(torch.full_like(host, 3.0))
        assert x._version == base_version and raw._version == raw_version + 1
        equal(x, torch.full_like(host, 3.0))
        d = x.detach()
        replacement = torch.ones((7, 3), dtype=torch.float32).to("vulkan")
        old_version = d._version
        d.data = replacement
        assert d.shape == (7, 3) and d.stride() == (3, 1)
        assert d._version == old_version
        equal(d, torch.ones(7, 3))
        equal(x, torch.full_like(host, 3.0))
        checks += ["data_shares_bytes_without_version", "data_rebind_updates_shape_stride",
                   "data_rebind_does_not_retarget_base"]
    elif case == "grad_gate":
        x.requires_grad_(True)
        before = dict(ops.native_buffer_stats())
        try:
            torch.sigmoid(x)
        except RuntimeError:
            pass
        else:
            raise AssertionError("unsupported gradient input was accepted")
        assert dict(ops.native_buffer_stats()) == before
        d = x.detach().view(7, 3)
        assert not d.requires_grad
        torch.testing.assert_close(torch.sigmoid(d).cpu(), torch.sigmoid(host).reshape(7, 3),
                                   rtol=2e-6, atol=2e-6)
        checks += ["gradient_input_rejected_before_dispatch", "explicit_detach_allows_inference"]
    else:
        raise ValueError(case)
    return {"case": case, "status": "PASS", "checks": checks, "versions": versions}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--case", choices=["view", "detach", "lifetime", "reshape", "data", "grad_gate"])
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.child:
        print(json.dumps(child(args.case)), flush=True)
        return 0
    if not args.run:
        print(json.dumps({"status": "READY", "gpu_workload_started": False}))
        return 0
    if args.output and args.output.exists():
        parser.error("choose a new output file")
    records = []
    env = os.environ.copy()
    env.update(VK_NATIVE_BUFFER="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    for case in ([args.case] if args.case else ["view", "detach", "lifetime", "reshape", "data", "grad_gate"]):
        result = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()),
                                 "--child", "--case", case], cwd=ROOT.parent, env=env,
                                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
        try:
            record = json.loads(result.stdout.strip().splitlines()[-1])
        except (IndexError, ValueError):
            record = {"case": case, "status": "FAIL"}
        record.update(exit_code=result.returncode, stderr=result.stderr[-4000:])
        records.append(record)
        if result.returncode:
            break
    passed = all(r["status"] == "PASS" and r["exit_code"] == 0 for r in records)
    report = {"status": "PASS" if passed else "FAIL", "cases": records,
              "scope": "bounded contiguous native FP32 buffer sharing",
              "general_strides": False, "native_backward": False,
              "binary_sha256": {name: hashlib.sha256((ROOT / "torch/lib" / name).read_bytes()).hexdigest()
                                for name in ("torch_cpu.dll", "torch_python.dll")}}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2)
    print(json.dumps(report))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
