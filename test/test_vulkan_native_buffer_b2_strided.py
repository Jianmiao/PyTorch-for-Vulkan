"""Bounded B2 positive-stride views and copies, with natural process exit.

Default mode never imports torch or initializes Vulkan. --run launches small
sequential source-tree subprocesses; this is not a training/performance test.
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
CASES = (
    "metadata_gather",
    "scatter",
    "device_copy",
    "offset_contiguous",
    "detach_data",
    "lifetime",
    "guards",
)


def child(case):
    sys.path.insert(0, str(ROOT))
    import torch

    torch.set_num_threads(1)
    assert Path(torch.__file__).resolve().parent == ROOT / "torch"
    assert not any(name.startswith(("comfy", "vulkan_gsv")) for name in sys.modules)
    ops = torch.ops.vulkan_ssbo
    checks = []
    layouts = []
    rejections = []

    def stats():
        return dict(ops.native_buffer_stats())

    def layout(actual, expected):
        assert actual.device.type == "vulkan" and actual.dtype == torch.float32
        assert ops.is_native_buffer(actual)
        assert actual.shape == expected.shape, (actual.shape, expected.shape)
        assert actual.stride() == expected.stride(), (actual.stride(), expected.stride())
        assert actual.storage_offset() == expected.storage_offset(), (
            actual.storage_offset(), expected.storage_offset()
        )
        assert actual.is_contiguous() == expected.is_contiguous()
        layouts.append({"shape": list(actual.shape), "stride": list(actual.stride()),
                        "storage_offset": actual.storage_offset()})

    def equal(actual, expected):
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)

    def no_host_copies(before):
        after = stats()
        assert after["uploads"] == before["uploads"]
        assert after["downloads"] == before["downloads"]

    def reject(name, fn):
        before = stats()
        try:
            fn()
        except (RuntimeError, NotImplementedError) as exc:
            rejections.append({"name": name, "message": str(exc)[:1200]})
        else:
            raise AssertionError(name + " silently accepted")
        assert stats() == before, name + " submitted device work before rejecting"

    if case == "metadata_gather":
        host = torch.linspace(-3, 3, 120, dtype=torch.float32).reshape(2, 3, 4, 5)
        x = host.to("vulkan")
        before = stats()
        transposed = x.transpose(-1, -3)
        expected_t = host.transpose(-1, -3)
        permuted = transposed.permute(3, 0, 2, 1)
        expected_p = expected_t.permute(3, 0, 2, 1)
        sliced = permuted[1:3, :, 1:4:2, 1:5:2]
        expected_s = expected_p[1:3, :, 1:4:2, 1:5:2]
        for actual, expected in ((transposed, expected_t), (permuted, expected_p),
                                 (sliced, expected_s)):
            layout(actual, expected)
        assert stats() == before, "metadata-only views copied or dispatched"
        for actual, expected in ((transposed, expected_t), (permuted, expected_p),
                                 (sliced, expected_s)):
            equal(actual, expected)
        before = stats()
        result = torch.sigmoid(torch.nn.functional.silu(sliced))
        no_host_copies(before)
        torch.testing.assert_close(result.cpu(), torch.sigmoid(torch.nn.functional.silu(expected_s)),
                                   rtol=2e-6, atol=2e-6)
        checks += ["transpose_permute_slice_metadata_match_cpu", "view_creation_zero_dispatch",
                   "strided_gather", "strided_unary_without_host_roundtrip"]

    elif case == "scatter":
        host = torch.arange(35, dtype=torch.float32).reshape(5, 7)
        x = host.to("vulkan")
        before = stats()
        y = x.transpose(0, 1)[1:7:2, 1:5:2]
        expected = host.transpose(0, 1)[1:7:2, 1:5:2]
        alias = torch.ops.aten.alias.default(y)
        layout(y, expected)
        layout(alias, expected)
        assert stats() == before
        replacement = torch.arange(6, dtype=torch.float32).reshape(2, 3).t() + 100
        assert not replacement.is_contiguous()
        old_version = x._version
        alias.copy_(replacement)
        expected.copy_(replacement)
        assert x._version == y._version == alias._version == old_version + 1
        equal(x, host)
        equal(y, expected)
        equal(alias, expected)
        checks += ["cpu_noncontiguous_source_scatter", "untouched_base_elements_preserved",
                   "alias_preserves_stride_offset", "one_shared_version_increment"]

    elif case == "device_copy":
        source_host = torch.arange(35, dtype=torch.float32).reshape(5, 7)
        target_host = torch.full((6, 6), -9.0, dtype=torch.float32)
        source = source_host.to("vulkan")
        target = target_host.to("vulkan")
        src = source[1:4, 1:7:2].t()
        dst = target[1:6:2, 1:4]
        expected_src = source_host[1:4, 1:7:2].t()
        expected_dst = target_host[1:6:2, 1:4]
        layout(src, expected_src)
        layout(dst, expected_dst)
        before = stats()
        version = target._version
        dst.copy_(src)
        no_host_copies(before)
        expected_dst.copy_(expected_src)
        assert target._version == dst._version == version + 1
        equal(target, target_host)
        equal(src, expected_src)
        before = stats()
        cloned = src.clone(memory_format=torch.contiguous_format)
        no_host_copies(before)
        assert cloned.is_contiguous() and cloned.storage_offset() == 0
        assert cloned.stride() == expected_src.contiguous().stride()
        clone_version = cloned._version
        source.copy_(torch.zeros_like(source_host))
        assert cloned._version == clone_version
        equal(cloned, expected_src)
        equal(source, torch.zeros_like(source_host))
        checks += ["gpu_strided_copy_without_host_roundtrip", "gpu_copy_preserves_other_elements",
                   "contiguous_clone_materializes_independently"]

    elif case == "offset_contiguous":
        host = torch.arange(35, dtype=torch.float32).reshape(5, 7)
        target_host = torch.full((5, 7), -1.0, dtype=torch.float32)
        x = host.to("vulkan")
        target = target_host.to("vulkan")
        before = stats()
        src = x[1:3]
        dst = target[2:4]
        flat = src.view(14)
        reshaped = src.reshape(7, 2)
        layout(src, host[1:3])
        layout(dst, target_host[2:4])
        layout(flat, host[1:3].view(14))
        layout(reshaped, host[1:3].reshape(7, 2))
        assert src.is_contiguous() and src.storage_offset() == 7
        assert stats() == before
        equal(flat, host[1:3].view(14))
        before = stats()
        dst.copy_(src)
        no_host_copies(before)
        target_host[2:4].copy_(host[1:3])
        equal(target, target_host)
        dst.copy_(torch.full((2, 7), 42.0))
        target_host[2:4].fill_(42.0)
        equal(target, target_host)
        checks += ["contiguous_nonzero_offset_download", "offset_preserved_by_view_reshape",
                   "contiguous_offset_device_copy", "contiguous_offset_upload"]

    elif case == "detach_data":
        host = torch.arange(35, dtype=torch.float32).reshape(1, 5, 7)
        x = host.to("vulkan")
        before = stats()
        view = x[:, 1:5:2, 1:7:2].transpose(1, 2)
        expected = host[:, 1:5:2, 1:7:2].transpose(1, 2)
        squeezed = view.squeeze(0)
        same = squeezed.squeeze(0)
        detached = squeezed.detach()
        raw = squeezed.data
        for actual, reference in ((view, expected), (squeezed, expected.squeeze(0)),
                                   (same, expected.squeeze(0)), (detached, expected.squeeze(0)),
                                   (raw, expected.squeeze(0))):
            layout(actual, reference)
        assert detached._base is None and stats() == before
        version = x._version
        raw_version = raw._version
        raw.copy_(torch.full(raw.shape, 13.0))
        expected.fill_(13.0)
        assert x._version == squeezed._version == detached._version == version
        assert raw._version == raw_version + 1
        equal(x, host)
        detached.copy_(torch.full(detached.shape, 17.0))
        expected.fill_(17.0)
        assert x._version == squeezed._version == detached._version == version + 1
        equal(x, host)
        other_host = torch.arange(24, dtype=torch.float32).reshape(4, 6)
        other = other_host.to("vulkan")
        replacement = other[1:4, 1:5:2].t()
        rebound_version = detached._version
        detached.data = replacement
        layout(detached, other_host[1:4, 1:5:2].t())
        assert detached._version == rebound_version
        equal(detached, other_host[1:4, 1:5:2].t())
        equal(x, host)
        checks += ["squeeze_keeps_stride_offset", "detach_shares_data_and_version",
                   "data_shares_bytes_with_independent_version", "data_rebind_copies_offset_stride"]

    elif case == "lifetime":
        host = torch.linspace(-2, 2, 60, dtype=torch.float32).reshape(3, 4, 5)
        x = host.to("vulkan")
        d = x.transpose(0, 2)[1:5:2, 1:4:2, 1:3].detach()
        expected = host.transpose(0, 2)[1:5:2, 1:4:2, 1:3]
        layout(d, expected)
        assert d._base is None
        del x
        gc.collect()
        before = stats()
        result = torch.sigmoid(torch.nn.functional.silu(d))
        no_host_copies(before)
        del d
        gc.collect()
        torch.testing.assert_close(result.cpu(), torch.sigmoid(torch.nn.functional.silu(expected)),
                                   rtol=2e-6, atol=2e-6)
        checks += ["strided_detach_survives_base_deletion", "queued_read_keeps_storage_alive",
                   "natural_process_exit_required"]

    elif case == "guards":
        host = torch.arange(25, dtype=torch.float32).reshape(5, 5)
        x = host.to("vulkan")
        transposed = x.t()
        left = x[:2]
        right = x[3:5]
        overlap = x[1:3]
        reject("noncontiguous_view", lambda: transposed.view(25))
        reject("noncontiguous_preserve_clone", lambda: transposed.clone())
        reject("overlapping_different_mapping", lambda: left.copy_(overlap))
        reject("disjoint_same_storage_different_mapping", lambda: left.copy_(right))
        reject("transpose_same_storage_different_mapping", lambda: x.copy_(transposed))
        reject("empty_slice", lambda: x[:, 2:2])
        reject("negative_step", lambda: torch.ops.aten.slice.Tensor(x, 1, 4, 0, -1))
        reject("overflow_step", lambda: torch.ops.aten.slice.Tensor(x, 1, 0, 5, 2**63 - 1))
        reject("as_strided", lambda: torch.as_strided(x, (2, 2), (5, 1), 1))
        equal(x, host)
        identical = x.transpose(0, 0).permute(0, 1)
        layout(identical, host)
        before = stats()
        old_version = x._version
        left.copy_(left.detach())
        assert stats() == before
        assert x._version == left._version == old_version + 1
        equal(x, host)
        checks += ["unsupported_requests_rejected_before_dispatch", "failed_copy_does_not_change_values",
                   "identical_offset_mapping_copy_is_noop_with_version_increment"]

    else:
        raise ValueError(case)

    return {"case": case, "status": "PASS", "checks": checks, "layouts": layouts,
            "rejections": rejections, "stats": stats(), "torch": torch.__version__}


def binary_sha256():
    hashes = {}
    for name in ("torch_cpu.dll", "torch_python.dll"):
        with (ROOT / "torch" / "lib" / name).open("rb") as stream:
            hashes[name] = hashlib.file_digest(stream, "sha256").hexdigest()
    return hashes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--case", choices=CASES)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.child:
        if not args.case:
            parser.error("--child requires --case")
        print(json.dumps(child(args.case)), flush=True)
        return 0
    if not args.run:
        print(json.dumps({"status": "READY", "gpu_workload_started": False, "cases": CASES}))
        return 0
    if args.output and args.output.exists():
        parser.error("choose a new output file")
    selected = [args.case] if args.case else list(CASES)
    initial_hashes = binary_sha256()
    records = []
    env = os.environ.copy()
    env.update(VK_NATIVE_BUFFER="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    for case in selected:
        try:
            result = subprocess.run(
                [sys.executable, "-B", str(Path(__file__).resolve()), "--child", "--case", case],
                cwd=ROOT.parent, env=env, capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=60,
            )
        except subprocess.TimeoutExpired:
            records.append({"case": case, "status": "FAIL", "exit_code": None,
                            "timed_out": True, "natural_exit": False})
            break
        try:
            record = json.loads(result.stdout.strip().splitlines()[-1])
        except (IndexError, ValueError):
            record = {"case": case, "status": "FAIL", "stdout": result.stdout[-4000:]}
        record.update(exit_code=result.returncode, stderr=result.stderr[-4000:],
                      natural_exit=result.returncode == 0)
        records.append(record)
        if result.returncode or record.get("status") != "PASS":
            break
    final_hashes = binary_sha256()
    passed = (len(records) == len(selected) and initial_hashes == final_hashes and
              all(r["status"] == "PASS" and r["exit_code"] == 0 for r in records))
    report = {
        "status": "PASS" if passed else "FAIL", "cases": records,
        "requested_cases": selected, "binary_sha256": initial_hashes,
        "binary_unchanged_during_run": initial_hashes == final_hashes,
        "scope": "bounded FP32 positive-stride views with offsets; ranks 1..4",
        "general_as_strided": False, "native_backward": False, "performance_claim": False,
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
