"""Small FP32 public SDPA compatibility path on Vulkan.

The current implementation is an explicit CPU math fallback with a Vulkan
result.  This test verifies public operator semantics for no-mask, bool mask,
additive mask and causal cases, while unsupported parameters are rejected.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def child():
    os.environ["VK_NATIVE_BUFFER"] = "1"
    sys.path.insert(0, str(ROOT))
    import torch
    import torch.nn.functional as F

    torch.set_num_threads(1)
    shape_q = (1, 2, 3, 4)
    shape_kv = (1, 2, 5, 4)
    q_cpu = torch.linspace(-0.5, 0.5, 24, dtype=torch.float32).reshape(shape_q)
    k_cpu = torch.linspace(0.75, -0.75, 40, dtype=torch.float32).reshape(shape_kv)
    v_cpu = torch.linspace(-1.0, 1.0, 40, dtype=torch.float32).reshape(shape_kv)
    q = q_cpu.to("vulkan")
    k = k_cpu.to("vulkan")
    v = v_cpu.to("vulkan")

    scale = 0.5
    output = F.scaled_dot_product_attention(
        q, k, v, dropout_p=0.0, is_causal=False, scale=scale
    )
    expected = F.scaled_dot_product_attention(
        q_cpu, k_cpu, v_cpu, dropout_p=0.0, is_causal=False, scale=scale
    )
    assert output.device.type == "vulkan"
    torch.testing.assert_close(output.cpu(), expected, rtol=2e-5, atol=2e-5)

    bool_mask_cpu = torch.ones((3, 5), dtype=torch.bool)
    bool_mask_cpu[0, 4] = False
    masked = F.scaled_dot_product_attention(q, k, v, attn_mask=bool_mask_cpu)
    expected_masked = F.scaled_dot_product_attention(
        q_cpu, k_cpu, v_cpu, attn_mask=bool_mask_cpu
    )
    torch.testing.assert_close(masked.cpu(), expected_masked, rtol=2e-5, atol=2e-5)

    additive_mask_cpu = torch.zeros((1, 2, 3, 5), dtype=torch.float32)
    additive_mask_cpu[..., 0] = -0.25
    additive = F.scaled_dot_product_attention(q, k, v, attn_mask=additive_mask_cpu)
    expected_additive = F.scaled_dot_product_attention(
        q_cpu, k_cpu, v_cpu, attn_mask=additive_mask_cpu
    )
    torch.testing.assert_close(additive.cpu(), expected_additive, rtol=2e-5, atol=2e-5)
    vulkan_additive_mask = additive_mask_cpu.to("vulkan")
    vulkan_additive = F.scaled_dot_product_attention(
        q, k, v, attn_mask=vulkan_additive_mask
    )
    torch.testing.assert_close(
        vulkan_additive.cpu(), expected_additive, rtol=2e-5, atol=2e-5
    )

    causal = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    expected_causal = F.scaled_dot_product_attention(q_cpu, k_cpu, v_cpu, is_causal=True)
    torch.testing.assert_close(causal.cpu(), expected_causal, rtol=2e-5, atol=2e-5)

    rejected = []

    def reject(name, fn, message):
        try:
            fn()
        except (RuntimeError, NotImplementedError) as exc:
            assert message in str(exc), (name, str(exc))
            rejected.append(name)
        else:
            raise AssertionError(name + " unexpectedly succeeded")

    reject("dropout", lambda: F.scaled_dot_product_attention(q, k, v, dropout_p=0.1),
           "requires dropout_p=0")
    reject("dropout_negative",
           lambda: F.scaled_dot_product_attention(q, k, v, dropout_p=-0.1),
           "range [0, 1]")
    reject("dropout_above_one",
           lambda: F.scaled_dot_product_attention(q, k, v, dropout_p=1.1),
           "range [0, 1]")
    reject("dropout_nan",
           lambda: F.scaled_dot_product_attention(q, k, v, dropout_p=float("nan")),
           "range [0, 1]")
    q_gqa_cpu = torch.linspace(-0.25, 0.75, 48, dtype=torch.float32).reshape(1, 4, 3, 4)
    q_gqa = q_gqa_cpu.to("vulkan")
    gqa = F.scaled_dot_product_attention(q_gqa, k, v, enable_gqa=True)
    expected_gqa = F.scaled_dot_product_attention(
        q_gqa_cpu, k_cpu, v_cpu, enable_gqa=True
    )
    torch.testing.assert_close(gqa.cpu(), expected_gqa, rtol=2e-5, atol=2e-5)

    q_bad_gqa_cpu = torch.zeros((1, 3, 3, 4), dtype=torch.float32)
    q_bad_gqa = q_bad_gqa_cpu.to("vulkan")
    reject(
        "gqa_non_divisible",
        lambda: F.scaled_dot_product_attention(q_bad_gqa, k, v, enable_gqa=True),
        "query heads to be a multiple of K/V heads",
    )
    bad_mask = torch.zeros((2, 5), dtype=torch.float32).to("vulkan")
    before_bad_mask = dict(torch.ops.vulkan_ssbo.native_buffer_stats())
    reject("bad_mask_shape",
           lambda: F.scaled_dot_product_attention(q, k, v, attn_mask=bad_mask),
           "2D mask must have shape")
    assert dict(torch.ops.vulkan_ssbo.native_buffer_stats()) == before_bad_mask

    q_grad = q.detach().requires_grad_(True)
    reject("requires_grad", lambda: F.scaled_dot_product_attention(q_grad, k, v),
           "inference-only")
    k_grad = k.detach().requires_grad_(True)
    reject("key_requires_grad", lambda: F.scaled_dot_product_attention(q, k_grad, v),
           "inference-only")
    v_grad = v.detach().requires_grad_(True)
    reject("value_requires_grad", lambda: F.scaled_dot_product_attention(q, k, v_grad),
           "inference-only")
    mask_grad = additive_mask_cpu.detach().requires_grad_(True)
    reject(
        "mask_requires_grad",
        lambda: F.scaled_dot_product_attention(q, k, v, attn_mask=mask_grad),
        "inference mask",
    )
    with torch.no_grad():
        reject("requires_grad_no_grad", lambda: F.scaled_dot_product_attention(q_grad, k, v),
               "inference-only")

    return {
        "status": "PASS",
        "torch": torch.__version__,
        "output_shape": list(output.shape),
        "max_error": float((output.cpu() - expected).abs().max()),
        "mask_max_error": float((masked.cpu() - expected_masked).abs().max()),
        "additive_mask_max_error": float((additive.cpu() - expected_additive).abs().max()),
        "vulkan_mask_max_error": float(
            (vulkan_additive.cpu() - expected_additive).abs().max()
        ),
        "causal_max_error": float((causal.cpu() - expected_causal).abs().max()),
        "gqa_max_error": float((gqa.cpu() - expected_gqa).abs().max()),
        "rejected": rejected,
        "implementation": "cpu_math_fallback_to_vulkan",
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
                          "scope": "FP32 rank-4 public SDPA compatibility"}))
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
        "scope": "bounded UMA FP32 public SDPA compatibility",
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
