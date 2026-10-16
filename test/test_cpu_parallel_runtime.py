"""NumPy-referenced CPU parallel regression, including OpenMP row ownership.

Run in a fresh process. No Vulkan initialization or application patches.
Checks every repeat, rather than only the last allocation's contents.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=12)
    args = parser.parse_args()
    if args.output.exists() or not 1 <= args.repeats <= 64:
        parser.error('choose a new output path; repeats must be 1..64')
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    import numpy as np
    import torch
    import torch.nn.functional as F

    assert Path(torch.__file__).resolve().parent == root / 'torch'
    rng = np.random.default_rng(104)
    checks = []

    def check(name, fn, expected, tolerance=3e-5):
        maximum = 0.0
        passed = True
        for _ in range(args.repeats):
            actual = fn().detach().float().numpy().copy()
            error = float(np.max(np.abs(actual - expected)))
            maximum = max(maximum, error)
            passed = bool(np.isfinite(actual).all()) and bool(
                np.allclose(actual, expected, rtol=tolerance, atol=tolerance)) and passed
        checks.append(dict(threads=torch.get_num_threads(), name=name,
                           passed=passed, max_abs=maximum))

    for threads in (1, 2, 4, 14):
        torch.set_num_threads(threads)
        for shape, axis in (((32, 12), -1), ((9, 16), -1),
                            ((33, 65), -1), ((4, 9, 13), 1)):
            for dtype in (torch.float32, torch.float64, torch.float16, torch.bfloat16):
                x = torch.from_numpy(rng.normal(size=shape)).to(dtype)
                n = x.double().numpy().copy()
                shifted = n - n.max(axis=axis, keepdims=True)
                exp = np.exp(shifted)
                probability = exp / exp.sum(axis=axis, keepdims=True)
                logprob = shifted - np.log(exp.sum(axis=axis, keepdims=True))
                tolerance = 2e-2 if dtype == torch.bfloat16 else 2e-3 if dtype == torch.float16 else 3e-5
                prefix = f'{shape}/{axis}/{dtype}'
                check(prefix + '/softmax', lambda: torch.softmax(x, axis), probability, tolerance)
                check(prefix + '/log_softmax', lambda: torch.log_softmax(x, axis), logprob, tolerance)
                grad = torch.from_numpy(rng.normal(size=shape)).to(dtype)
                gn = grad.double().numpy().copy()
                # Saved output has storage dtype; mirror that rounding only.
                stored = torch.from_numpy(probability).to(dtype)
                pn = stored.double().numpy().copy()
                expected_grad = pn * (gn - (pn * gn).sum(axis=axis, keepdims=True))
                check(prefix + '/backward', lambda: torch.ops.aten._softmax_backward_data(
                    grad, stored, axis, dtype), expected_grad, tolerance)
        x = torch.from_numpy(rng.normal(size=(128, 512)).astype(np.float32))
        n = x.double().numpy().copy()
        check('sum', lambda: x.sum(1), n.sum(1), 1e-4)
        check('mean', lambda: x.mean(1), n.mean(1))
        mean = n.mean(1, keepdims=True)
        norm = (n - mean) / np.sqrt(((n - mean)**2).mean(1, keepdims=True) + 1e-5)
        check('layer_norm', lambda: F.layer_norm(x, (512,)), norm)

    dll = root / 'torch/lib/torch_cpu.dll'
    report = dict(status='PASS' if all(c['passed'] for c in checks) else 'FAIL',
                  torch=torch.__file__, repeats=args.repeats, checks=checks,
                  torch_cpu_sha256=hashlib.sha256(dll.read_bytes()).hexdigest(),
                  gpu_initialized=False, numerical_reference='numpy_float64')
    with args.output.open('x', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    print(json.dumps(dict(status=report['status'], checks=len(checks),
                         failed=[c for c in checks if not c['passed']], output=str(args.output))))
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
