"""Native Vulkan SSBO regression checks.

Run from the source tree with the matching Python environment.  Convolution
references use float64 CPU kernels because the local float32 CPU convolution
implementation is not reliable for several grouped/dilated shapes.
"""

from __future__ import annotations

import os
import sys
import ctypes
from pathlib import Path

# Prefer the just-built source-tree package when this file is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch


def manual_conv1d(
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    stride: int,
    padding: int,
    dilation: int,
    groups: int,
) -> torch.Tensor:
    batch, channels, length = x.shape
    outputs, channels_per_group, kernel = w.shape
    outputs_per_group = outputs // groups
    out_length = (length + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1
    out = torch.empty((batch, outputs, out_length), dtype=torch.float64)
    for n in range(batch):
        for o in range(outputs):
            group = o // outputs_per_group
            for q in range(out_length):
                value = float(b[o])
                for i in range(channels_per_group):
                    channel = group * channels_per_group + i
                    for k in range(kernel):
                        index = q * stride + k * dilation - padding
                        if 0 <= index < length:
                            value += float(x[n, channel, index]) * float(w[o, i, k])
                out[n, o, q] = value
    return out


def manual_conv2d(
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    stride: tuple[int, int] = (1, 1),
    padding: tuple[int, int] = (0, 0),
    dilation: tuple[int, int] = (1, 1),
    groups: int = 1,
) -> torch.Tensor:
    batch, channels, height, width = x.shape
    outputs, channels_per_group, kh, kw = w.shape
    sh, sw = stride
    ph, pw = padding
    dh, dw = dilation
    ho = (height + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    wo = (width + 2 * pw - dw * (kw - 1) - 1) // sw + 1
    outputs_per_group = outputs // groups
    out = torch.empty((batch, outputs, ho, wo), dtype=torch.float64)
    for n in range(batch):
        for o in range(outputs):
            group = o // outputs_per_group
            for oh in range(ho):
                for ow in range(wo):
                    value = float(b[o])
                    for i in range(channels_per_group):
                        channel = group * channels_per_group + i
                        for iy in range(kh):
                            for ix in range(kw):
                                h = oh * sh + iy * dh - ph
                                widx = ow * sw + ix * dw - pw
                                if 0 <= h < height and 0 <= widx < width:
                                    value += float(x[n, channel, h, widx]) * float(w[o, i, iy, ix])
                    out[n, o, oh, ow] = value
    return out


def check(name: str, actual: torch.Tensor, expected: torch.Tensor, atol: float = 2e-2) -> None:
    diff = (actual.double() - expected.double()).abs()
    maximum = float(diff.max()) if diff.numel() else 0.0
    print(f"{name}: max_abs={maximum:.6f}")
    if maximum > atol:
        raise AssertionError(f"{name}: max_abs {maximum} > {atol}")


def main() -> None:
    torch.manual_seed(42)
    with torch.inference_mode():
        info = dict(torch.ops.vulkan_ssbo.device_info())
        if not info.get("device_name"):
            raise AssertionError("Vulkan SSBO device_info returned no device name")
        if info.get("memory_topology") not in {"uma", "discrete", "mixed", "unknown"}:
            raise AssertionError(f"invalid memory topology: {info.get('memory_topology')}")
        if int(info.get("device_local_budget", "0")) <= 0:
            raise AssertionError("Vulkan SSBO reported no device-local memory budget")
        print(
            f"device_info: {info['device_name']} shared={info['max_shared_memory']} "
            f"topology={info['memory_topology']} "
            f"device_local_budget={info['device_local_budget']}"
        )

        copy5 = torch.arange(2 * 3 * 2 * 5 * 7, dtype=torch.float32).reshape(2, 3, 2, 5, 7)
        check("5d buffer roundtrip", copy5.to("vulkan").cpu(), copy5, atol=0.0)

        # Compare against the values actually stored by stock Vulkan so this
        # isolates tiled SSBO matmul correctness from the initial FP16 upload.
        ma = torch.randn(96, 160).to("vulkan")
        mb = torch.randn(160, 128).to("vulkan")
        ma_ref, mb_ref = ma.cpu().double(), mb.cpu().double()
        check("mm tiled", torch.mm(ma, mb).cpu(), ma_ref @ mb_ref, atol=3e-2)

        ba = torch.randn(3, 65, 96).to("vulkan")
        bb = torch.randn(3, 96, 80).to("vulkan")
        ba_ref, bb_ref = ba.cpu().double(), bb.cpu().double()
        check("bmm tiled", torch.bmm(ba, bb).cpu(), torch.bmm(ba_ref, bb_ref), atol=3e-2)

        linear_x = torch.randn(2, 33, 160).to("vulkan")
        linear_w = torch.randn(128, 160).to("vulkan")
        linear_b = torch.randn(128).to("vulkan")
        linear_ref = linear_x.cpu().double() @ linear_w.cpu().double().t()
        check(
            "linear tiled",
            torch.nn.functional.linear(linear_x, linear_w).cpu(),
            linear_ref,
            atol=3e-2,
        )
        previous_bias_fusion = os.environ.get("VKSSBO_MATMUL_BIAS_FUSION")
        try:
            os.environ["VKSSBO_MATMUL_BIAS_FUSION"] = "0"
            linear_bias_fallback = torch.nn.functional.linear(
                linear_x, linear_w, linear_b
            ).cpu()
            os.environ["VKSSBO_MATMUL_BIAS_FUSION"] = "1"
            linear_bias_fused = torch.nn.functional.linear(
                linear_x, linear_w, linear_b
            ).cpu()
        finally:
            if previous_bias_fusion is None:
                os.environ.pop("VKSSBO_MATMUL_BIAS_FUSION", None)
            else:
                os.environ["VKSSBO_MATMUL_BIAS_FUSION"] = previous_bias_fusion
        check(
            "linear fused bias",
            linear_bias_fused,
            linear_ref + linear_b.cpu().double(),
            atol=3e-2,
        )
        # Fused bias rounds once after the addition; the fallback rounds the
        # matmul output before adding on CPU. Both paths retain the stricter
        # reference check above, while their two rounding orders can differ by
        # one FP16 step at this output magnitude.
        check("linear bias fusion A/B", linear_bias_fused, linear_bias_fallback, atol=4e-2)

        x1 = torch.rand(2, 20, 30)
        w1 = torch.rand(10, 4, 6)
        b1 = torch.rand(10)
        args1 = dict(stride=5, padding=5, dilation=3, groups=5)
        check(
            "conv1d grouped/strided/dilated",
            torch.conv1d(x1.to("vulkan"), w1.to("vulkan"), b1.to("vulkan"), **args1).cpu(),
            manual_conv1d(x1.double(), w1.double(), b1.double(), **args1),
        )

        x2 = torch.rand(1, 15, 30, 31)
        w2 = torch.rand(25, 3, 6, 5)
        b2 = torch.rand(25)
        args2 = dict(stride=(2, 3), padding=(5, 4), dilation=(3, 2), groups=5)
        check(
            "conv2d grouped/strided/dilated",
            torch.conv2d(x2.to("vulkan"), w2.to("vulkan"), b2.to("vulkan"), **args2).cpu(),
            torch.conv2d(x2.double(), w2.double(), b2.double(), **args2),
            atol=4e-2,
        )

        # Exercise both input-channel chunk sizes of the native 32-output-
        # channel fast path, including non-multiple-of-16 spatial extents.
        x2_fast = torch.rand(2, 32, 17, 19)
        w2_fast = torch.rand(64, 32, 3, 3)
        b2_fast = torch.rand(64)
        x2_fast_vk = x2_fast.to("vulkan")
        w2_fast_vk = w2_fast.to("vulkan")
        b2_fast_vk = b2_fast.to("vulkan")
        previous_c32 = os.environ.get("VKSSBO_CONV_C32")
        previous_v4 = os.environ.get("VKSSBO_CONV_V4")
        previous_v4acc = os.environ.get("VKSSBO_CONV_V4ACC")
        try:
            os.environ["VKSSBO_CONV_C32"] = "0"
            os.environ["VKSSBO_CONV_V4"] = "0"
            os.environ["VKSSBO_CONV_V4ACC"] = "0"
            y2_fast_scalar = torch.conv2d(
                x2_fast_vk, w2_fast_vk, b2_fast_vk, padding=1
            ).cpu()
            os.environ["VKSSBO_CONV_C32"] = "1"
            y2_fast_c32 = torch.conv2d(
                x2_fast_vk, w2_fast_vk, b2_fast_vk, padding=1
            ).cpu()
            os.environ["VKSSBO_CONV_C32"] = "0"
            os.environ["VKSSBO_CONV_V4"] = "1"
            os.environ["VKSSBO_CONV_V4ACC"] = "0"
            y2_fast_v4 = torch.conv2d(
                x2_fast_vk, w2_fast_vk, b2_fast_vk, padding=1
            ).cpu()
            os.environ["VKSSBO_CONV_V4ACC"] = "1"
            y2_fast_v4acc = torch.conv2d(
                x2_fast_vk, w2_fast_vk, b2_fast_vk, padding=1
            ).cpu()
        finally:
            if previous_c32 is None:
                os.environ.pop("VKSSBO_CONV_C32", None)
            else:
                os.environ["VKSSBO_CONV_C32"] = previous_c32
            if previous_v4 is None:
                os.environ.pop("VKSSBO_CONV_V4", None)
            else:
                os.environ["VKSSBO_CONV_V4"] = previous_v4
            if previous_v4acc is None:
                os.environ.pop("VKSSBO_CONV_V4ACC", None)
            else:
                os.environ["VKSSBO_CONV_V4ACC"] = previous_v4acc
        r2_fast = manual_conv2d(
            x2_fast.double(), w2_fast.double(), b2_fast.double(), padding=(1, 1)
        )
        # The SSBO path stores activations and weights as fp16; the wider
        # accumulation keeps the mean error low, while the max error on
        # positive [0, 1] inputs can reach roughly 0.1.
        check("conv2d x32 scalar fast path", y2_fast_scalar, r2_fast, atol=1e-1)
        check("conv2d x32 v4 fast path", y2_fast_v4, r2_fast, atol=1e-1)
        check("conv2d x32 vec4-acc fast path", y2_fast_v4acc, r2_fast, atol=1e-1)
        check("conv2d x32 C32 fast path", y2_fast_c32, r2_fast, atol=1e-1)
        check("conv2d scalar/v4 agreement", y2_fast_v4, y2_fast_scalar, atol=1e-3)
        check("conv2d v4/vec4-acc agreement", y2_fast_v4acc, y2_fast_v4, atol=1e-3)
        check("conv2d scalar/C32 agreement", y2_fast_c32, y2_fast_scalar, atol=1e-3)

        x3 = torch.rand(1, 6, 4, 8, 9)
        w3 = torch.rand(4, 3, 3, 3, 3)
        y3 = torch.conv3d(
            x3.to("vulkan"), w3.to("vulkan"), None,
            stride=(2, 1, 2), padding=1, groups=2,
        ).cpu()
        r3 = torch.conv3d(
            x3.double(), w3.double(), None,
            stride=(2, 1, 2), padding=1, groups=2,
        )
        check("conv3d grouped/strided", y3, r3, atol=3e-2)

        x4 = torch.rand(1, 4, 7, 8, 9)
        w4 = torch.rand(6, 2, 2, 3, 2)
        b4 = torch.rand(6)
        args4 = dict(stride=(2, 1, 2), padding=(2, 1, 2), dilation=(2, 2, 2), groups=2)
        y4 = torch.conv3d(x4.to("vulkan"), w4.to("vulkan"), b4.to("vulkan"), **args4).cpu()
        r4 = torch.conv3d(x4.double(), w4.double(), b4.double(), **args4)
        check("conv3d grouped/strided/dilated", y4, r4, atol=4e-2)

        xu = torch.rand(1, 3, 4, 5)
        for align in (False, True):
            yu = torch.nn.functional.interpolate(
                xu.to("vulkan"), size=(7, 9), mode="bilinear", align_corners=align
            ).cpu()
            ru = torch.nn.functional.interpolate(
                xu, size=(7, 9), mode="bilinear", align_corners=align
            )
            check(f"bilinear2d align_corners={align}", yu, ru, atol=3e-2)

        xg = torch.rand(2, 12, 3, 5, 7)
        wg = torch.rand(12)
        bg = torch.rand(12)
        yg = torch.group_norm(xg.to("vulkan"), 4, wg.to("vulkan"), bg.to("vulkan"), 1e-5).cpu()
        rg = torch.group_norm(xg.double(), 4, wg.double(), bg.double(), 1e-5)
        check("group_norm 5d", yg, rg, atol=3e-2)

        # A/B the resident bridge across the three representative chains used
        # by the Stage C acceptance criteria. Disabled mode is the materialize
        # fallback; enabled mode must agree and register four intermediate hits.
        previous_lease = os.environ.get("VKSSBO_HOST_LEASE")
        try:
            conv_x = torch.rand(1, 8, 9, 11).to("vulkan")
            conv_w1 = torch.rand(8, 8, 3, 3).to("vulkan")
            conv_b1 = torch.rand(8).to("vulkan")
            conv_w2 = torch.rand(6, 8, 3, 3).to("vulkan")
            conv_b2 = torch.rand(6).to("vulkan")
            chain_x = torch.randn(2, 17, 64).to("vulkan")
            chain_w = torch.randn(48, 64).to("vulkan")
            chain_b = torch.randn(48).to("vulkan")
            chain_a = torch.randn(2, 33, 64).to("vulkan")
            chain_m = torch.randn(2, 64, 40).to("vulkan")

            def run_chains(enabled: bool):
                os.environ["VKSSBO_HOST_LEASE"] = "1" if enabled else "0"
                torch.ops.vulkan_ssbo.empty_cache()
                before = dict(torch.ops.vulkan_ssbo.memory_stats())
                conv_first = torch.conv2d(
                    conv_x, conv_w1, conv_b1, padding=1
                )
                conv_activated = torch.nn.functional.silu(conv_first)
                conv_out = torch.conv2d(
                    conv_activated, conv_w2, conv_b2, padding=1
                ).cpu()
                linear_out = torch.nn.functional.silu(
                    torch.nn.functional.linear(chain_x, chain_w, chain_b)
                ).cpu()
                softmax_out = torch.softmax(torch.bmm(chain_a, chain_m), -1).cpu()
                after = dict(torch.ops.vulkan_ssbo.memory_stats())
                return (conv_out, linear_out, softmax_out), (
                    int(after["resident_hits"]) - int(before["resident_hits"])
                )

            fallback_outputs, fallback_hits = run_chains(False)
            resident_outputs, resident_hits = run_chains(True)
            check("resident conv-silu-conv A/B", resident_outputs[0], fallback_outputs[0], atol=2e-2)
            check("resident linear-bias-silu A/B", resident_outputs[1], fallback_outputs[1], atol=2e-2)
            check("resident matmul-softmax A/B", resident_outputs[2], fallback_outputs[2], atol=2e-2)
            if fallback_hits != 0 or resident_hits < 4:
                raise AssertionError(
                    f"unexpected resident hits: fallback={fallback_hits} enabled={resident_hits}"
                )
            print(
                f"resident chains: fallback_hits={fallback_hits} "
                f"enabled_hits={resident_hits}"
            )
            torch.ops.vulkan_ssbo.empty_cache()
        finally:
            if previous_lease is None:
                os.environ.pop("VKSSBO_HOST_LEASE", None)
            else:
                os.environ["VKSSBO_HOST_LEASE"] = previous_lease

    print("native Vulkan SSBO regression: PASS")
    # The stock Vulkan context has a known teardown failure on this Windows
    # build. Exit before this function frame releases its local Vulkan tensors;
    # every result above has already synchronized through .cpu().
    sys.stdout.flush()
    sys.stderr.flush()
    if os.name == "nt":
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.TerminateProcess.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        kernel32.TerminateProcess.restype = ctypes.c_int
        kernel32.TerminateProcess(kernel32.GetCurrentProcess(), 0)
    os._exit(0)


if __name__ == "__main__":
    main()
