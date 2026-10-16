"""Authoritative configuration registry for the Vulkan SSBO backend."""

from __future__ import annotations

import os
from typing import Mapping


SCHEMA_VERSION = 1

_VALID_STATUS = {"stable", "experimental", "diagnostic"}
_VALID_OWNER = {"runtime", "comfyui"}


def _entry(
    name: str,
    default: str,
    status: str,
    owner: str,
    description: str,
) -> dict[str, str]:
    if status not in _VALID_STATUS or owner not in _VALID_OWNER:
        raise ValueError(f"invalid Vulkan SSBO registry entry: {name}")
    return {
        "name": name,
        "default": default,
        "status": status,
        "owner": owner,
        "description": description,
    }


REGISTRY = (
    _entry("VKSSBO_DEVICE_NAME", "", "stable", "runtime", "Select a Vulkan device by name."),
    _entry("VKSSBO_DEVICE_INDEX", "", "stable", "runtime", "Select a Vulkan device by index when no name is set."),
    _entry("VKSSBO_PROFILE", "0", "diagnostic", "runtime", "Collect structured host and GPU runtime statistics."),
    _entry("VKSSBO_VALIDATE", "0", "diagnostic", "runtime", "Enable the Khronos Vulkan validation layer."),
    _entry("VKSSBO_ASYNC_TRANSFER", "0", "experimental", "runtime", "Use a dedicated transfer queue when supported."),
    _entry("VKSSBO_ASYNC_TRANSFER_MIN_BYTES", "262144", "experimental", "runtime", "Minimum transfer size for the dedicated queue."),
    _entry("VKSSBO_HOST_LEASE", "1", "stable", "runtime", "Reuse resident SSBO inputs through version-checked host leases."),
    _entry("VKSSBO_TRANSIENT_POOL", "1", "stable", "runtime", "Enable the transient device-buffer pool."),
    _entry("VKSSBO_TRANSIENT_POOL_MIN_KB", "64", "stable", "runtime", "Minimum allocation size eligible for the transient pool."),
    _entry("VKSSBO_TRANSIENT_POOL_MB", "auto", "experimental", "runtime", "Optional transient-pool capacity cap."),
    _entry("VKSSBO_STAGING_POOL", "1", "stable", "runtime", "Enable the host-visible staging-buffer pool."),
    _entry("VKSSBO_STAGING_POOL_MIN_KB", "64", "stable", "runtime", "Minimum allocation size eligible for the staging pool."),
    _entry("VKSSBO_STAGING_POOL_MB", "256", "stable", "runtime", "Maximum cached staging memory in MiB."),
    _entry("VKSSBO_WEIGHT_CACHE", "1", "stable", "runtime", "Enable the persistent weight-buffer cache."),
    _entry("VKSSBO_WEIGHT_CACHE_MB", "auto", "experimental", "runtime", "Optional weight-cache capacity cap."),
    _entry("VKSSBO_RESIDENT_CACHE", "1", "experimental", "runtime", "Keep output SSBOs resident across compatible operators."),
    _entry("VKSSBO_RESIDENT_CACHE_MB", "auto", "experimental", "runtime", "Optional resident-output cache capacity cap."),
    _entry("VKSSBO_AUTOTUNE", "1", "experimental", "runtime", "Enable eligible matmul and convolution autotuning."),
    _entry("VKSSBO_MATMUL_AUTOTUNE", "1", "experimental", "runtime", "Legacy matmul-only autotune control."),
    _entry("VKSSBO_CONV_AUTOTUNE", "1", "experimental", "runtime", "Enable eligible convolution autotuning."),
    _entry("VKSSBO_AUTOTUNE_SMALL", "0", "diagnostic", "runtime", "Allow autotuning below the normal work threshold."),
    _entry("VKSSBO_AUTOTUNE_PERSIST", "1", "experimental", "runtime", "Persist autotune choices between processes."),
    _entry("VKSSBO_AUTOTUNE_CACHE", "auto", "diagnostic", "runtime", "Override the matmul autotune cache path."),
    _entry("VKSSBO_CONV_AUTOTUNE_CACHE", "auto", "diagnostic", "runtime", "Override the convolution autotune cache path."),
    _entry("VKSSBO_MATMUL_128", "auto", "diagnostic", "runtime", "Force the 64 or 128 matmul tile variant."),
    _entry("VKSSBO_CONV_X32", "auto", "diagnostic", "runtime", "Force the scalar x32 convolution candidate."),
    _entry("VKSSBO_CONV_C32", "auto", "diagnostic", "runtime", "Force the C32 convolution candidate."),
    _entry("VKSSBO_CONV_V4", "auto", "diagnostic", "runtime", "Force the vec4 convolution candidate."),
    _entry("VKSSBO_CONV_V4ACC", "auto", "diagnostic", "runtime", "Force the vec4-accumulate convolution candidate."),
    _entry("VKSSBO_MATMUL_BIAS_FUSION", "1", "experimental", "runtime", "Fuse matmul and bias when supported."),
    _entry("VKSSBO_MATMUL_SILU_FUSION", "1", "experimental", "runtime", "Fuse matmul, bias, and SiLU when supported."),
    _entry("VKSSBO_MATMUL_GELU_FUSION", "1", "experimental", "runtime", "Fuse matmul, bias, and GELU when supported."),
    _entry("VKSSBO_MATMUL_SOFTMAX_FUSION", "1", "experimental", "runtime", "Fuse scaled matmul and softmax when supported."),
    _entry("VKSSBO_SIMULATE_DISCRETE", "0", "diagnostic", "runtime", "Apply discrete-GPU memory policy on the local adapter."),
    _entry("VKSSBO_SIMULATED_VRAM_MB", "4096", "diagnostic", "runtime", "Virtual VRAM budget used by discrete simulation."),
    _entry("VKSSBO_SIMULATION_FORCE_STAGING", "auto", "diagnostic", "runtime", "Force staging transfers during discrete simulation."),
    _entry("VKSSBO_SIMULATION_STRICT_BUDGET", "0", "diagnostic", "runtime", "Reject simulated allocations beyond the virtual budget."),
    _entry("VKSSBO_SIMULATION_TRANSFER_LATENCY_US", "0", "diagnostic", "runtime", "Inject transfer latency during discrete simulation."),
    _entry("VKSSBO_TRACE_SUBMIT", "0", "diagnostic", "runtime", "Print queue submission diagnostics."),
    _entry("VKSSBO_TRACE_STAGING_POOL", "0", "diagnostic", "runtime", "Print staging-pool diagnostics."),
    _entry("VKSSBO_PYTHON_DIR", "bundled", "stable", "comfyui", "Locate the Python compatibility adapter."),
    _entry("VKSSBO_ACT_FP16", "0", "experimental", "comfyui", "Use fp16 host activation mirrors."),
    _entry("VKSSBO_ACT_F32_OUT", "0", "experimental", "comfyui", "Request float32 activation outputs in fp16 mode."),
    _entry("VKSSBO_RETURN_CPU", "0", "diagnostic", "comfyui", "Keep eager operator results on the CPU for A/B testing."),
    _entry("VKSSBO_ATTENTION", "ssbo", "experimental", "comfyui", "Select SSBO attention or PyTorch SDPA."),
    _entry("VKSSBO_FUSED_XATTN", "1", "experimental", "comfyui", "Enable fused cross-attention when eligible."),
    _entry("VKSSBO_FUSED_XATTN_FP16", "0", "experimental", "comfyui", "Allow fused cross-attention in fp16 mode."),
    _entry("VKSSBO_LINEAR_ACTIVATION", "1", "experimental", "comfyui", "Enable linear plus activation fusion in model adapters."),
    _entry("VKSSBO_RESIDENT_ACTIVATIONS", "0", "experimental", "comfyui", "Keep bounded native activation sidecars."),
    _entry("VKSSBO_RESIDENT_MAX_MB", "512", "experimental", "comfyui", "Maximum activation-sidecar capacity in MiB."),
    _entry("VKSSBO_RESIDENT_RESBLOCK_GRAPH", "1", "experimental", "comfyui", "Fuse eligible standard 2D ResBlocks into one resident graph submission."),
    _entry("VKSSBO_NANCHECK", "0", "diagnostic", "comfyui", "Print tensor-shape diagnostics for numerical investigation."),
)


def configuration_snapshot(
    environ: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Return defaults and the effective process values in a stable schema."""
    values = os.environ if environ is None else environ
    settings = []
    for entry in REGISTRY:
        item: dict[str, object] = dict(entry)
        item["current"] = values.get(entry["name"], entry["default"])
        item["explicit"] = entry["name"] in values
        settings.append(item)
    return {"schema_version": SCHEMA_VERSION, "settings": settings}


__all__ = ["REGISTRY", "SCHEMA_VERSION", "configuration_snapshot"]
