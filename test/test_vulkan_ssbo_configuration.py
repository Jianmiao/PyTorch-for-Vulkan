"""Contract test for the Vulkan SSBO configuration registry."""

from __future__ import annotations

from torch.vulkan_ssbo_configuration import REGISTRY, configuration_snapshot


def main() -> None:
    names = [entry["name"] for entry in REGISTRY]
    assert len(names) == len(set(names)), "configuration names must be unique"
    assert all(name.startswith("VKSSBO_") for name in names), names
    assert {entry["status"] for entry in REGISTRY} <= {
        "stable", "experimental", "diagnostic"
    }
    assert {entry["owner"] for entry in REGISTRY} <= {"runtime", "comfyui"}
    assert {
        "VKSSBO_PROFILE",
        "VKSSBO_DEVICE_NAME",
        "VKSSBO_TRANSIENT_POOL",
        "VKSSBO_STAGING_POOL",
        "VKSSBO_WEIGHT_CACHE",
        "VKSSBO_ATTENTION",
    } <= set(names)

    snapshot = configuration_snapshot({"VKSSBO_PROFILE": "1"})
    assert snapshot["schema_version"] == 1, snapshot
    by_name = {item["name"]: item for item in snapshot["settings"]}
    assert by_name["VKSSBO_PROFILE"]["current"] == "1", by_name
    assert by_name["VKSSBO_PROFILE"]["explicit"] is True, by_name
    assert by_name["VKSSBO_STAGING_POOL"]["current"] == "1", by_name
    assert by_name["VKSSBO_STAGING_POOL"]["explicit"] is False, by_name
    print(f"Vulkan SSBO configuration registry: PASS entries={len(REGISTRY)}")


if __name__ == "__main__":
    main()
