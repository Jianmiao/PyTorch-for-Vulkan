#ifndef VKSSBO_MEMORY_POLICY_H
#define VKSSBO_MEMORY_POLICY_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// A snapshot contains only accounting data. Rust never owns Vulkan handles or
// VMA allocations; it returns policy decisions for the existing C++ owner.
typedef struct VkssboMemorySnapshot {
  uint64_t heap_budget;
  uint64_t heap_usage;
  uint64_t weight_cache_bytes;
  uint64_t transient_pool_bytes;
  uint64_t resident_bytes;
  uint64_t safety_reserve;
  uint64_t fallback_weight_limit;
  uint64_t fallback_pool_limit;
  uint64_t weight_cache_cap;
  uint64_t transient_pool_cap;
  uint64_t requested_bytes;
  uint32_t request_kind;
  uint32_t memory_topology;
  uint32_t has_device_local_only;
  uint32_t has_host_visible_device_local;
  uint32_t has_host_visible_system;
  uint64_t device_local_budget;
  uint64_t device_local_usage;
  uint64_t device_local_only_budget;
  uint64_t device_local_only_usage;
  uint64_t host_visible_device_local_budget;
  uint64_t host_visible_device_local_usage;
  uint64_t host_visible_system_budget;
  uint64_t host_visible_system_usage;
  uint32_t reserved;
} VkssboMemorySnapshot;

typedef struct VkssboMemoryPlan {
  uint64_t safe_budget;
  uint64_t available_bytes;
  uint64_t weight_cache_limit;
  uint64_t transient_pool_limit;
  uint64_t resident_limit;
  uint64_t non_managed_bytes;
  uint32_t dynamic_budget;
  uint32_t allow_new_allocation;
  uint32_t pressure_permille;
  uint32_t recommended_evict_kind;
  uint32_t memory_topology;
  uint32_t prefer_host_visible;
  uint32_t prefer_device_local;
  uint32_t reserved;
} VkssboMemoryPlan;

// request_kind: 0 generic, 1 weight cache, 2 transient pool, 3 resident.
// A cap of UINT64_MAX means no user cap. The function is deterministic and
// does not allocate, call Vulkan, or retain pointers to the snapshot.
int vkssbo_memory_plan(const VkssboMemorySnapshot* snapshot,
                       VkssboMemoryPlan* plan);

uint32_t vkssbo_memory_policy_version(void);

#ifdef __cplusplus
}
#endif

#endif
