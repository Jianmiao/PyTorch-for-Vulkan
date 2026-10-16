// vulkan_ssbo: standalone SSBO-based Vulkan compute backend for PyTorch.
// No texture storage: all tensors live in linear Shader Storage Buffers,
// which have no dimension limits (fixes maxImageDimension3D=16384 crashes).
#include <vulkan/vulkan.h>
#include <include/vk_mem_alloc.h>
#include <vector>
#include <array>
#include <algorithm>
#include <atomic>
#include <list>
#include <string>
#include <mutex>
#include <unordered_map>
#include <limits>
#include <chrono>
#include <memory>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cctype>
#include <stdexcept>
#include <thread>

#ifdef VKSSBO_USE_RUST_MEMORY_POLICY
#include "memory_policy.h"
#endif

#ifndef VKSSBO_EXPORT
#ifdef _WIN32
#define VKSSBO_EXPORT __declspec(dllexport)
#else
#define VKSSBO_EXPORT
#endif
#endif

namespace vkssbo {

// ---------- Error ----------
class Error : public std::runtime_error {
public:
  explicit Error(const std::string& m) : std::runtime_error(m) {}
};
#define VKSSBO_CHECK(expr)                                                     \
  do {                                                                         \
    VkResult _r = (expr);                                                      \
    if (_r != VK_SUCCESS) {                                                    \
      char _buf[128];                                                          \
      snprintf(_buf, sizeof(_buf), "%s failed: %d", #expr, (int)_r);           \
      throw Error(_buf);                                                       \
    }                                                                          \
  } while (0)

// ---------- Device ----------
struct Device {
  static constexpr uint32_t kTimestampQueryCount = 32;
  static constexpr VkDeviceSize kDefaultAsyncTransferMinBytes = 256 * 1024;
  enum class MemoryTopology : uint32_t {
    UNKNOWN = 0,
    UMA = 1,
    DISCRETE = 2,
    MIXED = 3,
  };

  VkInstance instance = VK_NULL_HANDLE;
  VkPhysicalDevice phys = VK_NULL_HANDLE;
  uint32_t physical_device_count = 0;
  uint32_t selected_device_index = 0;
  std::string device_selection = "first";
  VkDevice device = VK_NULL_HANDLE;
  uint32_t queue_family = 0;
  VkQueueFlags compute_queue_flags = 0;
  uint32_t transfer_queue_family = UINT32_MAX;
  VkQueueFlags transfer_queue_flags = 0;
  VkQueue queue = VK_NULL_HANDLE;
  VkQueue transfer_queue = VK_NULL_HANDLE;
  VmaAllocator allocator = VK_NULL_HANDLE;
  VkDescriptorPool desc_pool = VK_NULL_HANDLE;
  VkCommandPool cmd_pool = VK_NULL_HANDLE;
  VkCommandPool transfer_cmd_pool = VK_NULL_HANDLE;
  VkSemaphore transfer_timeline = VK_NULL_HANDLE;
  uint64_t transfer_timeline_value = 0;
  VkDescriptorSetLayout global_ds_layout = VK_NULL_HANDLE;
  VkQueryPool timestamp_pool = VK_NULL_HANDLE;
  bool has_fp16 = false;
  bool has_int64 = false;
  bool has_shader_f16 = false;
  bool has_shader_int8 = false;
  bool has_storage_buffer_8bit = false;
  bool has_uniform_and_storage_buffer_8bit = false;
  bool has_storage_push_constant_8bit = false;
  bool has_memory_budget = false;
  bool compute_queue_supported = false;
  bool transfer_queue_supported = false;
  bool dedicated_transfer_queue_available = false;
  bool timeline_semaphore_supported = false;
  bool async_transfer_requested = false;
  bool async_transfer_eligible = false;
  // Capability discovery does not turn on a code path. This becomes true only
  // after transfer/compute submissions actually use timeline synchronization.
  bool async_transfer_enabled = false;
  bool async_transfer_resources_ready = false;
  // Small copies stay on the compute queue because a standalone timeline
  // submission costs more than it can hide for a single small operator.
  VkDeviceSize async_transfer_min_bytes = kDefaultAsyncTransferMinBytes;
  VkPhysicalDeviceMemoryProperties memory_props{};
  uint32_t vendor_id = 0;
  uint32_t device_id = 0;
  uint32_t driver_version = 0;
  uint32_t max_shared_memory = 0;
  uint32_t max_workgroup_invocations = 0;
  uint32_t max_workgroup_size_x = 0;
  uint32_t max_workgroup_size_y = 0;
  uint32_t max_workgroup_size_z = 0;
  uint32_t timestamp_valid_bits = 0;
  float timestamp_period_ns = 1.0f;
  char device_name[VK_MAX_PHYSICAL_DEVICE_NAME_SIZE]{};
  MemoryTopology memory_topology = MemoryTopology::UNKNOWN;
  bool has_device_local_only = false;
  bool has_host_visible_device_local = false;
  bool has_host_visible_system = false;

  // Heap budgets are reported independently so the policy layer can avoid
  // treating UMA and discrete memory as interchangeable. A heap can appear
  // in both total device-local and host-visible accounting when the driver
  // exposes a HOST_VISIBLE | DEVICE_LOCAL memory type.
  VkDeviceSize device_local_budget = 0;
  VkDeviceSize device_local_usage = 0;
  VkDeviceSize device_local_only_budget = 0;
  VkDeviceSize device_local_only_usage = 0;
  VkDeviceSize host_visible_device_local_budget = 0;
  VkDeviceSize host_visible_device_local_usage = 0;
  VkDeviceSize host_visible_system_budget = 0;
  VkDeviceSize host_visible_system_usage = 0;

  // Opt-in software model used to exercise discrete-GPU policy branches on
  // UMA hardware. These fields never change Vulkan's actual memory
  // properties or the legal memory types selected by VMA.
  bool simulation_enabled = false;
  bool simulation_force_staging_requested = false;
  bool simulation_force_staging = false;
  bool simulation_strict_budget = false;
  uint64_t simulation_transfer_latency_us = 0;
  VkDeviceSize simulated_vram_budget = 0;
  std::atomic<VkDeviceSize> simulated_vram_usage{0};

  struct MemoryBudget {
    VkDeviceSize budget = 0;
    VkDeviceSize usage = 0;
  };

  static Device& get();
  void init();
  void destroy();

  bool async_transfer_for(VkDeviceSize bytes) const {
    return async_transfer_enabled &&
        (async_transfer_min_bytes == 0 || bytes >= async_transfer_min_bytes);
  }

  MemoryBudget query_memory_budget() const {
    MemoryBudget out{};
    if (!allocator || !has_memory_budget) return out;
    std::vector<VmaBudget> heaps(memory_props.memoryHeapCount);
    vmaGetHeapBudgets(allocator, heaps.data());
    for (uint32_t i = 0; i < memory_props.memoryHeapCount; ++i) {
      if (!(memory_props.memoryHeaps[i].flags &
            VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)) {
        continue;
      }
      out.budget += heaps[i].budget;
      out.usage += heaps[i].usage;
    }
    return out;
  }

  void classify_memory_topology() {
    has_device_local_only = false;
    has_host_visible_device_local = false;
    has_host_visible_system = false;
    device_local_budget = 0;
    device_local_usage = 0;
    device_local_only_budget = 0;
    device_local_only_usage = 0;
    host_visible_device_local_budget = 0;
    host_visible_device_local_usage = 0;
    host_visible_system_budget = 0;
    host_visible_system_usage = 0;

    std::vector<VmaBudget> budgets(memory_props.memoryHeapCount);
    if (allocator && has_memory_budget) {
      vmaGetHeapBudgets(allocator, budgets.data());
    }
    for (uint32_t heap = 0; heap < memory_props.memoryHeapCount; ++heap) {
      bool has_local = false;
      bool has_host = false;
      bool has_local_only_type = false;
      bool has_local_host_type = false;
      for (uint32_t type = 0; type < memory_props.memoryTypeCount; ++type) {
        if (memory_props.memoryTypes[type].heapIndex != heap) continue;
        const VkMemoryPropertyFlags flags =
            memory_props.memoryTypes[type].propertyFlags;
        const bool local =
            (flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0;
        const bool host =
            (flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0;
        has_local |= local;
        has_host |= host;
        has_local_only_type |= local && !host;
        has_local_host_type |= local && host;
      }
      const VkDeviceSize budget = (allocator && has_memory_budget)
          ? budgets[heap].budget : memory_props.memoryHeaps[heap].size;
      const VkDeviceSize usage = (allocator && has_memory_budget)
          ? budgets[heap].usage : 0;
      if (has_local) {
        device_local_budget += budget;
        device_local_usage += usage;
      }
      if (has_local_host_type) {
        has_host_visible_device_local = true;
        host_visible_device_local_budget += budget;
        host_visible_device_local_usage += usage;
      }
      if (has_local_only_type) {
        has_device_local_only = true;
        device_local_only_budget += budget;
        device_local_only_usage += usage;
      }
      if (has_host && !has_local) {
        has_host_visible_system = true;
        host_visible_system_budget += budget;
        host_visible_system_usage += usage;
      }
    }

    // A single shared heap can expose both local-only and host-visible memory
    // types even though it is physically UMA. Mixed requires a separate
    // host-visible system heap in addition to BAR/shared device-local memory.
    if (has_device_local_only && has_host_visible_device_local &&
        has_host_visible_system) {
      memory_topology = MemoryTopology::MIXED;
    } else if (has_device_local_only && has_host_visible_system) {
      memory_topology = MemoryTopology::DISCRETE;
    } else if (has_host_visible_device_local) {
      memory_topology = MemoryTopology::UMA;
    } else if (has_device_local_only) {
      memory_topology = MemoryTopology::DISCRETE;
    } else {
      memory_topology = MemoryTopology::UNKNOWN;
    }
  }

  static bool simulation_env_flag(const char* name, bool fallback) {
    const char* value = std::getenv(name);
    if (!value || !value[0]) return fallback;
    return value[0] != '0';
  }

  static uint64_t simulation_env_mb(const char* name, uint64_t fallback) {
    const char* value = std::getenv(name);
    if (!value || !value[0]) return fallback;
    char* end = nullptr;
    const unsigned long long mb = std::strtoull(value, &end, 10);
    if (end == value || *end != '\0' || mb == 0) return fallback;
    constexpr uint64_t kMiB = 1024ull * 1024ull;
    if (mb > UINT64_MAX / kMiB) return fallback;
    return static_cast<uint64_t>(mb) * kMiB;
  }

  void configure_simulation() {
    simulation_enabled = simulation_env_flag("VKSSBO_SIMULATE_DISCRETE", false);
    if (!simulation_enabled) {
      simulated_vram_budget = 0;
      simulation_force_staging_requested = false;
      simulation_force_staging = false;
      simulation_strict_budget = false;
      simulation_transfer_latency_us = 0;
      simulated_vram_usage.store(0, std::memory_order_relaxed);
      return;
    }
    simulated_vram_budget = simulation_env_mb(
        "VKSSBO_SIMULATED_VRAM_MB", 4096ull);
    simulation_strict_budget = simulation_env_flag(
        "VKSSBO_SIMULATION_STRICT_BUDGET", false);
    simulation_force_staging_requested = simulation_env_flag(
        "VKSSBO_SIMULATION_FORCE_STAGING", simulation_enabled);
    simulation_force_staging = simulation_enabled &&
        simulation_force_staging_requested && has_device_local_only;
    simulation_transfer_latency_us = 0;
    if (const char* value = std::getenv("VKSSBO_SIMULATION_TRANSFER_LATENCY_US")) {
      if (value[0]) {
        char* end = nullptr;
        const unsigned long long us = std::strtoull(value, &end, 10);
        if (end != value && *end == '\0') {
          simulation_transfer_latency_us = static_cast<uint64_t>(us);
        }
      }
    }
    simulated_vram_usage.store(0, std::memory_order_relaxed);
  }

  MemoryTopology policy_memory_topology() const {
    return simulation_enabled ? MemoryTopology::DISCRETE : memory_topology;
  }

  VkDeviceSize policy_memory_budget() const {
    return simulation_enabled ? simulated_vram_budget : device_local_budget;
  }

  VkDeviceSize policy_memory_usage() const {
    return simulation_enabled
        ? simulated_vram_usage.load(std::memory_order_relaxed)
        : device_local_usage;
  }

  void simulation_reserve(VkDeviceSize size) {
    if (!simulation_enabled || size == 0) return;
    const VkDeviceSize budget = simulated_vram_budget;
    VkDeviceSize old = simulated_vram_usage.load(std::memory_order_relaxed);
    for (;;) {
      if (simulation_strict_budget &&
          (old > budget || size > budget - old)) {
        throw Error("simulated discrete VRAM budget exceeded");
      }
      if (simulated_vram_usage.compare_exchange_weak(
              old, old + size, std::memory_order_relaxed,
              std::memory_order_relaxed)) {
        return;
      }
    }
  }

  void simulation_release(VkDeviceSize size) {
    if (!simulation_enabled || size == 0) return;
    VkDeviceSize old = simulated_vram_usage.load(std::memory_order_relaxed);
    for (;;) {
      const VkDeviceSize next = old > size ? old - size : 0;
      if (simulated_vram_usage.compare_exchange_weak(
              old, next, std::memory_order_relaxed,
              std::memory_order_relaxed)) {
        return;
      }
    }
  }
};

Device& Device::get() {
  // Heap-allocated and never destroyed: teardown clashes with the stock
  // Vulkan context cleanup at process exit (stack-overrun in its dtor).
  static Device* d = []() {
    Device* p = new Device();
    p->init();
    return p;
  }();
  return *d;
}

#ifdef VKSSBO_USE_RUST_MEMORY_POLICY
inline uint64_t memory_policy_env_cap(const char* name) {
  const char* value = std::getenv(name);
  if (!value || !value[0]) return UINT64_MAX;
  long long mb = atoll(value);
  if (mb < 0) return UINT64_MAX;
  const uint64_t unit = 1024ull * 1024ull;
  const uint64_t amount = static_cast<uint64_t>(mb);
  if (amount > UINT64_MAX / unit) return UINT64_MAX;
  return amount * unit;
}

inline VkssboMemoryPlan memory_policy_plan(
    const Device::MemoryBudget& budget, size_t weight_bytes,
    size_t pool_bytes, size_t resident_bytes, size_t safety_reserve,
    size_t fallback_weight_limit, size_t fallback_pool_limit,
    uint64_t requested_bytes = 0, uint32_t request_kind = 0) {
  VkssboMemorySnapshot snapshot{};
  snapshot.heap_budget = static_cast<uint64_t>(budget.budget);
  snapshot.heap_usage = static_cast<uint64_t>(budget.usage);
  snapshot.weight_cache_bytes = static_cast<uint64_t>(weight_bytes);
  snapshot.transient_pool_bytes = static_cast<uint64_t>(pool_bytes);
  snapshot.resident_bytes = static_cast<uint64_t>(resident_bytes);
  snapshot.safety_reserve = static_cast<uint64_t>(safety_reserve);
  snapshot.fallback_weight_limit = static_cast<uint64_t>(fallback_weight_limit);
  snapshot.fallback_pool_limit = static_cast<uint64_t>(fallback_pool_limit);
  snapshot.weight_cache_cap = memory_policy_env_cap("VKSSBO_WEIGHT_CACHE_MB");
  snapshot.transient_pool_cap = memory_policy_env_cap("VKSSBO_TRANSIENT_POOL_MB");
  snapshot.requested_bytes = requested_bytes;
  snapshot.request_kind = request_kind;
  const Device& device = Device::get();
  snapshot.memory_topology = static_cast<uint32_t>(device.memory_topology);
  snapshot.has_device_local_only = device.has_device_local_only ? 1u : 0u;
  snapshot.has_host_visible_device_local =
      device.has_host_visible_device_local ? 1u : 0u;
  snapshot.has_host_visible_system = device.has_host_visible_system ? 1u : 0u;
  snapshot.device_local_budget = static_cast<uint64_t>(device.device_local_budget);
  snapshot.device_local_usage = static_cast<uint64_t>(device.device_local_usage);
  snapshot.device_local_only_budget = static_cast<uint64_t>(device.device_local_only_budget);
  snapshot.device_local_only_usage = static_cast<uint64_t>(device.device_local_only_usage);
  snapshot.host_visible_device_local_budget =
      static_cast<uint64_t>(device.host_visible_device_local_budget);
  snapshot.host_visible_device_local_usage =
      static_cast<uint64_t>(device.host_visible_device_local_usage);
  snapshot.host_visible_system_budget = static_cast<uint64_t>(device.host_visible_system_budget);
  snapshot.host_visible_system_usage = static_cast<uint64_t>(device.host_visible_system_usage);
  if (device.simulation_enabled) {
    // Keep the physical topology untouched, but present a bounded discrete
    // policy view to the Rust planner for cache/eviction decisions.
    snapshot.heap_budget = static_cast<uint64_t>(device.policy_memory_budget());
    snapshot.heap_usage = static_cast<uint64_t>(device.policy_memory_usage());
    snapshot.memory_topology = static_cast<uint32_t>(
        Device::MemoryTopology::DISCRETE);
    snapshot.has_device_local_only = 1u;
    snapshot.has_host_visible_device_local = 0u;
    snapshot.has_host_visible_system = 1u;
    snapshot.device_local_budget = static_cast<uint64_t>(device.simulated_vram_budget);
    snapshot.device_local_usage = static_cast<uint64_t>(device.policy_memory_usage());
    snapshot.device_local_only_budget = static_cast<uint64_t>(device.simulated_vram_budget);
    snapshot.device_local_only_usage = static_cast<uint64_t>(device.policy_memory_usage());
    if (snapshot.host_visible_system_budget == 0) {
      snapshot.host_visible_system_budget = snapshot.heap_budget;
    }
  }
  VkssboMemoryPlan plan{};
  if (!vkssbo_memory_plan(&snapshot, &plan)) {
    throw Error("Rust memory policy returned an invalid plan");
  }
  return plan;
}
#endif

void Device::init() {
  VkApplicationInfo app{};
  app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  app.pApplicationName = "vulkan_ssbo";
  app.apiVersion = VK_API_VERSION_1_0;

  VkInstanceCreateInfo ici{};
  ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  ici.pApplicationInfo = &app;
  const char* layers[] = {"VK_LAYER_KHRONOS_validation"};
  if (getenv("VKSSBO_VALIDATE")) {
    ici.enabledLayerCount = 1;
    ici.ppEnabledLayerNames = layers;
  }
  VKSSBO_CHECK(vkCreateInstance(&ici, nullptr, &instance));

  uint32_t nphys = 0;
  VKSSBO_CHECK(vkEnumeratePhysicalDevices(instance, &nphys, nullptr));
  if (nphys == 0) throw Error("no vulkan physical device");
  std::vector<VkPhysicalDevice> phys_list(nphys);
  VKSSBO_CHECK(vkEnumeratePhysicalDevices(instance, &nphys, phys_list.data()));
  physical_device_count = nphys;
  selected_device_index = 0;
  device_selection = "first";
  const char* requested_name = std::getenv("VKSSBO_DEVICE_NAME");
  const char* requested_index = std::getenv("VKSSBO_DEVICE_INDEX");
  if (requested_name && requested_name[0] != '\0') {
    std::string wanted;
    for (const unsigned char ch : std::string(requested_name)) {
      wanted.push_back(static_cast<char>(std::tolower(ch)));
    }
    for (uint32_t i = 0; i < nphys; ++i) {
      VkPhysicalDeviceProperties candidate_props{};
      vkGetPhysicalDeviceProperties(phys_list[i], &candidate_props);
      std::string candidate;
      for (const unsigned char ch : std::string(candidate_props.deviceName)) {
        candidate.push_back(static_cast<char>(std::tolower(ch)));
      }
      if (candidate.find(wanted) != std::string::npos) {
        selected_device_index = i;
        device_selection = "name";
        break;
      }
    }
    if (device_selection != "name") device_selection = "name_fallback_first";
  } else if (requested_index && requested_index[0] != '\0') {
    char* end = nullptr;
    const unsigned long parsed = std::strtoul(requested_index, &end, 10);
    if (end != requested_index && *end == '\0' && parsed < nphys) {
      selected_device_index = static_cast<uint32_t>(parsed);
      device_selection = "index";
    } else {
      device_selection = "index_fallback_first";
    }
  }
  phys = phys_list[selected_device_index];
  vkGetPhysicalDeviceMemoryProperties(phys, &memory_props);

  VkPhysicalDeviceProperties props{};
  vkGetPhysicalDeviceProperties(phys, &props);
  vendor_id = props.vendorID;
  device_id = props.deviceID;
  driver_version = props.driverVersion;
  max_shared_memory = props.limits.maxComputeSharedMemorySize;
  max_workgroup_invocations = props.limits.maxComputeWorkGroupInvocations;
  max_workgroup_size_x = props.limits.maxComputeWorkGroupSize[0];
  max_workgroup_size_y = props.limits.maxComputeWorkGroupSize[1];
  max_workgroup_size_z = props.limits.maxComputeWorkGroupSize[2];
  timestamp_period_ns = props.limits.timestampPeriod;
  std::strncpy(device_name, props.deviceName, sizeof(device_name) - 1);
  has_fp16 = true;
  has_int64 = false;
  fprintf(stderr, "[vkssbo] maxWGCount=%u,%u,%u maxWGSize=%u,%u,%u\n",
          props.limits.maxComputeWorkGroupCount[0],
          props.limits.maxComputeWorkGroupCount[1],
          props.limits.maxComputeWorkGroupCount[2],
          props.limits.maxComputeWorkGroupSize[0],
          props.limits.maxComputeWorkGroupSize[1],
          props.limits.maxComputeWorkGroupSize[2]);

  // Detect shaderFloat16 and timeline semaphore support. Querying a feature is
  // intentionally separate from enabling it on the logical device.
  has_shader_f16 = false;
  has_shader_int8 = false;
  has_storage_buffer_8bit = false;
  has_uniform_and_storage_buffer_8bit = false;
  has_storage_push_constant_8bit = false;
  has_memory_budget = false;
  timeline_semaphore_supported = false;
  bool has_timeline_ext = false;
  bool has_core_timeline = false;
  {
    uint32_t n_ext = 0;
    vkEnumerateDeviceExtensionProperties(phys, nullptr, &n_ext, nullptr);
    std::vector<VkExtensionProperties> exts(n_ext);
    vkEnumerateDeviceExtensionProperties(phys, nullptr, &n_ext, exts.data());
    bool has_shader_f16_int8_ext = false;
    bool has_8bit_storage_ext = false;
    for (auto& e : exts) {
      if (strcmp(e.extensionName, VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME) == 0) {
        has_shader_f16_int8_ext = true;
      }
#if defined(VK_KHR_8bit_storage)
      if (strcmp(e.extensionName, VK_KHR_8BIT_STORAGE_EXTENSION_NAME) == 0) {
        has_8bit_storage_ext = true;
      }
#endif
#ifdef VK_EXT_MEMORY_BUDGET_EXTENSION_NAME
      if (strcmp(e.extensionName, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME) == 0) {
        has_memory_budget = true;
      }
#endif
#if defined(VK_KHR_timeline_semaphore) || defined(VK_VERSION_1_2)
      if (strcmp(e.extensionName, VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME) == 0) {
        has_timeline_ext = true;
      }
#endif
    }
    const bool has_core_shader_f16_int8 =
        VK_VERSION_MAJOR(props.apiVersion) > 1 ||
        (VK_VERSION_MAJOR(props.apiVersion) == 1 &&
         VK_VERSION_MINOR(props.apiVersion) >= 2);
    if (has_shader_f16_int8_ext || has_core_shader_f16_int8) {
      VkPhysicalDeviceShaderFloat16Int8Features f16{};
      f16.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES;
      VkPhysicalDeviceFeatures2 f2{};
      f2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
      f2.pNext = &f16;
      vkGetPhysicalDeviceFeatures2(phys, &f2);
      has_shader_f16 = f16.shaderFloat16 == VK_TRUE;
      has_shader_int8 = f16.shaderInt8 == VK_TRUE;
    }
#if defined(VK_KHR_8bit_storage) || defined(VK_VERSION_1_2)
    const bool has_core_8bit_storage =
        VK_VERSION_MAJOR(props.apiVersion) > 1 ||
        (VK_VERSION_MAJOR(props.apiVersion) == 1 &&
         VK_VERSION_MINOR(props.apiVersion) >= 2);
    if (has_8bit_storage_ext || has_core_8bit_storage) {
      VkPhysicalDevice8BitStorageFeatures storage8{};
      storage8.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES;
      VkPhysicalDeviceFeatures2 f2{};
      f2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
      f2.pNext = &storage8;
      vkGetPhysicalDeviceFeatures2(phys, &f2);
      has_storage_buffer_8bit = storage8.storageBuffer8BitAccess == VK_TRUE;
      has_uniform_and_storage_buffer_8bit =
          storage8.uniformAndStorageBuffer8BitAccess == VK_TRUE;
      has_storage_push_constant_8bit =
          storage8.storagePushConstant8 == VK_TRUE;
    }
#endif
#if defined(VK_KHR_timeline_semaphore) || defined(VK_VERSION_1_2)
    has_core_timeline =
        VK_VERSION_MAJOR(props.apiVersion) > 1 ||
        (VK_VERSION_MAJOR(props.apiVersion) == 1 &&
         VK_VERSION_MINOR(props.apiVersion) >= 2);
    if (has_timeline_ext || has_core_timeline) {
      VkPhysicalDeviceTimelineSemaphoreFeatures timeline{};
      timeline.sType =
          VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES;
      VkPhysicalDeviceFeatures2 f2{};
      f2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
      f2.pNext = &timeline;
      vkGetPhysicalDeviceFeatures2(phys, &f2);
      timeline_semaphore_supported = timeline.timelineSemaphore == VK_TRUE;
    }
#endif
  }
  fprintf(stderr,
          "[vkssbo] shaderFloat16=%d shaderInt8=%d memoryBudgetExt=%d timelineSemaphore=%d\n",
          (int)has_shader_f16, (int)has_shader_int8, (int)has_memory_budget,
          (int)timeline_semaphore_supported);
  fprintf(stderr, "[vkssbo] device=%s vendor=0x%04x shared=%u workgroupInvocations=%u\n",
          device_name, vendor_id, max_shared_memory,
          max_workgroup_invocations);

  uint32_t nqf = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(phys, &nqf, nullptr);
  std::vector<VkQueueFamilyProperties> qfp(nqf);
  vkGetPhysicalDeviceQueueFamilyProperties(phys, &nqf, qfp.data());
  int chosen = -1;
  for (uint32_t i = 0; i < nqf; i++) {
    if (qfp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
      chosen = (int)i;
      break;
    }
  }
  if (chosen < 0) throw Error("no compute queue");
  queue_family = (uint32_t)chosen;
  compute_queue_supported = true;
  compute_queue_flags = qfp[queue_family].queueFlags;
  timestamp_valid_bits = qfp[queue_family].timestampValidBits;

  int dedicated_transfer = -1;
  int separate_transfer = -1;
  int any_transfer = -1;
  for (uint32_t i = 0; i < nqf; ++i) {
    const VkQueueFlags flags = qfp[i].queueFlags;
    const bool supports_transfer =
        (flags & (VK_QUEUE_TRANSFER_BIT | VK_QUEUE_COMPUTE_BIT |
                  VK_QUEUE_GRAPHICS_BIT)) != 0;
    if (!supports_transfer) continue;
    if (any_transfer < 0) any_transfer = static_cast<int>(i);
    if (i != queue_family && separate_transfer < 0) {
      separate_transfer = static_cast<int>(i);
    }
    if ((flags & VK_QUEUE_TRANSFER_BIT) != 0 &&
        (flags & (VK_QUEUE_COMPUTE_BIT | VK_QUEUE_GRAPHICS_BIT)) == 0) {
      dedicated_transfer = static_cast<int>(i);
      break;
    }
  }
  const int transfer_choice = dedicated_transfer >= 0
      ? dedicated_transfer
      : (separate_transfer >= 0 ? separate_transfer : any_transfer);
  if (transfer_choice >= 0) {
    transfer_queue_supported = true;
    transfer_queue_family = static_cast<uint32_t>(transfer_choice);
    transfer_queue_flags = qfp[transfer_queue_family].queueFlags;
  }
  dedicated_transfer_queue_available = dedicated_transfer >= 0;

  // Topology depends only on physical-device memory properties, so it can be
  // classified before logical-device creation. Budgets are refreshed after
  // VMA is initialized below.
  classify_memory_topology();
  configure_simulation();
  const char* async_env = std::getenv("VKSSBO_ASYNC_TRANSFER");
  async_transfer_requested =
      async_env != nullptr && async_env[0] != '\0' && strcmp(async_env, "0") != 0;
  async_transfer_min_bytes = kDefaultAsyncTransferMinBytes;
  const char* async_min_env = std::getenv("VKSSBO_ASYNC_TRANSFER_MIN_BYTES");
  if (async_min_env != nullptr && async_min_env[0] != '\0') {
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(async_min_env, &end, 10);
    if (end != async_min_env && end != nullptr && *end == '\0') {
      async_transfer_min_bytes = static_cast<VkDeviceSize>(parsed);
    }
  }
  async_transfer_eligible =
      (memory_topology == MemoryTopology::DISCRETE ||
       memory_topology == MemoryTopology::MIXED ||
       (simulation_enabled && async_transfer_requested)) &&
      dedicated_transfer_queue_available && timeline_semaphore_supported;
  bool async_candidate = async_transfer_requested && async_transfer_eligible;

  const float prio = 1.0f;
  VkDeviceQueueCreateInfo queue_infos[2]{};
  queue_infos[0].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  queue_infos[0].queueFamilyIndex = queue_family;
  queue_infos[0].queueCount = 1;
  queue_infos[0].pQueuePriorities = &prio;
  uint32_t queue_info_count = 1;
  if (async_candidate) {
    queue_infos[1].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_infos[1].queueFamilyIndex = transfer_queue_family;
    queue_infos[1].queueCount = 1;
    queue_infos[1].pQueuePriorities = &prio;
    queue_info_count = 2;
  }

  VkPhysicalDeviceFeatures feats{};

  std::vector<const char*> dev_exts;
  if (has_shader_f16) {
    dev_exts.push_back(VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME);
  }
#ifdef VK_EXT_MEMORY_BUDGET_EXTENSION_NAME
  if (has_memory_budget) {
    dev_exts.push_back(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
  }
#endif
  const bool enable_timeline_extension =
      async_candidate && has_timeline_ext && !has_core_timeline;
  if (enable_timeline_extension) {
    dev_exts.push_back(VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME);
  }
  VkPhysicalDeviceShaderFloat16Int8Features f16_feat{};
  f16_feat.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES;
  f16_feat.shaderFloat16 = VK_TRUE;
  VkPhysicalDeviceTimelineSemaphoreFeatures timeline_feat{};
  timeline_feat.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES;
  timeline_feat.timelineSemaphore = async_candidate ? VK_TRUE : VK_FALSE;
  if (async_candidate) {
    if (has_shader_f16) {
      f16_feat.pNext = &timeline_feat;
    }
  }

  VkDeviceCreateInfo dci{};
  dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  dci.queueCreateInfoCount = queue_info_count;
  dci.pQueueCreateInfos = queue_infos;
  dci.pEnabledFeatures = &feats;
  if (!dev_exts.empty()) {
    dci.enabledExtensionCount = (uint32_t)dev_exts.size();
    dci.ppEnabledExtensionNames = dev_exts.data();
  }
  if (has_shader_f16) {
    dci.pNext = &f16_feat;
  } else if (async_candidate) {
    dci.pNext = &timeline_feat;
  }
  VkResult device_result = vkCreateDevice(phys, &dci, nullptr, &device);
  if (device_result != VK_SUCCESS && async_candidate) {
    // Capability advertisement is not a guarantee that logical-device
    // creation with a second queue and timeline feature will succeed. Retry
    // the proven synchronous configuration instead of disabling the backend.
    fprintf(stderr,
            "[vkssbo] async device creation failed (%d); retrying sync path\n",
            static_cast<int>(device_result));
    async_candidate = false;
    queue_info_count = 1;
    dci.queueCreateInfoCount = 1;
    if (has_shader_f16) {
      f16_feat.pNext = nullptr;
      dci.pNext = &f16_feat;
    } else {
      dci.pNext = nullptr;
    }
    if (enable_timeline_extension && !dev_exts.empty()) {
      dev_exts.pop_back();
      dci.enabledExtensionCount = static_cast<uint32_t>(dev_exts.size());
      dci.ppEnabledExtensionNames =
          dev_exts.empty() ? nullptr : dev_exts.data();
    }
    device_result = vkCreateDevice(phys, &dci, nullptr, &device);
  }
  VKSSBO_CHECK(device_result);
  vkGetDeviceQueue(device, queue_family, 0, &queue);
  if (async_candidate) {
    vkGetDeviceQueue(device, transfer_queue_family, 0, &transfer_queue);
  }

  async_transfer_resources_ready = false;
  async_transfer_enabled = false;
  if (async_candidate && transfer_queue != VK_NULL_HANDLE) {
    VkCommandPoolCreateInfo transfer_cpci{};
    transfer_cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    transfer_cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    transfer_cpci.queueFamilyIndex = transfer_queue_family;
    const VkResult pool_result =
        vkCreateCommandPool(device, &transfer_cpci, nullptr, &transfer_cmd_pool);
    if (pool_result == VK_SUCCESS) {
      VkSemaphoreTypeCreateInfo timeline_type{};
      timeline_type.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
      timeline_type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
      timeline_type.initialValue = 0;
      VkSemaphoreCreateInfo sem_ci{};
      sem_ci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
      sem_ci.pNext = &timeline_type;
      const VkResult sem_result =
          vkCreateSemaphore(device, &sem_ci, nullptr, &transfer_timeline);
      if (sem_result == VK_SUCCESS) {
        async_transfer_resources_ready = true;
        async_transfer_enabled = true;
      } else {
        vkDestroyCommandPool(device, transfer_cmd_pool, nullptr);
        transfer_cmd_pool = VK_NULL_HANDLE;
      }
    }
  }

  // Timestamp queries are optional in Vulkan.  Keep the autotuner disabled
  // when the selected compute queue does not expose valid timestamp bits.
  if (timestamp_valid_bits != 0) {
    VkQueryPoolCreateInfo qpci{};
    qpci.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    qpci.queryType = VK_QUERY_TYPE_TIMESTAMP;
    qpci.queryCount = Device::kTimestampQueryCount;
    VkResult qpr = vkCreateQueryPool(device, &qpci, nullptr, &timestamp_pool);
    if (qpr != VK_SUCCESS) {
      timestamp_pool = VK_NULL_HANDLE;
      timestamp_valid_bits = 0;
    }
  }

  VmaAllocatorCreateInfo aci{};
  aci.vulkanApiVersion = VK_API_VERSION_1_0;
  aci.physicalDevice = phys;
  aci.device = device;
  aci.instance = instance;
  #ifdef VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT
  if (has_memory_budget) {
    aci.flags |= VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT;
  }
  #endif
  VKSSBO_CHECK(vmaCreateAllocator(&aci, &allocator));
  classify_memory_topology();
  // Refresh topology budgets after VMA is available; eligibility itself was
  // computed before logical-device creation so queue creation could be gated.
  async_transfer_eligible =
      (memory_topology == MemoryTopology::DISCRETE ||
       memory_topology == MemoryTopology::MIXED ||
       (simulation_enabled && async_transfer_requested)) &&
      dedicated_transfer_queue_available && timeline_semaphore_supported;
  fprintf(stderr,
          "[vkssbo] memory_topology=%u local_only=%d local_host=%d "
          "host_system=%d local_budget=%llu local_only_budget=%llu "
          "local_host_budget=%llu host_system_budget=%llu\n",
          static_cast<unsigned>(memory_topology),
          has_device_local_only ? 1 : 0,
          has_host_visible_device_local ? 1 : 0,
          has_host_visible_system ? 1 : 0,
          static_cast<unsigned long long>(device_local_budget),
          static_cast<unsigned long long>(device_local_only_budget),
          static_cast<unsigned long long>(host_visible_device_local_budget),
          static_cast<unsigned long long>(host_visible_system_budget));
  fprintf(stderr,
          "[vkssbo] computeQ=%u flags=0x%x transferQ=%d flags=0x%x "
          "dedicatedTransfer=%d asyncRequested=%d asyncEligible=%d "
          "asyncEnabled=%d resourcesReady=%d\n",
          queue_family, static_cast<unsigned>(compute_queue_flags),
          transfer_queue_supported ? static_cast<int>(transfer_queue_family) : -1,
          static_cast<unsigned>(transfer_queue_flags),
          dedicated_transfer_queue_available ? 1 : 0,
          async_transfer_requested ? 1 : 0,
          async_transfer_eligible ? 1 : 0,
          async_transfer_enabled ? 1 : 0,
          async_transfer_resources_ready ? 1 : 0);
  if (simulation_enabled) {
    fprintf(stderr,
            "[vkssbo] simulation=1 policy_topology=discrete "
            "virtual_vram_budget=%llu force_staging=%d strict_budget=%d "
            "transfer_latency_us=%llu\n",
            static_cast<unsigned long long>(simulated_vram_budget),
            simulation_force_staging ? 1 : 0,
            simulation_strict_budget ? 1 : 0,
            static_cast<unsigned long long>(simulation_transfer_latency_us));
  }

  // Descriptor pool: one uniform (params) + up to 8 SSBOs per dispatch.
  VkDescriptorPoolSize pool_sizes[] = {
      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 8192},
      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1024},
  };
  VkDescriptorPoolCreateInfo dpci{};
  dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  dpci.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
  dpci.poolSizeCount = 2;
  dpci.pPoolSizes = pool_sizes;
  dpci.maxSets = 4096;
  VKSSBO_CHECK(vkCreateDescriptorPool(device, &dpci, nullptr, &desc_pool));

  VkCommandPoolCreateInfo cpci{};
  cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  cpci.queueFamilyIndex = queue_family;
  VKSSBO_CHECK(vkCreateCommandPool(device, &cpci, nullptr, &cmd_pool));

  // Global descriptor set layout: up to 8 storage buffers, bindings 0..7.
  // All shaders use this layout; pipelines bind it at creation.
  {
    VkDescriptorSetLayoutBinding gb[8];
    for (int i = 0; i < 8; i++) {
      gb[i].binding = (uint32_t)i;
      gb[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      gb[i].descriptorCount = 1;
      gb[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
      gb[i].pImmutableSamplers = nullptr;
    }
    VkDescriptorSetLayoutCreateInfo dslci{};
    dslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dslci.bindingCount = 8;
    dslci.pBindings = gb;
    VKSSBO_CHECK(vkCreateDescriptorSetLayout(device, &dslci, nullptr, &global_ds_layout));
  }

  this->phys = phys;
}

void Device::destroy() {
  if (device) {
    if (timestamp_pool) {
      vkDestroyQueryPool(device, timestamp_pool, nullptr);
      timestamp_pool = VK_NULL_HANDLE;
    }
    if (transfer_timeline) {
      vkDestroySemaphore(device, transfer_timeline, nullptr);
      transfer_timeline = VK_NULL_HANDLE;
    }
    if (transfer_cmd_pool) {
      vkDestroyCommandPool(device, transfer_cmd_pool, nullptr);
      transfer_cmd_pool = VK_NULL_HANDLE;
    }
    vkDestroyCommandPool(device, cmd_pool, nullptr);
    vkDestroyDescriptorPool(device, desc_pool, nullptr);
    vmaDestroyAllocator(allocator);
    vkDestroyDevice(device, nullptr);
    device = VK_NULL_HANDLE;
  }
  if (instance) {
    vkDestroyInstance(instance, nullptr);
    instance = VK_NULL_HANDLE;
  }
}

// ---------- SSBO buffer ----------
struct Buffer;
void recycle_transient_buffer(Buffer* b);
void recycle_staging_buffer(Buffer* b);

struct Buffer {
  VmaAllocation allocation = VK_NULL_HANDLE;
  VkBuffer buffer = VK_NULL_HANDLE;
  VkDeviceSize size = 0;
  bool mapped = false;
  bool persistent_mapped = false;
  bool recyclable = false;
  bool staging_recyclable = false;
  bool host_visible = false;
  bool host_coherent = false;
  bool simulation_accounted = false;
  void* map_ptr = nullptr;

  Buffer() = default;
  Buffer(Buffer&& o) noexcept
      : allocation(o.allocation), buffer(o.buffer), size(o.size),
        mapped(o.mapped), persistent_mapped(o.persistent_mapped),
        recyclable(o.recyclable), staging_recyclable(o.staging_recyclable),
        host_visible(o.host_visible),
        host_coherent(o.host_coherent),
        simulation_accounted(o.simulation_accounted), map_ptr(o.map_ptr) {
    o.allocation = VK_NULL_HANDLE;
    o.buffer = VK_NULL_HANDLE;
    o.size = 0;
    o.mapped = false;
    o.persistent_mapped = false;
    o.recyclable = false;
    o.staging_recyclable = false;
    o.host_visible = false;
    o.host_coherent = false;
    o.simulation_accounted = false;
    o.map_ptr = nullptr;
  }
  Buffer& operator=(Buffer&& o) noexcept {
    if (this != &o) {
      destroy();
      allocation = o.allocation;
      buffer = o.buffer;
      size = o.size;
      mapped = o.mapped;
      persistent_mapped = o.persistent_mapped;
      recyclable = o.recyclable;
      staging_recyclable = o.staging_recyclable;
      host_visible = o.host_visible;
      host_coherent = o.host_coherent;
      simulation_accounted = o.simulation_accounted;
      map_ptr = o.map_ptr;
      o.allocation = VK_NULL_HANDLE;
      o.buffer = VK_NULL_HANDLE;
      o.size = 0;
      o.mapped = false;
      o.persistent_mapped = false;
      o.recyclable = false;
      o.staging_recyclable = false;
      o.host_visible = false;
      o.host_coherent = false;
      o.simulation_accounted = false;
      o.map_ptr = nullptr;
    }
    return *this;
  }
  ~Buffer() { destroy(); }
  void destroy() {
    auto& d = Device::get();
    if (buffer) {
      if (mapped && !persistent_mapped) {
        vmaUnmapMemory(d.allocator, allocation);
      }
      if (staging_recyclable) {
        recycle_staging_buffer(this);
        return;
      }
      if (recyclable) {
        recycle_transient_buffer(this);
        return;
      }
      vmaDestroyBuffer(d.allocator, buffer, allocation);
      if (simulation_accounted) {
        d.simulation_release(size);
      }
      buffer = VK_NULL_HANDLE;
      allocation = VK_NULL_HANDLE;
      size = 0;
      mapped = false;
      persistent_mapped = false;
    recyclable = false;
    staging_recyclable = false;
      host_visible = false;
      host_coherent = false;
      simulation_accounted = false;
      map_ptr = nullptr;
    }
  }
};

Buffer create_buffer(VkDeviceSize size, VkBufferUsageFlags usage, VmaAllocationCreateFlags flags, VmaMemoryUsage mem) {
  Buffer b;
  VkBufferCreateInfo bci{};
  bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bci.size = size;
  bci.usage = usage;
  auto& device = Device::get();
  uint32_t sharing_families[2] = {
      device.queue_family, device.transfer_queue_family};
  if (device.async_transfer_enabled &&
      device.transfer_queue_family != UINT32_MAX &&
      device.transfer_queue_family != device.queue_family) {
    bci.sharingMode = VK_SHARING_MODE_CONCURRENT;
    bci.queueFamilyIndexCount = 2;
    bci.pQueueFamilyIndices = sharing_families;
  }
  VmaAllocationCreateInfo aci{};
  aci.usage = mem;
  aci.flags = flags;
  VmaAllocationInfo allocation_info{};
  const bool simulation_accounted = device.simulation_enabled &&
      mem == VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
  if (simulation_accounted) {
    device.simulation_reserve(size);
  }
  const VkResult create_result = vmaCreateBuffer(
      device.allocator, &bci, &aci, &b.buffer, &b.allocation, &allocation_info);
  if (create_result != VK_SUCCESS) {
    if (simulation_accounted) device.simulation_release(size);
    VKSSBO_CHECK(create_result);
  }
  b.size = size;
  b.simulation_accounted = simulation_accounted;
  if (allocation_info.pMappedData) {
    b.mapped = true;
    b.persistent_mapped = true;
    b.map_ptr = allocation_info.pMappedData;
  }
  VkMemoryPropertyFlags memory_properties = 0;
  vmaGetAllocationMemoryProperties(
      Device::get().allocator, b.allocation, &memory_properties);
  b.host_visible = (memory_properties & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0;
  b.host_coherent = (memory_properties & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
  return b;
}

// Temporary activation buffers are safe to recycle after submit_and_wait().
// The pool is bounded by the live Vulkan budget and never owns weights.
struct PooledBuffer {
  VkBuffer buffer = VK_NULL_HANDLE;
  VmaAllocation allocation = VK_NULL_HANDLE;
  VkDeviceSize size = 0;
  bool persistent_mapped = false;
  bool host_visible = false;
  bool host_coherent = false;
  bool simulation_accounted = false;
  void* map_ptr = nullptr;
  uint64_t last = 0;
};

struct TransientBufferPool {
  std::mutex mutex;
  std::vector<PooledBuffer> free;
  size_t bytes = 0;
  uint64_t clock = 0;
  uint64_t hits = 0;
  uint64_t misses = 0;
  static constexpr size_t kFallbackLimit =
      2ull * 1024ull * 1024ull * 1024ull;
  static constexpr size_t kSafetyReserve =
      256ull * 1024ull * 1024ull;
  size_t cached_limit = kFallbackLimit;
  std::chrono::steady_clock::time_point last_budget_query{};

  static size_t min_size() {
    static const size_t value = [] {
      const char* env = std::getenv("VKSSBO_TRANSIENT_POOL_MIN_KB");
      if (!env || !env[0]) return 64ull * 1024ull;
      long long kb = atoll(env);
      return kb > 0 ? (size_t)kb * 1024ull : 0ull;
    }();
    return value;
  }

  static bool enabled() {
    static const bool value = [] {
      const char* env = std::getenv("VKSSBO_TRANSIENT_POOL");
      return !env || env[0] != '0';
    }();
    return value;
  }

  bool eligible(VkDeviceSize requested) const {
    return requested >= min_size();
  }

  size_t limit_locked(VkDeviceSize incoming = 0) {
    const auto now = std::chrono::steady_clock::now();
    if (incoming == 0 && last_budget_query.time_since_epoch().count() != 0 &&
        now - last_budget_query < std::chrono::milliseconds(250)) {
      return cached_limit;
    }
    auto budget = Device::get().query_memory_budget();
#ifdef VKSSBO_USE_RUST_MEMORY_POLICY
    VkssboMemoryPlan plan = memory_policy_plan(
        budget, 0, bytes, 0, kSafetyReserve,
        12ull * 1024ull * 1024ull * 1024ull, kFallbackLimit,
        incoming, 2);
    // The candidate buffer is already allocated when release() asks for a
    // limit, so the live Vulkan budget includes `incoming`.  The Rust policy
    // therefore returns the maximum total pool size directly; expanding it by
    // incoming here would let the pool exceed the safe budget.
    const uint64_t limit = plan.transient_pool_limit;
    cached_limit = static_cast<size_t>(plan.transient_pool_limit);
    last_budget_query = now;
    return static_cast<size_t>(limit);
#else
    size_t limit = kFallbackLimit;
    size_t base_limit = kFallbackLimit;
    if (budget.budget > 0) {
      VkDeviceSize safe_budget = (budget.budget / 5u) * 4u;
      safe_budget = safe_budget > kSafetyReserve
          ? safe_budget - kSafetyReserve : 0;
      VkDeviceSize non_pool = budget.usage > bytes
          ? budget.usage - (VkDeviceSize)bytes : 0;
      size_t available = safe_budget > non_pool
          ? (size_t)(safe_budget - non_pool) : 0;
      base_limit = available;
      if (incoming != 0 && available <=
          std::numeric_limits<size_t>::max() - (size_t)incoming) {
        available += (size_t)incoming;
      }
      limit = available;
    }
    const char* env = std::getenv("VKSSBO_TRANSIENT_POOL_MB");
    if (env && env[0]) {
      long long mb = atoll(env);
      if (mb >= 0) {
        size_t configured = (size_t)mb * 1024ull * 1024ull;
        if (configured < limit) limit = configured;
        if (configured < base_limit) base_limit = configured;
      }
    }
    cached_limit = base_limit;
    last_budget_query = now;
    return limit;
#endif
  }

  void destroy_entry_locked(PooledBuffer entry) {
    if (entry.buffer) {
      vmaDestroyBuffer(Device::get().allocator, entry.buffer, entry.allocation);
      if (entry.simulation_accounted) {
        Device::get().simulation_release(entry.size);
      }
      bytes -= (size_t)entry.size;
    }
  }

  bool acquire(VkDeviceSize requested, Buffer* out) {
    if (!enabled() || !eligible(requested)) return false;
    std::lock_guard<std::mutex> lock(mutex);
    size_t best = (size_t)-1;
    VkDeviceSize best_size = (VkDeviceSize)-1;
    for (size_t i = 0; i < free.size(); ++i) {
      if (free[i].size >= requested && free[i].size < best_size) {
        best = i;
        best_size = free[i].size;
      }
    }
    if (best == (size_t)-1) {
      ++misses;
      return false;
    }
    PooledBuffer entry = free[best];
    free[best] = free.back();
    free.pop_back();
    bytes -= (size_t)entry.size;
    out->buffer = entry.buffer;
    out->allocation = entry.allocation;
    out->size = entry.size;
    out->mapped = false;
    out->persistent_mapped = entry.persistent_mapped;
    out->host_visible = entry.host_visible;
    out->host_coherent = entry.host_coherent;
    out->simulation_accounted = entry.simulation_accounted;
    out->recyclable = true;
    out->mapped = entry.persistent_mapped;
    out->map_ptr = entry.map_ptr;
    ++hits;
    return true;
  }

  void release(Buffer* b) {
    std::lock_guard<std::mutex> lock(mutex);
    const size_t incoming = (size_t)b->size;
    const size_t limit = limit_locked((VkDeviceSize)incoming);
    while ((incoming > limit || bytes > limit - incoming) && !free.empty()) {
      size_t oldest = 0;
      for (size_t i = 1; i < free.size(); ++i) {
        if (free[i].last < free[oldest].last) oldest = i;
      }
      PooledBuffer entry = free[oldest];
      free[oldest] = free.back();
      free.pop_back();
      destroy_entry_locked(entry);
    }
    if (incoming > limit || bytes > limit - incoming) {
      vmaDestroyBuffer(Device::get().allocator, b->buffer, b->allocation);
      if (b->simulation_accounted) {
        Device::get().simulation_release(b->size);
      }
    } else {
      free.push_back(PooledBuffer{b->buffer, b->allocation, b->size,
                                  b->persistent_mapped, b->host_visible,
                                  b->host_coherent, b->simulation_accounted,
                                  b->map_ptr, ++clock});
      bytes += incoming;
    }
    b->buffer = VK_NULL_HANDLE;
    b->allocation = VK_NULL_HANDLE;
    b->size = 0;
    b->recyclable = false;
    b->mapped = false;
    b->persistent_mapped = false;
    b->host_visible = false;
    b->host_coherent = false;
    b->simulation_accounted = false;
    b->map_ptr = nullptr;
  }

  void stats(size_t* pooled_bytes, size_t* entries,
             uint64_t* pool_hits, uint64_t* pool_misses) {
    std::lock_guard<std::mutex> lock(mutex);
    if (pooled_bytes) *pooled_bytes = bytes;
    if (entries) *entries = free.size();
    if (pool_hits) *pool_hits = hits;
    if (pool_misses) *pool_misses = misses;
  }

  void trim() {
    std::lock_guard<std::mutex> lock(mutex);
    for (const PooledBuffer& entry : free) {
      if (entry.buffer) {
        vmaDestroyBuffer(Device::get().allocator, entry.buffer,
                         entry.allocation);
        if (entry.simulation_accounted) {
          Device::get().simulation_release(entry.size);
        }
      }
    }
    free.clear();
    bytes = 0;
    cached_limit = kFallbackLimit;
    last_budget_query = {};
  }

  size_t current_limit() {
    std::lock_guard<std::mutex> lock(mutex);
    return limit_locked();
  }
};

inline TransientBufferPool& transient_pool() {
  static TransientBufferPool pool;
  return pool;
}

inline void recycle_transient_buffer(Buffer* b) {
  transient_pool().release(b);
}

inline bool acquire_transient_buffer(VkDeviceSize size, Buffer* out) {
  return transient_pool().acquire(size, out);
}

inline void transient_pool_stats(size_t* pooled_bytes, size_t* entries,
                                 uint64_t* hits, uint64_t* misses) {
  transient_pool().stats(pooled_bytes, entries, hits, misses);
}

inline size_t transient_pool_limit() {
  return transient_pool().current_limit();
}

inline void transient_pool_trim() {
  transient_pool().trim();
}

// Host-visible staging allocations are kept in a separate bounded pool. They
// have different usage and memory properties from device-local activations,
// so sharing the transient pool would risk binding a host buffer as a compute
// output on the next operator.
struct StagingPooledBuffer {
  VkBuffer buffer = VK_NULL_HANDLE;
  VmaAllocation allocation = VK_NULL_HANDLE;
  VkDeviceSize size = 0;
  bool persistent_mapped = false;
  bool host_visible = false;
  bool host_coherent = false;
  void* map_ptr = nullptr;
  uint64_t last = 0;
};

struct StagingBufferPool {
  std::mutex mutex;
  std::vector<StagingPooledBuffer> free;
  size_t bytes = 0;
  uint64_t clock = 0;
  uint64_t hits = 0;
  uint64_t misses = 0;
  static constexpr size_t kDefaultLimit = 256ull * 1024ull * 1024ull;
  static constexpr size_t kDefaultMinSize = 64ull * 1024ull;
  size_t cached_limit = kDefaultLimit;

  static bool enabled() {
    const char* env = std::getenv("VKSSBO_STAGING_POOL");
    return !env || env[0] != '0';
  }

  static size_t min_size() {
    const char* env = std::getenv("VKSSBO_STAGING_POOL_MIN_KB");
    if (!env || !env[0]) return kDefaultMinSize;
    const long long kb = atoll(env);
    return kb > 0 ? static_cast<size_t>(kb) * 1024ull : 0ull;
  }

  size_t limit() {
    const char* env = std::getenv("VKSSBO_STAGING_POOL_MB");
    if (!env || !env[0]) return kDefaultLimit;
    const long long mb = atoll(env);
    return mb >= 0 ? static_cast<size_t>(mb) * 1024ull * 1024ull
                   : kDefaultLimit;
  }

  void destroy_entry_locked(const StagingPooledBuffer& entry) {
    if (!entry.buffer) return;
    vmaDestroyBuffer(
        Device::get().allocator, entry.buffer, entry.allocation);
    bytes -= static_cast<size_t>(entry.size);
  }

  bool acquire(VkDeviceSize requested, Buffer* out) {
    if (!out || !enabled() || requested < min_size()) return false;
    std::lock_guard<std::mutex> lock(mutex);
    size_t best = (size_t)-1;
    VkDeviceSize best_size = (VkDeviceSize)-1;
    for (size_t i = 0; i < free.size(); ++i) {
      if (free[i].size >= requested && free[i].size < best_size) {
        best = i;
        best_size = free[i].size;
      }
    }
    if (best == (size_t)-1) {
      ++misses;
      if (std::getenv("VKSSBO_TRACE_STAGING_POOL")) {
        fprintf(stderr, "[vkssbo] staging pool acquire miss bytes=%llu\n",
                static_cast<unsigned long long>(requested));
      }
      return false;
    }
    StagingPooledBuffer entry = free[best];
    free[best] = free.back();
    free.pop_back();
    bytes -= static_cast<size_t>(entry.size);
    out->buffer = entry.buffer;
    out->allocation = entry.allocation;
    out->size = entry.size;
    out->mapped = false;
    out->persistent_mapped = entry.persistent_mapped;
    out->recyclable = false;
    out->staging_recyclable = true;
    out->host_visible = entry.host_visible;
    out->host_coherent = entry.host_coherent;
    out->map_ptr = entry.map_ptr;
    ++hits;
    if (std::getenv("VKSSBO_TRACE_STAGING_POOL")) {
      fprintf(stderr,
              "[vkssbo] staging pool acquire hit requested=%llu bytes=%llu\n",
              static_cast<unsigned long long>(requested),
              static_cast<unsigned long long>(entry.size));
    }
    return true;
  }

  void release(Buffer* b) {
    if (!b || !b->buffer) return;
    std::lock_guard<std::mutex> lock(mutex);
    const size_t incoming = static_cast<size_t>(b->size);
    const size_t max_bytes = limit();
    while ((incoming > max_bytes || bytes > max_bytes - incoming) &&
           !free.empty()) {
      size_t oldest = 0;
      for (size_t i = 1; i < free.size(); ++i) {
        if (free[i].last < free[oldest].last) oldest = i;
      }
      StagingPooledBuffer entry = free[oldest];
      free[oldest] = free.back();
      free.pop_back();
      destroy_entry_locked(entry);
    }
    if (incoming <= max_bytes && bytes <= max_bytes - incoming) {
      free.push_back(StagingPooledBuffer{
          b->buffer, b->allocation, b->size, b->persistent_mapped,
          b->host_visible, b->host_coherent, b->map_ptr, ++clock});
      bytes += incoming;
    } else {
      vmaDestroyBuffer(Device::get().allocator, b->buffer, b->allocation);
    }
    if (std::getenv("VKSSBO_TRACE_STAGING_POOL")) {
      fprintf(stderr, "[vkssbo] staging pool release bytes=%llu pooled=%llu\n",
              static_cast<unsigned long long>(incoming),
              static_cast<unsigned long long>(bytes));
    }
    b->buffer = VK_NULL_HANDLE;
    b->allocation = VK_NULL_HANDLE;
    b->size = 0;
    b->mapped = false;
    b->persistent_mapped = false;
    b->recyclable = false;
    b->staging_recyclable = false;
    b->host_visible = false;
    b->host_coherent = false;
    b->map_ptr = nullptr;
  }

  void stats(size_t* pooled_bytes, size_t* entries,
             uint64_t* pool_hits, uint64_t* pool_misses) {
    std::lock_guard<std::mutex> lock(mutex);
    if (pooled_bytes) *pooled_bytes = bytes;
    if (entries) *entries = free.size();
    if (pool_hits) *pool_hits = hits;
    if (pool_misses) *pool_misses = misses;
  }

  void trim() {
    std::lock_guard<std::mutex> lock(mutex);
    for (const StagingPooledBuffer& entry : free) destroy_entry_locked(entry);
    free.clear();
    bytes = 0;
    cached_limit = kDefaultLimit;
  }
};

inline StagingBufferPool& staging_pool() {
  static StagingBufferPool pool;
  return pool;
}

inline void recycle_staging_buffer(Buffer* b) {
  staging_pool().release(b);
}

inline bool acquire_staging_buffer(VkDeviceSize size, Buffer* out) {
  return staging_pool().acquire(size, out);
}

inline void staging_pool_stats(size_t* pooled_bytes, size_t* entries,
                               uint64_t* hits, uint64_t* misses) {
  staging_pool().stats(pooled_bytes, entries, hits, misses);
}

inline void staging_pool_trim() {
  staging_pool().trim();
}

// Persistent weight buffers are separate from the transient activation pool.
// Keys are supplied by the bridge from Tensor storage identity + version, so
// an in-place weight update naturally creates a new cache entry.
struct WeightCacheEntry {
  uint64_t key = 0;
  VkDeviceSize size = 0;
  Buffer buffer;
  uint64_t last = 0;
};

struct WeightBufferCache {
  std::mutex mutex;
  std::list<WeightCacheEntry> entries;
  std::unordered_map<uint64_t, std::list<WeightCacheEntry>::iterator> index;
  size_t bytes = 0;
  uint64_t clock = 0;
  uint64_t hits = 0;
  uint64_t misses = 0;
  static constexpr size_t kFallbackLimit =
      12ull * 1024ull * 1024ull * 1024ull;
  static constexpr size_t kSafetyReserve =
      256ull * 1024ull * 1024ull;
  size_t cached_limit = kFallbackLimit;
  std::chrono::steady_clock::time_point last_budget_query{};

  static bool enabled() {
    static const bool value = [] {
      const char* env = std::getenv("VKSSBO_WEIGHT_CACHE");
      return !env || env[0] != '0';
    }();
    return value;
  }

  size_t limit_locked(VkDeviceSize incoming = 0) {
    const auto now = std::chrono::steady_clock::now();
    if (incoming == 0 && last_budget_query.time_since_epoch().count() != 0 &&
        now - last_budget_query < std::chrono::milliseconds(250)) {
      return cached_limit;
    }
    auto budget = Device::get().query_memory_budget();
#ifdef VKSSBO_USE_RUST_MEMORY_POLICY
    VkssboMemoryPlan plan = memory_policy_plan(
        budget, bytes, 0, 0, kSafetyReserve, kFallbackLimit, 0,
        incoming, 1);
    cached_limit = static_cast<size_t>(plan.weight_cache_limit);
    last_budget_query = now;
    return cached_limit;
#else
    size_t limit = kFallbackLimit;
    if (budget.budget > 0) {
      VkDeviceSize safe_budget = (budget.budget / 5u) * 4u;
      safe_budget = safe_budget > kSafetyReserve
          ? safe_budget - kSafetyReserve : 0;
      VkDeviceSize non_cache = budget.usage > bytes
          ? budget.usage - (VkDeviceSize)bytes : 0;
      limit = safe_budget > non_cache
          ? static_cast<size_t>(safe_budget - non_cache) : 0;
    }
    const char* env = std::getenv("VKSSBO_WEIGHT_CACHE_MB");
    if (env && env[0]) {
      long long mb = atoll(env);
      if (mb >= 0) {
        size_t configured = static_cast<size_t>(mb) * 1024ull * 1024ull;
        if (configured < limit) limit = configured;
      }
    }
    cached_limit = limit;
    last_budget_query = now;
    return limit;
#endif
  }

  void erase_locked(std::list<WeightCacheEntry>::iterator it) {
    bytes -= static_cast<size_t>(it->size);
    index.erase(it->key);
    entries.erase(it);
  }

  Buffer* find(uint64_t key, VkDeviceSize requested) {
    if (!enabled() || key == 0) return nullptr;
    std::lock_guard<std::mutex> lock(mutex);
    auto it = index.find(key);
    if (it == index.end() || it->second->size < requested) {
      ++misses;
      return nullptr;
    }
    it->second->last = ++clock;
    entries.splice(entries.begin(), entries, it->second);
    ++hits;
    return &entries.front().buffer;
  }

  Buffer* insert(uint64_t key, Buffer&& candidate) {
    if (!enabled() || key == 0 || !candidate.buffer) return nullptr;
    std::lock_guard<std::mutex> lock(mutex);
    auto existing = index.find(key);
    if (existing != index.end()) erase_locked(existing->second);
    const size_t incoming = static_cast<size_t>(candidate.size);
    const size_t limit = limit_locked(static_cast<VkDeviceSize>(incoming));
    while ((incoming > limit || bytes > limit - incoming) && !entries.empty()) {
      auto oldest = entries.begin();
      for (auto it = entries.begin(); it != entries.end(); ++it) {
        if (it->last < oldest->last) oldest = it;
      }
      erase_locked(oldest);
    }
    if (incoming > limit || bytes > limit - incoming) return nullptr;
    entries.emplace_front();
    entries.front().key = key;
    entries.front().size = candidate.size;
    entries.front().buffer = std::move(candidate);
    entries.front().last = ++clock;
    index[key] = entries.begin();
    bytes += incoming;
    return &entries.front().buffer;
  }

  void stats(size_t* cached_bytes, size_t* entry_count,
             uint64_t* cache_hits, uint64_t* cache_misses) {
    std::lock_guard<std::mutex> lock(mutex);
    if (cached_bytes) *cached_bytes = bytes;
    if (entry_count) *entry_count = entries.size();
    if (cache_hits) *cache_hits = hits;
    if (cache_misses) *cache_misses = misses;
  }

  size_t current_limit() {
    std::lock_guard<std::mutex> lock(mutex);
    return limit_locked();
  }

  void trim() {
    std::lock_guard<std::mutex> lock(mutex);
    entries.clear();
    index.clear();
    bytes = 0;
    cached_limit = kFallbackLimit;
    last_budget_query = {};
  }
};

inline WeightBufferCache& weight_cache() {
  static WeightBufferCache cache;
  return cache;
}

inline void weight_cache_stats(size_t* cached_bytes, size_t* entries,
                               uint64_t* hits, uint64_t* misses) {
  weight_cache().stats(cached_bytes, entries, hits, misses);
}

inline size_t weight_cache_limit() {
  return weight_cache().current_limit();
}

inline void weight_cache_trim() {
  weight_cache().trim();
}

// A bounded bridge cache keyed by the CPU mirror returned for an SSBO result.
// It is deliberately separate from the weight cache: entries are evictable
// activation buffers, and a shared_ptr held by an in-flight operator keeps an
// evicted buffer alive until that dispatch has finished.
struct ResidentBufferCache {
  struct Entry {
    std::shared_ptr<Buffer> buffer;
    size_t bytes = 0;
    uint64_t last = 0;
  };

  std::mutex mutex;
  std::unordered_map<const float*, Entry> entries;
  size_t bytes = 0;
  uint64_t clock = 0;
  uint64_t hits = 0;
  uint64_t misses = 0;
  static constexpr size_t kFallbackLimit = 1ull * 1024ull * 1024ull * 1024ull;
  static constexpr size_t kSafetyReserve = 256ull * 1024ull * 1024ull;
  size_t cached_limit = kFallbackLimit;
  std::chrono::steady_clock::time_point last_budget_query{};

  static bool enabled() {
    static const bool value = [] {
      const char* env = std::getenv("VKSSBO_RESIDENT_CACHE");
      return !env || env[0] != '0';
    }();
    return value;
  }

  size_t limit_locked(VkDeviceSize incoming = 0) {
    const auto now = std::chrono::steady_clock::now();
    if (incoming == 0 && last_budget_query.time_since_epoch().count() != 0 &&
        now - last_budget_query < std::chrono::milliseconds(250)) {
      return cached_limit;
    }
    auto budget = Device::get().query_memory_budget();
#ifdef VKSSBO_USE_RUST_MEMORY_POLICY
    VkssboMemoryPlan plan = memory_policy_plan(
        budget, 0, 0, bytes, kSafetyReserve, 0, 0, incoming, 3);
    size_t limit = static_cast<size_t>(plan.resident_limit);
#else
    size_t limit = kFallbackLimit;
    if (budget.budget > 0) {
      VkDeviceSize safe_budget = (budget.budget / 5u) * 4u;
      safe_budget = safe_budget > kSafetyReserve
          ? safe_budget - kSafetyReserve : 0;
      VkDeviceSize non_resident = budget.usage > bytes
          ? budget.usage - static_cast<VkDeviceSize>(bytes) : 0;
      limit = safe_budget > non_resident
          ? static_cast<size_t>(safe_budget - non_resident) : 0;
    }
#endif
    const char* env = std::getenv("VKSSBO_RESIDENT_CACHE_MB");
    if (env && env[0]) {
      long long mb = atoll(env);
      if (mb >= 0) {
        const size_t configured = static_cast<size_t>(mb) * 1024ull * 1024ull;
        if (configured < limit) limit = configured;
      }
    }
    cached_limit = limit;
    last_budget_query = now;
    return limit;
  }

  std::shared_ptr<Buffer> find(const float* host, VkDeviceSize requested) {
    if (!enabled() || !host) return {};
    std::lock_guard<std::mutex> lock(mutex);
    auto it = entries.find(host);
    if (it == entries.end() || it->second.bytes < requested) {
      ++misses;
      return {};
    }
    it->second.last = ++clock;
    ++hits;
    return it->second.buffer;
  }

  void stats(size_t* cached_bytes, size_t* entry_count,
             uint64_t* cache_hits, uint64_t* cache_misses) {
    std::lock_guard<std::mutex> lock(mutex);
    if (cached_bytes) *cached_bytes = bytes;
    if (entry_count) *entry_count = entries.size();
    if (cache_hits) *cache_hits = hits;
    if (cache_misses) *cache_misses = misses;
  }

  void insert(const float* host, std::shared_ptr<Buffer> buffer) {
    if (!enabled() || !host || !buffer || !buffer->buffer) return;
    std::lock_guard<std::mutex> lock(mutex);
    auto existing = entries.find(host);
    if (existing != entries.end()) {
      bytes -= existing->second.bytes;
      entries.erase(existing);
    }
    const size_t incoming = static_cast<size_t>(buffer->size);
    const size_t limit = limit_locked(static_cast<VkDeviceSize>(incoming));
    while ((incoming > limit || bytes > limit - incoming) && !entries.empty()) {
      auto oldest = entries.begin();
      for (auto it = entries.begin(); it != entries.end(); ++it) {
        if (it->second.last < oldest->second.last) oldest = it;
      }
      bytes -= oldest->second.bytes;
      entries.erase(oldest);
    }
    if (incoming > limit || bytes > limit - incoming) return;
    entries.emplace(host, Entry{std::move(buffer), incoming, ++clock});
    bytes += incoming;
  }

  void erase(const float* host) {
    if (!host) return;
    std::lock_guard<std::mutex> lock(mutex);
    auto it = entries.find(host);
    if (it != entries.end()) {
      bytes -= it->second.bytes;
      entries.erase(it);
    }
  }

  void trim() {
    std::lock_guard<std::mutex> lock(mutex);
    entries.clear();
    bytes = 0;
    hits = 0;
    misses = 0;
    cached_limit = kFallbackLimit;
    last_budget_query = {};
  }
};

inline ResidentBufferCache& resident_cache() {
  static ResidentBufferCache cache;
  return cache;
}

inline size_t resident_cache_limit() {
  std::lock_guard<std::mutex> lock(resident_cache().mutex);
  return resident_cache().limit_locked();
}

inline void resident_cache_trim() {
  resident_cache().trim();
}

inline void resident_cache_stats(size_t* cached_bytes, size_t* entries,
                                 uint64_t* hits, uint64_t* misses) {
  resident_cache().stats(cached_bytes, entries, hits, misses);
}

// Host-visible buffer for staging data in/out
Buffer create_host_buffer(VkDeviceSize size) {
  Buffer pooled;
  if (acquire_staging_buffer(size, &pooled)) return pooled;
  Buffer b = create_buffer(
      size,
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
          VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT,
      VMA_MEMORY_USAGE_AUTO_PREFER_HOST);
  b.staging_recyclable = StagingBufferPool::enabled() &&
      size >= StagingBufferPool::min_size();
  return b;
}

// Device-local buffer for compute; on iGPU shared memory this is
// host-visible so we can map/write directly (no staging copies).
Buffer create_device_buffer(VkDeviceSize size) {
  Buffer pooled;
  if (acquire_transient_buffer(size, &pooled)) return pooled;
  VmaAllocationCreateFlags flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
  if (Device::get().memory_topology == Device::MemoryTopology::UMA) {
    flags |= VMA_ALLOCATION_CREATE_MAPPED_BIT;
  }
  if (Device::get().has_device_local_only &&
      Device::get().memory_topology != Device::MemoryTopology::UMA) {
    flags = 0;
  }
  if (Device::get().simulation_force_staging) {
    flags = 0;
  }
  Buffer b = create_buffer(size,
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
          VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      flags,
      VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
  b.recyclable = transient_pool().enabled() && transient_pool().eligible(size);
  return b;
}

Buffer create_weight_buffer(VkDeviceSize size) {
  VmaAllocationCreateFlags flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
  if (Device::get().memory_topology == Device::MemoryTopology::UMA) {
    flags |= VMA_ALLOCATION_CREATE_MAPPED_BIT;
  }
  if (Device::get().has_device_local_only &&
      Device::get().memory_topology != Device::MemoryTopology::UMA) {
    flags = 0;
  }
  if (Device::get().simulation_force_staging) {
    flags = 0;
  }
  return create_buffer(
      size,
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
          VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      flags,
      VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
}

void* map_buffer(Buffer& b) {
  if (b.map_ptr) return b.map_ptr;
  auto& d = Device::get();
  void* p = nullptr;
  VKSSBO_CHECK(vmaMapMemory(d.allocator, b.allocation, &p));
  b.mapped = true;
  b.map_ptr = p;
  return p;
}

void unmap_buffer(Buffer& b) {
  if (b.persistent_mapped) return;
  if (b.mapped) {
    vmaUnmapMemory(Device::get().allocator, b.allocation);
    b.mapped = false;
    b.map_ptr = nullptr;
  }
}

void flush_buffer(Buffer& b, VkDeviceSize offset = 0,
                  VkDeviceSize size = VK_WHOLE_SIZE) {
  VKSSBO_CHECK(vmaFlushAllocation(
      Device::get().allocator, b.allocation, offset, size));
}

void invalidate_buffer(Buffer& b, VkDeviceSize offset = 0,
                       VkDeviceSize size = VK_WHOLE_SIZE) {
  VKSSBO_CHECK(vmaInvalidateAllocation(
      Device::get().allocator, b.allocation, offset, size));
}

// ---------- Compute pipeline ----------
struct Pipeline {
  VkPipelineLayout layout = VK_NULL_HANDLE;
  VkPipeline pipeline = VK_NULL_HANDLE;
};

enum class RuntimeSubmissionKind : uint32_t {
  Compute = 0,
  Transfer = 1,
  Copy = 2,
};

// Opt-in profiling for the bridge boundary. Host counters cover conversion,
// staging, submission, and synchronization; gpu_command_ns comes from Vulkan
// timestamp queries around the compute command buffer.
struct RuntimeProfileSnapshot {
  uint64_t schema_version = 2;
  uint64_t upload_host_ns = 0;
  uint64_t upload_bytes = 0;
  uint64_t upload_calls = 0;
  uint64_t submit_wait_ns = 0;
  uint64_t compute_submissions = 0;
  uint64_t transfer_submissions = 0;
  uint64_t copy_submissions = 0;
  uint64_t fence_waits = 0;
  uint64_t queue_wait_idle_count = 0;
  uint64_t gpu_command_ns = 0;
  uint64_t gpu_timestamp_submissions = 0;
  uint64_t gpu_timestamp_failures = 0;
  uint64_t dispatches = 0;
  uint64_t download_host_ns = 0;
  uint64_t download_transfer_wait_ns = 0;
  uint64_t download_bytes = 0;
  uint64_t download_submissions = 0;
  uint64_t download_calls = 0;
};

struct RuntimeProfile {
  static constexpr uint64_t kSchemaVersion = 2;
  std::mutex mutex;
  RuntimeProfileSnapshot snapshot{ kSchemaVersion };

  static bool enabled() {
    const char* value = std::getenv("VKSSBO_PROFILE");
    return value && value[0] != '\0' && value[0] != '0';
  }

  void reset() {
    std::lock_guard<std::mutex> lock(mutex);
    snapshot = {};
    snapshot.schema_version = kSchemaVersion;
  }

  RuntimeProfileSnapshot read() {
    std::lock_guard<std::mutex> lock(mutex);
    return snapshot;
  }

  void record_upload(uint64_t elapsed_ns, uint64_t bytes) {
    std::lock_guard<std::mutex> lock(mutex);
    snapshot.upload_host_ns += elapsed_ns;
    snapshot.upload_bytes += bytes;
    ++snapshot.upload_calls;
  }

  void record_submit_wait(uint64_t elapsed_ns) {
    std::lock_guard<std::mutex> lock(mutex);
    snapshot.submit_wait_ns += elapsed_ns;
  }

  void record_submission(RuntimeSubmissionKind kind, bool fence_wait) {
    std::lock_guard<std::mutex> lock(mutex);
    switch (kind) {
      case RuntimeSubmissionKind::Compute:
        ++snapshot.compute_submissions;
        break;
      case RuntimeSubmissionKind::Transfer:
        ++snapshot.transfer_submissions;
        break;
      case RuntimeSubmissionKind::Copy:
        ++snapshot.copy_submissions;
        break;
    }
    if (fence_wait) ++snapshot.fence_waits;
  }

  void record_gpu_timestamp(bool success, uint64_t elapsed_ns) {
    std::lock_guard<std::mutex> lock(mutex);
    if (success) {
      snapshot.gpu_command_ns += elapsed_ns;
      ++snapshot.gpu_timestamp_submissions;
    } else {
      ++snapshot.gpu_timestamp_failures;
    }
  }

  void record_dispatch() {
    std::lock_guard<std::mutex> lock(mutex);
    ++snapshot.dispatches;
  }

  void record_queue_wait_idle() {
    std::lock_guard<std::mutex> lock(mutex);
    ++snapshot.queue_wait_idle_count;
  }

  void record_download(uint64_t elapsed_ns, uint64_t bytes) {
    std::lock_guard<std::mutex> lock(mutex);
    snapshot.download_host_ns += elapsed_ns;
    snapshot.download_bytes += bytes;
    ++snapshot.download_submissions;
    ++snapshot.download_calls;
  }

  void record_download_transfer_wait(uint64_t elapsed_ns) {
    std::lock_guard<std::mutex> lock(mutex);
    snapshot.download_transfer_wait_ns += elapsed_ns;
  }
};

inline RuntimeProfile& runtime_profile() {
  static RuntimeProfile profile;
  return profile;
}

inline uint64_t runtime_profile_elapsed_ns(
    const std::chrono::steady_clock::time_point& started) {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - started)
          .count());
}

struct ComputeManager {
  struct InFlightBatch {
    uint64_t value = 0;
    VkFence fence = VK_NULL_HANDLE;
    VkCommandBuffer command_buffer = VK_NULL_HANDLE;
    VkCommandBuffer transfer_command_buffer = VK_NULL_HANDLE;
    std::vector<Buffer> staging_buffers;
    bool had_transfer = false;
    bool profile_timestamp_requested = false;
    bool profile_timestamp_recording = false;
  };

  static constexpr uint32_t kAutotuneTimestampQueryCount = 12;
  static constexpr uint32_t kProfileTimestampBegin = 30;
  static constexpr uint32_t kProfileTimestampEnd = 31;

  std::mutex mutex;
  VkCommandBuffer cmd = VK_NULL_HANDLE;
  VkCommandBuffer transfer_cmd = VK_NULL_HANDLE;
  bool transfer_recording = false;
  std::vector<Buffer> staging_buffers;
  bool transfer_pending = false;
  bool same_queue_transfer_pending = false;
  size_t staging_bytes = 0;
  size_t staging_peak_bytes = 0;
  uint64_t staging_uploads = 0;
  uint64_t staging_downloads = 0;
  uint64_t submit_count = 0;
  uint64_t fence_wait_count = 0;
  bool profile_timestamp_requested = false;
  bool profile_timestamp_recording = false;
  // Reuse one fence for the serialized submit-and-wait path. Fence creation
  // is relatively expensive for the small transfers used by many operators.
  VkFence submit_fence = VK_NULL_HANDLE;
  std::vector<InFlightBatch> in_flight;
  std::vector<VkCommandBuffer> reusable_command_buffers;
  std::vector<VkCommandBuffer> reusable_transfer_command_buffers;
  uint64_t next_completion_value = 0;
  uint64_t completed_value = 0;
  uint64_t poll_count = 0;
  uint64_t wait_count = 0;
  uint64_t in_flight_peak = 0;
  bool recording = false;

  void simulation_transfer_delay() const {
    const uint64_t us = Device::get().simulation_transfer_latency_us;
    if (Device::get().simulation_enabled && us != 0) {
      std::this_thread::sleep_for(std::chrono::microseconds(us));
    }
  }
  uint64_t queue_wait_idle_count = 0;
  std::unordered_map<std::string, Pipeline> pipelines;
  std::unordered_map<std::string, VkShaderModule> modules;

  static ComputeManager& get() {
    static ComputeManager m;
    return m;
  }

  VkShaderModule compile(const std::string& name, const std::vector<uint32_t>& spv) {
    auto it = modules.find(name);
    if (it != modules.end()) return it->second;
    VkShaderModuleCreateInfo smci{};
    smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = spv.size() * sizeof(uint32_t);
    smci.pCode = spv.data();
    VkShaderModule sm = VK_NULL_HANDLE;
    VKSSBO_CHECK(vkCreateShaderModule(Device::get().device, &smci, nullptr, &sm));
    modules[name] = sm;
    return sm;
  }

  Pipeline& get_pipeline_locked(const std::string& name, const std::vector<uint32_t>& spv) {
    // caller holds mutex
    fprintf(stderr, "[pipe] %s spv_size=%zu\n", name.c_str(), spv.size());
    auto it = pipelines.find(name);
    if (it != pipelines.end()) return it->second;
    auto& d = Device::get();
    VkShaderModule sm = compile(name, spv);

    VkPipelineShaderStageCreateInfo psci{};
    psci.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    psci.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    psci.module = sm;
    psci.pName = "main";

    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.offset = 0;
    pcr.size = 128;

    VkPipelineLayoutCreateInfo plci{};
    plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &d.global_ds_layout;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &pcr;

    Pipeline p;
    VKSSBO_CHECK(vkCreatePipelineLayout(d.device, &plci, nullptr, &p.layout));
    fprintf(stderr, "[pipe] layout ok %p\n", (void*)p.layout);

    VkComputePipelineCreateInfo cpci{};
    cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.stage = psci;
    cpci.layout = p.layout;
    VkResult pr = vkCreateComputePipelines(d.device, VK_NULL_HANDLE, 1, &cpci, nullptr, &p.pipeline);
    fprintf(stderr, "[pipe] create pipelines res=%d\n", (int)pr);
    VKSSBO_CHECK(pr);

    pipelines[name] = p;
    return pipelines[name];
  }

  Pipeline& get_pipeline(const std::string& name, const std::vector<uint32_t>& spv) {
    std::lock_guard<std::mutex> lock(mutex);
    return get_pipeline_locked(name, spv);
  }

  void begin() {
    poll();
    if (recording) throw Error("command buffer is already recording");
    auto& d = Device::get();
    staging_buffers.clear();
    staging_bytes = 0;
    transfer_pending = false;
    same_queue_transfer_pending = false;
    transfer_recording = false;
    if (!cmd) {
      if (!reusable_command_buffers.empty()) {
        cmd = reusable_command_buffers.back();
        reusable_command_buffers.pop_back();
      }
    }
    if (!cmd) {
      VkCommandBufferAllocateInfo cbai{};
      cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
      cbai.commandPool = d.cmd_pool;
      cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
      cbai.commandBufferCount = 1;
      VKSSBO_CHECK(vkAllocateCommandBuffers(d.device, &cbai, &cmd));
    }
    VKSSBO_CHECK(vkResetCommandBuffer(cmd, 0));
    VkCommandBufferBeginInfo cbbi{};
    cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VKSSBO_CHECK(vkBeginCommandBuffer(cmd, &cbbi));
    recording = true;
    profile_timestamp_requested = RuntimeProfile::enabled();
    profile_timestamp_recording =
        profile_timestamp_requested && timestamps_supported() &&
        in_flight.empty();
    if (profile_timestamp_recording) {
      vkCmdResetQueryPool(
          cmd, d.timestamp_pool, kProfileTimestampBegin, 2);
      vkCmdWriteTimestamp(
          cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, d.timestamp_pool,
          kProfileTimestampBegin);
    }
  }

  void begin_transfer_if_needed() {
    auto& d = Device::get();
    if (!d.async_transfer_enabled || transfer_recording) return;
    if (!transfer_cmd) {
      if (!reusable_transfer_command_buffers.empty()) {
        transfer_cmd = reusable_transfer_command_buffers.back();
        reusable_transfer_command_buffers.pop_back();
      } else {
        VkCommandBufferAllocateInfo cbai{};
        cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cbai.commandPool = d.transfer_cmd_pool;
        cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbai.commandBufferCount = 1;
        VKSSBO_CHECK(vkAllocateCommandBuffers(d.device, &cbai, &transfer_cmd));
      }
    }
    VKSSBO_CHECK(vkResetCommandBuffer(transfer_cmd, 0));
    VkCommandBufferBeginInfo cbbi{};
    cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VKSSBO_CHECK(vkBeginCommandBuffer(transfer_cmd, &cbbi));
    transfer_recording = true;
  }

  void mark_transfer_pending(bool on_transfer_queue) {
    transfer_pending = true;
    if (!on_transfer_queue) same_queue_transfer_pending = true;
  }

  void retain_staging(Buffer&& buffer) {
    staging_bytes += static_cast<size_t>(buffer.size);
    staging_peak_bytes = std::max(staging_peak_bytes, staging_bytes);
    staging_buffers.push_back(std::move(buffer));
  }

  void record_staging_upload(VkDeviceSize bytes) {
    (void)bytes;
    ++staging_uploads;
  }

  void record_staging_download(VkDeviceSize bytes) {
    (void)bytes;
    ++staging_downloads;
  }

  void clear_staging() {
    staging_buffers.clear();
    staging_bytes = 0;
  }

  void barrier_transfer_to_compute() {
    if (!transfer_pending) return;
    if (!same_queue_transfer_pending) {
      // A timeline wait on the compute submission provides the cross-queue
      // execution and memory dependency. Buffers are concurrent-share mode.
      return;
    }
    VkMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(
        cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 1, &barrier, 0, nullptr, 0, nullptr);
    same_queue_transfer_pending = false;
    if (!transfer_recording) transfer_pending = false;
  }

  // Intel and a few other Vulkan drivers do not reliably infer ordering
  // between adjacent compute dispatches in one command buffer.  Use an
  // explicit shader barrier when the autotuner records two candidates.
  void barrier_compute_to_compute() {
    VkMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(
        cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr,
        0, nullptr);
  }

  bool timestamps_supported() const {
    return Device::get().timestamp_pool != VK_NULL_HANDLE &&
        Device::get().timestamp_valid_bits != 0;
  }

  void reset_timestamps() {
    if (timestamps_supported()) {
      vkCmdResetQueryPool(
          cmd, Device::get().timestamp_pool, 0,
          kAutotuneTimestampQueryCount);
    }
  }

  void write_timestamp(uint32_t index) {
    if (timestamps_supported() && index < Device::kTimestampQueryCount) {
      vkCmdWriteTimestamp(
          cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
          Device::get().timestamp_pool, index);
    }
  }

  bool read_timestamp_pair(uint32_t begin, uint32_t end, double* ns) {
    if (!ns || !timestamps_supported() ||
        begin >= Device::kTimestampQueryCount ||
        end >= Device::kTimestampQueryCount || end != begin + 1) return false;
    std::array<uint64_t, 2> values{};
    VkResult r = vkGetQueryPoolResults(
        Device::get().device, Device::get().timestamp_pool, begin, 2,
        sizeof(values), values.data(), sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);
    if (r != VK_SUCCESS) return false;
    uint64_t delta = 0;
    const uint32_t valid_bits = Device::get().timestamp_valid_bits;
    if (valid_bits > 0 && valid_bits < 64) {
      const uint64_t mask = (1ull << valid_bits) - 1ull;
      delta = (values[1] - values[0]) & mask;
    } else {
      delta = values[1] - values[0];
    }
    *ns = static_cast<double>(delta) *
        static_cast<double>(Device::get().timestamp_period_ns);
    return *ns >= 0.0;
  }

  void finish_profile_timestamp() {
    if (profile_timestamp_recording) {
      vkCmdWriteTimestamp(
          cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
          Device::get().timestamp_pool, kProfileTimestampEnd);
    }
  }

  void collect_profile_timestamp() {
    if (!profile_timestamp_requested) return;
    double elapsed_ns = 0.0;
    const bool success = profile_timestamp_recording &&
        read_timestamp_pair(
            kProfileTimestampBegin, kProfileTimestampEnd, &elapsed_ns);
    runtime_profile().record_gpu_timestamp(
        success, success ? static_cast<uint64_t>(elapsed_ns + 0.5) : 0);
    profile_timestamp_requested = false;
    profile_timestamp_recording = false;
  }

  void ensure_submit_fence() {
    if (submit_fence != VK_NULL_HANDLE) return;
    VkFenceCreateInfo fci{};
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VKSSBO_CHECK(vkCreateFence(Device::get().device, &fci, nullptr, &submit_fence));
  }

  void submit_command_buffer(
      VkQueue submit_queue, VkCommandBuffer submit_cmd, const char* kind,
      VkFence fence) {
    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &submit_cmd;
    VKSSBO_CHECK(vkQueueSubmit(submit_queue, 1, &si, fence));
    if (RuntimeProfile::enabled()) {
      const RuntimeSubmissionKind submission_kind =
          kind && std::strcmp(kind, "copy") == 0
          ? RuntimeSubmissionKind::Copy
          : RuntimeSubmissionKind::Compute;
      runtime_profile().record_submission(submission_kind, true);
    }
  }

  // Wait only for this submission instead of idling the entire queue. This
  // preserves the old synchronous API contract while allowing future
  // independent transfer/compute submissions to remain in flight.
  void submit_command_buffer_and_wait(
      VkQueue submit_queue, VkCommandBuffer submit_cmd, const char* kind) {
    auto& d = Device::get();
    ensure_submit_fence();
    VKSSBO_CHECK(vkResetFences(d.device, 1, &submit_fence));
    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &submit_cmd;
    const VkResult submit_result =
        vkQueueSubmit(submit_queue, 1, &si, submit_fence);
    VKSSBO_CHECK(submit_result);
    const VkResult wait_result =
        vkWaitForFences(d.device, 1, &submit_fence, VK_TRUE, UINT64_MAX);
    VKSSBO_CHECK(wait_result);
    ++submit_count;
    ++fence_wait_count;
    if (RuntimeProfile::enabled()) {
      const RuntimeSubmissionKind submission_kind =
          kind && std::strcmp(kind, "copy") == 0
          ? RuntimeSubmissionKind::Copy
          : RuntimeSubmissionKind::Compute;
      runtime_profile().record_submission(submission_kind, true);
    }
    if (std::getenv("VKSSBO_TRACE_SUBMIT")) {
      fprintf(stderr,
              "[vkssbo] submit kind=%s mode=fence wait_idle=0 "
              "submissions=%llu fence_waits=%llu\n",
              kind ? kind : "unknown",
              static_cast<unsigned long long>(submit_count),
              static_cast<unsigned long long>(fence_wait_count));
    }
  }

  void submit_transfer_signal(uint64_t signal_value) {
    auto& d = Device::get();
    VkTimelineSemaphoreSubmitInfo timeline_info{};
    timeline_info.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
    timeline_info.signalSemaphoreValueCount = 1;
    timeline_info.pSignalSemaphoreValues = &signal_value;
    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.pNext = &timeline_info;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &transfer_cmd;
    si.signalSemaphoreCount = 1;
    si.pSignalSemaphores = &d.transfer_timeline;
    VKSSBO_CHECK(vkQueueSubmit(d.transfer_queue, 1, &si, VK_NULL_HANDLE));
    if (RuntimeProfile::enabled()) {
      runtime_profile().record_submission(
          RuntimeSubmissionKind::Transfer, false);
    }
    if (std::getenv("VKSSBO_TRACE_SUBMIT")) {
      fprintf(stderr, "[vkssbo] submit kind=transfer mode=timeline signal=%llu\n",
              static_cast<unsigned long long>(signal_value));
    }
  }

  void submit_compute_with_timeline(
      uint64_t wait_value, VkFence fence) {
    auto& d = Device::get();
    VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    VkTimelineSemaphoreSubmitInfo timeline_info{};
    timeline_info.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
    timeline_info.waitSemaphoreValueCount = 1;
    timeline_info.pWaitSemaphoreValues = &wait_value;
    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.pNext = &timeline_info;
    si.waitSemaphoreCount = 1;
    si.pWaitSemaphores = &d.transfer_timeline;
    si.pWaitDstStageMask = &wait_stage;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    VKSSBO_CHECK(vkQueueSubmit(d.queue, 1, &si, fence));
    if (RuntimeProfile::enabled()) {
      runtime_profile().record_submission(
          RuntimeSubmissionKind::Compute, true);
    }
  }

  void submit_compute_with_timeline_and_wait(uint64_t wait_value) {
    auto& d = Device::get();
    ensure_submit_fence();
    VKSSBO_CHECK(vkResetFences(d.device, 1, &submit_fence));
    VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    VkTimelineSemaphoreSubmitInfo timeline_info{};
    timeline_info.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
    timeline_info.waitSemaphoreValueCount = 1;
    timeline_info.pWaitSemaphoreValues = &wait_value;
    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.pNext = &timeline_info;
    si.waitSemaphoreCount = 1;
    si.pWaitSemaphores = &d.transfer_timeline;
    si.pWaitDstStageMask = &wait_stage;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    const VkResult submit_result = vkQueueSubmit(d.queue, 1, &si, submit_fence);
    VKSSBO_CHECK(submit_result);
    const VkResult wait_result =
        vkWaitForFences(d.device, 1, &submit_fence, VK_TRUE, UINT64_MAX);
    VKSSBO_CHECK(wait_result);
    ++submit_count;
    ++fence_wait_count;
    if (RuntimeProfile::enabled()) {
      runtime_profile().record_submission(
          RuntimeSubmissionKind::Compute, true);
    }
    if (std::getenv("VKSSBO_TRACE_SUBMIT")) {
      fprintf(stderr,
              "[vkssbo] submit kind=compute mode=timeline wait=%llu "
              "wait_idle=0 submissions=%llu fence_waits=%llu\n",
              static_cast<unsigned long long>(wait_value),
              static_cast<unsigned long long>(submit_count),
              static_cast<unsigned long long>(fence_wait_count));
    }
  }

  uint64_t submit() {
    if (!recording || !cmd) throw Error("no command buffer is recording");
    auto& d = Device::get();
    const bool has_transfer =
        d.async_transfer_enabled && transfer_pending && transfer_recording;
    const bool had_transfer = transfer_pending;
    uint64_t transfer_value = 0;
    if (has_transfer) {
      VKSSBO_CHECK(vkEndCommandBuffer(transfer_cmd));
      transfer_recording = false;
      transfer_value = ++d.transfer_timeline_value;
      submit_transfer_signal(transfer_value);
    }
    finish_profile_timestamp();
    VKSSBO_CHECK(vkEndCommandBuffer(cmd));
    recording = false;

    VkFenceCreateInfo fci{};
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence fence = VK_NULL_HANDLE;
    VKSSBO_CHECK(vkCreateFence(d.device, &fci, nullptr, &fence));
    try {
      if (has_transfer) {
        submit_compute_with_timeline(transfer_value, fence);
      } else {
        submit_command_buffer(d.queue, cmd, "compute", fence);
      }
    } catch (...) {
      vkDestroyFence(d.device, fence, nullptr);
      throw;
    }

    InFlightBatch batch;
    batch.value = ++next_completion_value;
    batch.fence = fence;
    batch.command_buffer = cmd;
    batch.transfer_command_buffer = has_transfer ? transfer_cmd : VK_NULL_HANDLE;
    batch.staging_buffers = std::move(staging_buffers);
    batch.had_transfer = had_transfer;
    batch.profile_timestamp_requested = profile_timestamp_requested;
    batch.profile_timestamp_recording = profile_timestamp_recording;
    in_flight.push_back(std::move(batch));
    in_flight_peak = std::max<uint64_t>(
        in_flight_peak, static_cast<uint64_t>(in_flight.size()));
    cmd = VK_NULL_HANDLE;
    if (has_transfer) transfer_cmd = VK_NULL_HANDLE;
    staging_bytes = 0;
    transfer_pending = false;
    same_queue_transfer_pending = false;
    profile_timestamp_requested = false;
    profile_timestamp_recording = false;
    ++submit_count;
    if (std::getenv("VKSSBO_TRACE_SUBMIT")) {
      if (has_transfer) {
        // The compute submission waits on the transfer timeline, while the
        // fence is retained only for completion/reclamation of the batch.
        fprintf(stderr,
                "[vkssbo] submit kind=compute mode=timeline wait=%llu "
                "completion=fence value=%llu in_flight=%zu\n",
                static_cast<unsigned long long>(transfer_value),
                static_cast<unsigned long long>(next_completion_value),
                in_flight.size());
      } else {
        fprintf(stderr,
                "[vkssbo] submit kind=compute mode=fence value=%llu "
                "in_flight=%zu\n",
                static_cast<unsigned long long>(next_completion_value),
                in_flight.size());
      }
    }
    return next_completion_value;
  }

  void collect_profile_timestamp(
      bool requested, bool recording_timestamp) {
    if (!requested) return;
    double elapsed_ns = 0.0;
    const bool success = recording_timestamp &&
        read_timestamp_pair(
            kProfileTimestampBegin, kProfileTimestampEnd, &elapsed_ns);
    runtime_profile().record_gpu_timestamp(
        success, success ? static_cast<uint64_t>(elapsed_ns + 0.5) : 0);
  }

  void retire_completed(bool force, uint64_t target) {
    auto& d = Device::get();
    while (!in_flight.empty()) {
      InFlightBatch& batch = in_flight.front();
      if (target != 0 && batch.value > target) break;
      const VkResult status = force
          ? vkWaitForFences(d.device, 1, &batch.fence, VK_TRUE, UINT64_MAX)
          : vkGetFenceStatus(d.device, batch.fence);
      if (status == VK_NOT_READY) break;
      VKSSBO_CHECK(status);
      collect_profile_timestamp(
          batch.profile_timestamp_requested,
          batch.profile_timestamp_recording);
      if (batch.had_transfer) simulation_transfer_delay();
      vkDestroyFence(d.device, batch.fence, nullptr);
      batch.fence = VK_NULL_HANDLE;
      if (batch.command_buffer != VK_NULL_HANDLE) {
        if (cmd == VK_NULL_HANDLE && !recording) {
          cmd = batch.command_buffer;
        } else {
          reusable_command_buffers.push_back(batch.command_buffer);
        }
        batch.command_buffer = VK_NULL_HANDLE;
      }
      if (batch.transfer_command_buffer != VK_NULL_HANDLE) {
        if (transfer_cmd == VK_NULL_HANDLE && !transfer_recording) {
          transfer_cmd = batch.transfer_command_buffer;
        } else {
          reusable_transfer_command_buffers.push_back(
              batch.transfer_command_buffer);
        }
        batch.transfer_command_buffer = VK_NULL_HANDLE;
      }
      batch.staging_buffers.clear();
      completed_value = batch.value;
      in_flight.erase(in_flight.begin());
    }
  }

  bool poll() {
    ++poll_count;
    const uint64_t before = completed_value;
    retire_completed(false, 0);
    return completed_value != before;
  }

  void wait(uint64_t value) {
    if (value == 0 || value <= completed_value) return;
    if (value > next_completion_value) {
      throw Error("completion value is not submitted");
    }
    ++wait_count;
    ++fence_wait_count;
    retire_completed(true, value);
  }

  void submit_and_wait() {
    const bool profile_enabled = RuntimeProfile::enabled();
    const auto profile_started = profile_enabled
        ? std::chrono::steady_clock::now()
        : std::chrono::steady_clock::time_point{};
    const uint64_t value = submit();
    wait(value);
    if (profile_enabled) {
      runtime_profile().record_submit_wait(
          runtime_profile_elapsed_ns(profile_started));
    }
  }

  // Abort only work that is still being recorded. Submitted batches remain
  // tracked in `in_flight` and are retired normally; clearing them here could
  // release resources while the GPU still owns them.
  void abort() noexcept {
    if (recording && cmd != VK_NULL_HANDLE) {
      (void)vkResetCommandBuffer(cmd, 0);
      recording = false;
    }
    if (transfer_recording && transfer_cmd != VK_NULL_HANDLE) {
      (void)vkResetCommandBuffer(transfer_cmd, 0);
      transfer_recording = false;
    }
    transfer_pending = false;
    same_queue_transfer_pending = false;
    profile_timestamp_requested = false;
    profile_timestamp_recording = false;
    staging_buffers.clear();
    staging_bytes = 0;
  }

  void scheduler_stats(
      uint64_t* last_submitted, uint64_t* completed, uint64_t* in_flight_count,
      uint64_t* peak_in_flight, uint64_t* submits, uint64_t* polls,
      uint64_t* waits) const {
    if (last_submitted) *last_submitted = next_completion_value;
    if (completed) *completed = completed_value;
    if (in_flight_count) *in_flight_count = in_flight.size();
    if (peak_in_flight) *peak_in_flight = in_flight_peak;
    if (submits) *submits = submit_count;
    if (polls) *polls = poll_count;
    if (waits) *waits = wait_count;
  }

  // Copy between a device-local buffer and a host-visible staging buffer after
  // the producer command buffer has completed. This is intentionally
  // synchronous until a discrete-GPU transfer queue/timeline path is proven.
  void copy_buffer_and_wait(VkBuffer src, VkBuffer dst, VkDeviceSize size) {
    const bool profile_enabled = RuntimeProfile::enabled();
    const auto profile_started = profile_enabled
        ? std::chrono::steady_clock::now()
        : std::chrono::steady_clock::time_point{};
    auto& d = Device::get();
    VkCommandBuffer copy_cmd = cmd;
    VkQueue copy_queue = d.queue;
    const bool use_async = d.async_transfer_for(size);
    if (use_async) {
      begin_transfer_if_needed();
      copy_cmd = transfer_cmd;
      copy_queue = d.transfer_queue;
    } else {
      VKSSBO_CHECK(vkResetCommandBuffer(cmd, 0));
    }
    VkCommandBufferBeginInfo cbbi{};
    cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (!use_async) {
      VKSSBO_CHECK(vkBeginCommandBuffer(copy_cmd, &cbbi));
    }
    VkBufferCopy region{};
    region.size = size;
    vkCmdCopyBuffer(copy_cmd, src, dst, 1, &region);
    VKSSBO_CHECK(vkEndCommandBuffer(copy_cmd));
    if (use_async) transfer_recording = false;
    submit_command_buffer_and_wait(copy_queue, copy_cmd, "copy");
    record_staging_download(size);
    simulation_transfer_delay();
    if (profile_enabled) {
      runtime_profile().record_download_transfer_wait(
          runtime_profile_elapsed_ns(profile_started));
    }
  }

  void staging_stats(size_t* current, size_t* peak, uint64_t* uploads,
                    uint64_t* downloads) const {
    if (current) *current = staging_bytes;
    if (peak) *peak = staging_peak_bytes;
    if (uploads) *uploads = staging_uploads;
    if (downloads) *downloads = staging_downloads;
  }
};

inline void staging_stats(size_t* current, size_t* peak, uint64_t* uploads,
                          uint64_t* downloads) {
  ComputeManager::get().staging_stats(current, peak, uploads, downloads);
}

inline void abort_compute_manager() noexcept {
  try {
    ComputeManager::get().abort();
  } catch (...) {
    // Cleanup must never mask the original operation failure.
  }
}

} // namespace vkssbo
