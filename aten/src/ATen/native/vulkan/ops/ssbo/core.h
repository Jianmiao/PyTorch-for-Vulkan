// vulkan_ssbo: standalone SSBO-based Vulkan compute backend for PyTorch.
// No texture storage: all tensors live in linear Shader Storage Buffers,
// which have no dimension limits (fixes maxImageDimension3D=16384 crashes).
#include <vulkan/vulkan.h>
#include <include/vk_mem_alloc.h>
#include <vector>
#include <string>
#include <mutex>
#include <unordered_map>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <stdexcept>

#ifdef _WIN32
#define VKSSBO_EXPORT __declspec(dllexport)
#else
#define VKSSBO_EXPORT
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
  VkInstance instance = VK_NULL_HANDLE;
  VkPhysicalDevice phys = VK_NULL_HANDLE;
  VkDevice device = VK_NULL_HANDLE;
  uint32_t queue_family = 0;
  VkQueue queue = VK_NULL_HANDLE;
  VmaAllocator allocator = VK_NULL_HANDLE;
  VkDescriptorPool desc_pool = VK_NULL_HANDLE;
  VkCommandPool cmd_pool = VK_NULL_HANDLE;
  VkDescriptorSetLayout global_ds_layout = VK_NULL_HANDLE;
  bool has_fp16 = false;
  bool has_int64 = false;
  bool has_shader_f16 = false;

  static Device& get();
  void init();
  void destroy();
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
  phys = phys_list[0];  // first device (matches torch backend behavior)

  VkPhysicalDeviceProperties props{};
  vkGetPhysicalDeviceProperties(phys, &props);
  has_fp16 = true;
  has_int64 = false;
  fprintf(stderr, "[vkssbo] maxWGCount=%u,%u,%u maxWGSize=%u,%u,%u\n",
          props.limits.maxComputeWorkGroupCount[0],
          props.limits.maxComputeWorkGroupCount[1],
          props.limits.maxComputeWorkGroupCount[2],
          props.limits.maxComputeWorkGroupSize[0],
          props.limits.maxComputeWorkGroupSize[1],
          props.limits.maxComputeWorkGroupSize[2]);

  // Detect shaderFloat16 support (VK_KHR_shader_float16_int8 / Vulkan 1.2+).
  has_shader_f16 = false;
  {
    uint32_t n_ext = 0;
    vkEnumerateDeviceExtensionProperties(phys, nullptr, &n_ext, nullptr);
    std::vector<VkExtensionProperties> exts(n_ext);
    vkEnumerateDeviceExtensionProperties(phys, nullptr, &n_ext, exts.data());
    bool has_ext = false;
    for (auto& e : exts) {
      if (strcmp(e.extensionName, VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME) == 0) {
        has_ext = true;
        break;
      }
    }
    if (has_ext) {
      VkPhysicalDeviceShaderFloat16Int8Features f16{};
      f16.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES;
      VkPhysicalDeviceFeatures2 f2{};
      f2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
      f2.pNext = &f16;
      vkGetPhysicalDeviceFeatures2(phys, &f2);
      has_shader_f16 = f16.shaderFloat16 == VK_TRUE;
    }
  }
  fprintf(stderr, "[vkssbo] shaderFloat16=%d\n", (int)has_shader_f16);

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

  const float prio = 1.0f;
  VkDeviceQueueCreateInfo dqci{};
  dqci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  dqci.queueFamilyIndex = queue_family;
  dqci.queueCount = 1;
  dqci.pQueuePriorities = &prio;

  VkPhysicalDeviceFeatures feats{};

  const char* dev_exts[] = {VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME};
  VkPhysicalDeviceShaderFloat16Int8Features f16_feat{};
  f16_feat.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES;
  f16_feat.shaderFloat16 = VK_TRUE;

  VkDeviceCreateInfo dci{};
  dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  dci.queueCreateInfoCount = 1;
  dci.pQueueCreateInfos = &dqci;
  dci.pEnabledFeatures = &feats;
  if (has_shader_f16) {
    dci.enabledExtensionCount = 1;
    dci.ppEnabledExtensionNames = dev_exts;
    dci.pNext = &f16_feat;
  }
  VKSSBO_CHECK(vkCreateDevice(phys, &dci, nullptr, &device));
  vkGetDeviceQueue(device, queue_family, 0, &queue);

  VmaAllocatorCreateInfo aci{};
  aci.vulkanApiVersion = VK_API_VERSION_1_0;
  aci.physicalDevice = phys;
  aci.device = device;
  aci.instance = instance;
  VKSSBO_CHECK(vmaCreateAllocator(&aci, &allocator));

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
struct Buffer {
  VmaAllocation allocation = VK_NULL_HANDLE;
  VkBuffer buffer = VK_NULL_HANDLE;
  VkDeviceSize size = 0;
  bool mapped = false;
  void* map_ptr = nullptr;

  Buffer() = default;
  Buffer(Buffer&& o) noexcept
      : allocation(o.allocation), buffer(o.buffer), size(o.size),
        mapped(o.mapped), map_ptr(o.map_ptr) {
    o.allocation = VK_NULL_HANDLE;
    o.buffer = VK_NULL_HANDLE;
    o.size = 0;
    o.mapped = false;
    o.map_ptr = nullptr;
  }
  Buffer& operator=(Buffer&& o) noexcept {
    if (this != &o) {
      destroy();
      allocation = o.allocation;
      buffer = o.buffer;
      size = o.size;
      mapped = o.mapped;
      map_ptr = o.map_ptr;
      o.allocation = VK_NULL_HANDLE;
      o.buffer = VK_NULL_HANDLE;
      o.size = 0;
      o.mapped = false;
      o.map_ptr = nullptr;
    }
    return *this;
  }
  ~Buffer() { destroy(); }
  void destroy() {
    auto& d = Device::get();
    if (buffer) {
      if (mapped) vmaUnmapMemory(d.allocator, allocation);
      vmaDestroyBuffer(d.allocator, buffer, allocation);
      buffer = VK_NULL_HANDLE;
      allocation = VK_NULL_HANDLE;
      size = 0;
      mapped = false;
    }
  }
};

Buffer create_buffer(VkDeviceSize size, VkBufferUsageFlags usage, VmaAllocationCreateFlags flags, VmaMemoryUsage mem) {
  Buffer b;
  VkBufferCreateInfo bci{};
  bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bci.size = size;
  bci.usage = usage;
  VmaAllocationCreateInfo aci{};
  aci.usage = mem;
  aci.flags = flags;
  VKSSBO_CHECK(vmaCreateBuffer(Device::get().allocator, &bci, &aci, &b.buffer, &b.allocation, nullptr));
  b.size = size;
  return b;
}

// Host-visible buffer for staging data in/out
Buffer create_host_buffer(VkDeviceSize size) {
  return create_buffer(size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                       VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT,
                       VMA_MEMORY_USAGE_AUTO);
}

// Device-local buffer for compute; on iGPU shared memory this is
// host-visible so we can map/write directly (no staging copies).
Buffer create_device_buffer(VkDeviceSize size) {
  return create_buffer(size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                       VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT,
                       VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
}

void* map_buffer(Buffer& b) {
  auto& d = Device::get();
  void* p = nullptr;
  VKSSBO_CHECK(vmaMapMemory(d.allocator, b.allocation, &p));
  b.mapped = true;
  b.map_ptr = p;
  return p;
}

void unmap_buffer(Buffer& b) {
  if (b.mapped) {
    vmaUnmapMemory(Device::get().allocator, b.allocation);
    b.mapped = false;
    b.map_ptr = nullptr;
  }
}

// ---------- Compute pipeline ----------
struct Pipeline {
  VkPipelineLayout layout = VK_NULL_HANDLE;
  VkPipeline pipeline = VK_NULL_HANDLE;
};

struct ComputeManager {
  std::mutex mutex;
  VkCommandBuffer cmd = VK_NULL_HANDLE;
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
    auto& d = Device::get();
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
  }

  void submit_and_wait() {
    auto& d = Device::get();
    VKSSBO_CHECK(vkEndCommandBuffer(cmd));
    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    VKSSBO_CHECK(vkQueueSubmit(d.queue, 1, &si, VK_NULL_HANDLE));
    VKSSBO_CHECK(vkQueueWaitIdle(d.queue));
  }
};

} // namespace vkssbo
