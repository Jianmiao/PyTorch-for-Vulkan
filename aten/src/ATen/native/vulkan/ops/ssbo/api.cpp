// vulkan_ssbo: C API exposing SSBO-based compute ops (compiled into torch).
#ifndef VKSSBO_EXPORT
#ifdef _WIN32
#define VKSSBO_EXPORT __declspec(dllexport)
#else
#define VKSSBO_EXPORT
#endif
#endif
// Standalone DLL owns the VMA implementation; inside torch, Allocator.cpp
// provides it (same header from third_party/VulkanMemoryAllocator).
#ifdef VKSSBO_STANDALONE
#define VMA_IMPLEMENTATION
#endif
#include <include/vk_mem_alloc.h>

#include "core.h"
#include "spv_embedded.h"
#include <cstdlib>
#include <array>
#include <vector>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <atomic>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <functional>
#include <chrono>
#include <thread>
#include <string>
#include <cmath>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#ifdef small
#undef small
#endif
#else
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#endif

using namespace vkssbo;

static std::vector<uint32_t> load_spv(const std::string& name) {
  auto& m = at::native::vulkan::ssbo::spv_map();
  auto it = m.find(name + ".spv");
  if (it == m.end()) throw Error("cannot find embedded shader: " + name + ".spv");
  const uint8_t* data = it->second.first;
  size_t sz = it->second.second;
  std::vector<uint32_t> spv(sz / 4);
  memcpy(spv.data(), data, sz);
  return spv;
}

struct Dispatch {
  Device& d;
  ComputeManager& cm;
  VkDescriptorSetLayout ds_layout;
  VkDescriptorSet desc_set;

  Dispatch(Device& dev, ComputeManager& mgr)
      : d(dev), cm(mgr), ds_layout(VK_NULL_HANDLE), desc_set(VK_NULL_HANDLE) {}

  void begin(int num_bindings) {
    (void)num_bindings;
    ds_layout = d.global_ds_layout;
    VkDescriptorSetAllocateInfo dsai{};
    dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsai.descriptorPool = d.desc_pool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts = &ds_layout;
    VKSSBO_CHECK(vkAllocateDescriptorSets(d.device, &dsai, &desc_set));
  }

  void bind(int binding, VkBuffer buf, VkDeviceSize size) {
    VkDescriptorBufferInfo dbi{};
    dbi.buffer = buf;
    dbi.offset = 0;
    dbi.range = size;
    VkWriteDescriptorSet wds{};
    wds.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    wds.dstSet = desc_set;
    wds.dstBinding = (uint32_t)binding;
    wds.descriptorCount = 1;
    wds.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    wds.pBufferInfo = &dbi;
    vkUpdateDescriptorSets(d.device, 1, &wds, 0, nullptr);
  }

  void run(const std::string& shader, const void* params, uint32_t params_size,
           uint32_t gx, uint32_t gy, uint32_t gz) {
    // Uploads to device-local allocations are recorded as copies in the same
    // command buffer. Make their writes visible before the first compute use.
    cm.barrier_transfer_to_compute();
    // Select fp16-compute variant when the device supports shaderFloat16.
    std::string name = d.has_shader_f16 ? shader + "_fp16" : shader + "_fp32";
    Pipeline& p = cm.get_pipeline_locked(name, load_spv(name));
    VkCommandBuffer cmd = cm.cmd;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p.pipeline);
    vkCmdPushConstants(cmd, p.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, params_size, params);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p.layout, 0, 1, &desc_set, 0, nullptr);
    vkCmdDispatch(cmd, gx, gy, gz);
    if (RuntimeProfile::enabled()) runtime_profile().record_dispatch();
  }

  void cleanup() {
    // Return the descriptor set to the pool for reuse.
    if (desc_set) {
      vkFreeDescriptorSets(d.device, d.desc_pool, 1, &desc_set);
      desc_set = VK_NULL_HANDLE;
    }
    ds_layout = VK_NULL_HANDLE;
  }
};

// Keep the resident reference alive for the entire command-buffer recording
// window so LRU eviction cannot invalidate an input descriptor.
struct InputBuffer {
  std::shared_ptr<Buffer> resident;
  Buffer owned;

  Buffer& get() { return resident ? *resident : owned; }
};

enum class MatmulVariant : uint32_t {
  Tile64 = 0,
  Tile128 = 1,
};

enum class Conv2dVariant : uint32_t {
  V4Acc = 0,
  C32 = 1,
};

struct MatmulTuneEntry {
  MatmulVariant variant = MatmulVariant::Tile64;
  double elapsed_ns = 0.0;
};

struct Conv2dTuneEntry {
  Conv2dVariant variant = Conv2dVariant::V4Acc;
  double elapsed_ns = 0.0;
};

template <typename Entry>
struct AutotuneCacheState {
  std::mutex mutex;
  std::unordered_map<std::string, Entry> entries;
  bool loaded = false;
  uint64_t cache_hits = 0;
  uint64_t cache_misses = 0;
  uint64_t tune_runs = 0;
  uint64_t tune_failures = 0;
  uint64_t benchmark_dispatches = 0;
};

using MatmulTuneState = AutotuneCacheState<MatmulTuneEntry>;
struct Conv2dTuneState : AutotuneCacheState<Conv2dTuneEntry> {
  std::unordered_set<std::string> failed_keys;
};

static MatmulTuneState& matmul_tune_state() {
  static MatmulTuneState state;
  return state;
}

static Conv2dTuneState& conv2d_tune_state() {
  static Conv2dTuneState state;
  return state;
}

static std::string autotune_cache_path(
    const char* explicit_path_environment, const char* filename) {
  const char* explicit_path = std::getenv(explicit_path_environment);
  if (explicit_path && explicit_path[0]) return explicit_path;
  const char* persist = std::getenv("VKSSBO_AUTOTUNE_PERSIST");
  if (persist && persist[0] == '0') return std::string();
#ifdef _WIN32
  const char* base = std::getenv("LOCALAPPDATA");
  if (!base || !base[0]) return std::string();
  return std::string(base) + "\\torch\\vulkan_ssbo\\" + filename;
#else
  const char* base = std::getenv("XDG_CACHE_HOME");
  if (base && base[0]) {
    return std::string(base) + "/torch/vulkan_ssbo/" + filename;
  }
  const char* home = std::getenv("HOME");
  if (!home || !home[0]) return std::string();
  return std::string(home) + "/.cache/torch/vulkan_ssbo/" + filename;
#endif
}

static std::string matmul_tune_cache_path() {
  return autotune_cache_path(
      "VKSSBO_AUTOTUNE_CACHE", "matmul_autotune_v1.txt");
}

static std::string conv2d_tune_cache_path() {
  return autotune_cache_path(
      "VKSSBO_CONV_AUTOTUNE_CACHE", "conv2d_autotune_v1.txt");
}

struct MatmulTuneCacheCodec {
  static bool read(
      std::istream& in, std::string& key, MatmulTuneEntry& entry,
      bool& valid) {
    int variant = 0;
    double elapsed_ns = 0.0;
    if (!(in >> key >> variant >> elapsed_ns)) return false;
    valid = (variant == 0 || variant == 1) && !(elapsed_ns < 0.0);
    if (valid) {
      entry = MatmulTuneEntry{
          static_cast<MatmulVariant>(variant), elapsed_ns};
    }
    return true;
  }

  static void write(
      std::ostream& out, const std::string& key,
      const MatmulTuneEntry& entry) {
    out << key << ' ' << static_cast<int>(entry.variant) << ' '
        << entry.elapsed_ns << '\n';
  }
};

struct Conv2dTuneCacheCodec {
  static bool read(
      std::istream& in, std::string& key, Conv2dTuneEntry& entry,
      bool& valid) {
    int variant = 0;
    double elapsed_ns = 0.0;
    if (!(in >> key >> variant >> elapsed_ns)) return false;
    valid = (variant == 0 || variant == 1) && !(elapsed_ns < 0.0);
    if (valid) {
      entry = Conv2dTuneEntry{
          static_cast<Conv2dVariant>(variant), elapsed_ns};
    }
    return true;
  }

  static void write(
      std::ostream& out, const std::string& key,
      const Conv2dTuneEntry& entry) {
    out << key << ' ' << static_cast<int>(entry.variant) << ' '
        << entry.elapsed_ns << '\n';
  }
};

constexpr size_t kAutotuneCacheMaxEntries = 128;

template <typename Entry, typename Codec>
static void load_autotune_cache_locked(
    AutotuneCacheState<Entry>& state, std::string (*cache_path)()) {
  if (state.loaded) return;
  state.loaded = true;
  const std::string path = cache_path();
  if (path.empty()) return;
  std::ifstream in(path);
  if (!in) return;
  std::string key;
  Entry entry;
  bool valid = false;
  while (Codec::read(in, key, entry, valid)) {
    if (!valid) continue;
    if (state.entries.size() >= kAutotuneCacheMaxEntries) break;
    state.entries.emplace(std::move(key), std::move(entry));
  }
}

static void load_matmul_tune_cache_locked(MatmulTuneState& state) {
  load_autotune_cache_locked<MatmulTuneEntry, MatmulTuneCacheCodec>(
      state, matmul_tune_cache_path);
}

static void load_conv2d_tune_cache_locked(Conv2dTuneState& state) {
  load_autotune_cache_locked<Conv2dTuneEntry, Conv2dTuneCacheCodec>(
      state, conv2d_tune_cache_path);
}

struct AutotuneCacheFileLock {
#ifdef _WIN32
  HANDLE handle = INVALID_HANDLE_VALUE;
  explicit AutotuneCacheFileLock(const std::filesystem::path& target) {
    std::filesystem::path lock_path = target;
    lock_path += ".lock";
    for (int attempt = 0; attempt < 100; ++attempt) {
      handle = CreateFileW(
          lock_path.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr,
          OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
      if (handle != INVALID_HANDLE_VALUE) break;
      const DWORD error = GetLastError();
      if (error != ERROR_SHARING_VIOLATION && error != ERROR_LOCK_VIOLATION) {
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }
  ~AutotuneCacheFileLock() {
    if (handle != INVALID_HANDLE_VALUE) CloseHandle(handle);
  }
  bool locked() const { return handle != INVALID_HANDLE_VALUE; }
#else
  int fd = -1;
  explicit AutotuneCacheFileLock(const std::filesystem::path& target) {
    const std::string lock_path = target.string() + ".lock";
    fd = open(lock_path.c_str(), O_CREAT | O_RDWR, 0600);
    if (fd >= 0 && flock(fd, LOCK_EX) != 0) {
      close(fd);
      fd = -1;
    }
  }
  ~AutotuneCacheFileLock() {
    if (fd >= 0) {
      flock(fd, LOCK_UN);
      close(fd);
    }
  }
  bool locked() const { return fd >= 0; }
#endif
  AutotuneCacheFileLock(const AutotuneCacheFileLock&) = delete;
  AutotuneCacheFileLock& operator=(const AutotuneCacheFileLock&) = delete;
};

template <typename Entry, typename Codec>
static void save_autotune_cache_locked(
    AutotuneCacheState<Entry>& state, std::string (*cache_path)()) {
  const std::string path = cache_path();
  if (path.empty()) return;
  try {
    const std::filesystem::path target(path);
    if (target.has_parent_path()) {
      std::error_code ec;
      std::filesystem::create_directories(target.parent_path(), ec);
      if (ec) return;
    }
    AutotuneCacheFileLock cache_lock(target);
    if (!cache_lock.locked()) return;
    // Merge entries written by another process since this process loaded the
    // cache, so an atomic replacement does not discard unrelated keys.
    {
      std::ifstream current(target);
      std::string key;
      Entry entry;
      bool valid = false;
      while (Codec::read(current, key, entry, valid)) {
        if (!valid) continue;
        if (state.entries.find(key) != state.entries.end()) continue;
        if (state.entries.size() >= kAutotuneCacheMaxEntries) break;
        state.entries.emplace(key, entry);
      }
    }
    std::filesystem::path temp = target;
    temp += ".tmp." + std::to_string(
        static_cast<unsigned long long>(
            std::hash<std::thread::id>{}(std::this_thread::get_id()))) +
        "." + std::to_string(static_cast<unsigned long long>(
            std::chrono::steady_clock::now().time_since_epoch().count()));
    std::ofstream out(temp, std::ios::trunc);
    if (!out) return;
    for (const auto& item : state.entries) {
      Codec::write(out, item.first, item.second);
    }
    out.flush();
    if (!out) {
      out.close();
      std::error_code ignored;
      std::filesystem::remove(temp, ignored);
      return;
    }
    out.close();
#ifdef _WIN32
    if (!MoveFileExW(temp.wstring().c_str(), target.wstring().c_str(),
                     MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
      std::error_code ignored;
      std::filesystem::remove(temp, ignored);
    }
#else
    std::error_code ec;
    std::filesystem::rename(temp, target, ec);
    if (ec) {
      std::error_code ignored;
      std::filesystem::remove(temp, ignored);
    }
#endif
  } catch (...) {
    // Autotuning must never make an otherwise valid Vulkan operation fail.
  }
}

static void save_matmul_tune_cache_locked(MatmulTuneState& state) {
  save_autotune_cache_locked<MatmulTuneEntry, MatmulTuneCacheCodec>(
      state, matmul_tune_cache_path);
}

static void save_conv2d_tune_cache_locked(Conv2dTuneState& state) {
  save_autotune_cache_locked<Conv2dTuneEntry, Conv2dTuneCacheCodec>(
      state, conv2d_tune_cache_path);
}

static uint32_t matmul_tune_bucket(int value, bool batch = false) {
  if (value <= 0) return 0;
  if (batch) return static_cast<uint32_t>(std::min(value, 16));
  const uint32_t v = static_cast<uint32_t>(value);
  const uint32_t quantum = v <= 128 ? 8u : 64u;
  if (v > 4096u) return ((v + 255u) / 256u) * 256u;
  return ((v + quantum - 1u) / quantum) * quantum;
}

static std::string matmul_tune_key(
    const Device& d, int B, int M, int K, int N, int transB, bool has_bias,
    int activation = 0) {
  std::ostringstream key;
  const uint64_t budget_bucket =
      static_cast<uint64_t>(d.device_local_budget) / (256ull * 1024ull * 1024ull);
  key << "v4:" << d.vendor_id << ':' << d.device_id << ':' << d.driver_version
      << ':' << static_cast<uint32_t>(d.memory_topology) << ':'
      << (d.has_shader_f16 ? 1 : 0) << ':' << d.max_shared_memory << ':'
      << d.max_workgroup_invocations << ':' << budget_bucket << ':'
      << matmul_tune_bucket(B, true) << ':' << matmul_tune_bucket(M) << ':'
      << matmul_tune_bucket(K) << ':' << matmul_tune_bucket(N) << ':'
      << (N > 4096 ? N : 0) << ':'
      << transB << ':' << (has_bias ? 1 : 0) << ':' << activation;
  return key.str();
}

static bool matmul_variant_override(MatmulVariant* variant) {
  const char* env = std::getenv("VKSSBO_MATMUL_128");
  if (!env || !env[0] || !variant) return false;
  if (env[0] == '1') {
    *variant = MatmulVariant::Tile128;
    return true;
  }
  if (env[0] == '0') {
    *variant = MatmulVariant::Tile64;
    return true;
  }
  return false;
}

static bool matmul_autotune_enabled() {
  const char* env = std::getenv("VKSSBO_AUTOTUNE");
  if (!env) env = std::getenv("VKSSBO_MATMUL_AUTOTUNE");
  return !env || env[0] != '0';
}

static bool matmul_should_autotune(int B, int M, int K, int N) {
  const char* small = std::getenv("VKSSBO_AUTOTUNE_SMALL");
  if (small && small[0] == '1') return true;
  if (B < 1 || M < 64 || K < 64 || N < 64) return false;
  const uint64_t work = static_cast<uint64_t>(B) * static_cast<uint64_t>(M) *
      static_cast<uint64_t>(K) * static_cast<uint64_t>(N);
  // A/B is only worthwhile once the kernel has enough work to amortize the
  // extra candidate. Small attention projections keep the portable default.
  return work >= 32ull * 1024ull * 1024ull;
}

static bool matmul_cached_variant(
    const std::string& key, MatmulVariant* variant, double* elapsed_ns) {
  auto& state = matmul_tune_state();
  std::lock_guard<std::mutex> lock(state.mutex);
  load_matmul_tune_cache_locked(state);
  auto it = state.entries.find(key);
  if (it == state.entries.end()) {
    ++state.cache_misses;
    return false;
  }
  ++state.cache_hits;
  if (variant) *variant = it->second.variant;
  if (elapsed_ns) *elapsed_ns = it->second.elapsed_ns;
  return true;
}

static void remember_matmul_variant(
    const std::string& key, MatmulVariant variant, double elapsed_ns) {
  auto& state = matmul_tune_state();
  std::lock_guard<std::mutex> lock(state.mutex);
  load_matmul_tune_cache_locked(state);
  if (state.entries.size() >= kAutotuneCacheMaxEntries &&
      state.entries.find(key) == state.entries.end()) {
    state.entries.erase(state.entries.begin());
  }
  state.entries[key] = MatmulTuneEntry{variant, elapsed_ns};
  ++state.tune_runs;
  save_matmul_tune_cache_locked(state);
}

static void note_matmul_benchmark_dispatch() {
  auto& state = matmul_tune_state();
  std::lock_guard<std::mutex> lock(state.mutex);
  ++state.benchmark_dispatches;
}

static void record_matmul_dispatch(
    Dispatch& disp, InputBuffer& da, Buffer& db, Buffer& dout, Buffer& dbias,
    int B, int M, int K, int N, int transB, bool has_bias,
    int activation, MatmulVariant variant, float scale = 1.0f) {
  disp.begin(4);
  disp.bind(0, da.get().buffer, da.get().size);
  disp.bind(1, db.buffer, db.size);
  disp.bind(2, dout.buffer, dout.size);
  disp.bind(3, dbias.buffer, dbias.size);
  const uint32_t tile_m = variant == MatmulVariant::Tile128 ? 128u : 64u;
  const uint32_t m_tiles = (static_cast<uint32_t>(M) + tile_m - 1u) / tile_m;
  const uint32_t total_tiles = static_cast<uint32_t>(B) * m_tiles;
  const uint32_t tiles_y = std::min(total_tiles, 65535u);
  struct P {
    int B, M, K, N, transB, hasBias, tiles_y;
    float scale;
    int activation;
  };
  P p{B, M, K, N, transB, has_bias ? 1 : 0, static_cast<int>(tiles_y),
      scale, activation};
  disp.run(variant == MatmulVariant::Tile128 ? "matmul_128x64" : "matmul",
           &p, sizeof(p), (static_cast<uint32_t>(N) + 63u) / 64u, tiles_y,
           (total_tiles + tiles_y - 1u) / tiles_y);
}

struct MatmulTuneMeasurement {
  MatmulVariant variant = MatmulVariant::Tile64;
  double elapsed_ns = 0.0;
  bool valid = false;
  uint32_t valid_samples = 0;
};

static double median_three(std::array<double, 3> values) {
  std::sort(values.begin(), values.end());
  return values[1];
}

static MatmulTuneMeasurement tune_matmul_dispatches(
    Device& d, ComputeManager& cm, InputBuffer& da, Buffer& db, Buffer& out64,
    Buffer& out128, Buffer& dbias, int B, int M, int K, int N, int transB,
    bool has_bias, int activation, MatmulTuneMeasurement* second) {
  MatmulTuneMeasurement first;
  if (!second) return first;
  std::vector<std::unique_ptr<Dispatch>> recorded_dispatches;
  recorded_dispatches.reserve(6);
  constexpr uint32_t kRounds = 3;
  constexpr uint32_t kQueriesPerRound = 4;
  std::array<double, kRounds> samples64{};
  std::array<double, kRounds> samples128{};
  std::array<bool, kRounds> valid64{};
  std::array<bool, kRounds> valid128{};
  cm.barrier_transfer_to_compute();
  cm.reset_timestamps();
  auto record_candidate = [&](MatmulVariant variant) {
    recorded_dispatches.emplace_back(std::make_unique<Dispatch>(d, cm));
    record_matmul_dispatch(
        *recorded_dispatches.back(), da, db,
        variant == MatmulVariant::Tile128 ? out128 : out64, dbias,
        B, M, K, N, transB, has_bias, activation, variant);
    note_matmul_benchmark_dispatch();
  };
  for (uint32_t round = 0; round < kRounds; ++round) {
    const uint32_t base = round * kQueriesPerRound;
    const bool first64 = (round != 1);
    cm.write_timestamp(base);
    record_candidate(
        first64 ? MatmulVariant::Tile64 : MatmulVariant::Tile128);
    cm.write_timestamp(base + 1);
    cm.barrier_compute_to_compute();
    cm.write_timestamp(base + 2);
    record_candidate(
        first64 ? MatmulVariant::Tile128 : MatmulVariant::Tile64);
    cm.write_timestamp(base + 3);
    if (round + 1 < kRounds) cm.barrier_compute_to_compute();
  }
  cm.submit_and_wait();
  first.variant = MatmulVariant::Tile64;
  for (uint32_t round = 0; round < kRounds; ++round) {
    const uint32_t base = round * kQueriesPerRound;
    const bool first64 = (round != 1);
    double first_ns = 0.0;
    double second_ns = 0.0;
    const bool first_valid = cm.read_timestamp_pair(base, base + 1, &first_ns);
    const bool second_valid =
        cm.read_timestamp_pair(base + 2, base + 3, &second_ns);
    if (first64) {
      valid64[round] = first_valid;
      valid128[round] = second_valid;
      samples64[round] = first_ns;
      samples128[round] = second_ns;
    } else {
      valid128[round] = first_valid;
      valid64[round] = second_valid;
      samples128[round] = first_ns;
      samples64[round] = second_ns;
    }
  }
  first.valid_samples = static_cast<uint32_t>(
      std::count(valid64.begin(), valid64.end(), true));
  first.valid = first.valid_samples == kRounds;
  if (first.valid) first.elapsed_ns = median_three(samples64);
  second->variant = MatmulVariant::Tile128;
  second->valid_samples = static_cast<uint32_t>(
      std::count(valid128.begin(), valid128.end(), true));
  second->valid = second->valid_samples == kRounds;
  if (second->valid) second->elapsed_ns = median_three(samples128);
  for (auto& dispatch : recorded_dispatches) dispatch->cleanup();
  return first;
}

static void note_matmul_tune_failure() {
  auto& state = matmul_tune_state();
  std::lock_guard<std::mutex> lock(state.mutex);
  ++state.tune_failures;
}

static void matmul_autotune_stats_impl(
    size_t* entries, uint64_t* cache_hits, uint64_t* cache_misses,
    uint64_t* tune_runs, uint64_t* tune_failures,
    uint64_t* benchmark_dispatches) {
  auto& state = matmul_tune_state();
  std::lock_guard<std::mutex> lock(state.mutex);
  load_matmul_tune_cache_locked(state);
  if (entries) *entries = state.entries.size();
  if (cache_hits) *cache_hits = state.cache_hits;
  if (cache_misses) *cache_misses = state.cache_misses;
  if (tune_runs) *tune_runs = state.tune_runs;
  if (tune_failures) *tune_failures = state.tune_failures;
  if (benchmark_dispatches) *benchmark_dispatches = state.benchmark_dispatches;
}

static bool matmul_tune_memory_ok(const Device& d, VkDeviceSize output_bytes) {
  if (output_bytes == 0 ||
      output_bytes > (std::numeric_limits<VkDeviceSize>::max() / 2)) {
    return false;
  }
  const Device::MemoryBudget live = d.query_memory_budget();
  const VkDeviceSize budget = live.budget ? live.budget : d.device_local_budget;
  const VkDeviceSize usage = live.budget ? live.usage : d.device_local_usage;
  if (budget == 0) return true;
  if (usage >= budget) return false;
  const VkDeviceSize required = output_bytes * 2;
  const VkDeviceSize reserve = budget / 10;
  return required <= budget - usage &&
      required <= budget - usage - std::min(reserve, budget - usage);
}

struct Conv2dTiledParams {
  int B, C, O, H, W, Ho, Wo, KH, KW, SH, SW, PH, PW, G;
};

static bool conv2d_has_explicit_variant_override() {
  constexpr std::array<const char*, 4> names{{
      "VKSSBO_CONV_X32", "VKSSBO_CONV_C32", "VKSSBO_CONV_V4",
      "VKSSBO_CONV_V4ACC"}};
  for (const char* name : names) {
    const char* value = std::getenv(name);
    if (value && value[0]) return true;
  }
  return false;
}

static bool conv2d_autotune_enabled() {
  const char* global = std::getenv("VKSSBO_AUTOTUNE");
  const char* conv = std::getenv("VKSSBO_CONV_AUTOTUNE");
  return (!global || global[0] != '0') && (!conv || conv[0] != '0');
}

static bool conv2d_autotune_eligible(
    const Device& d, int B, int C, int O, int H, int W, int Ho, int Wo,
    int KH, int KW, int SH, int SW, int DH, int DW, int PH, int PW, int G) {
  return d.has_shader_f16 && d.max_workgroup_invocations >= 256u &&
      d.max_workgroup_size_x >= 16u && d.max_workgroup_size_y >= 16u &&
      d.max_shared_memory >= 40u * 1024u && B > 0 && C > 0 && O >= 32 &&
      H > 0 && W > 0 && Ho > 0 && Wo > 0 && KH == 3 && KW == 3 &&
      SH == 1 && SW == 1 && DH == 1 && DW == 1 && G == 1 &&
      PH <= 1 && PW <= 1 && PH >= 0 && PW >= 0 &&
      (C % 16) == 0 && (C % 32) == 0;
}

static std::string conv2d_tune_key(
    const Device& d, int B, int C, int O, int H, int W, int Ho, int Wo,
    int KH, int KW, int SH, int SW, int DH, int DW, int PH, int PW, int G,
    bool has_bias) {
  std::ostringstream key;
  const uint64_t budget_bucket =
      static_cast<uint64_t>(d.device_local_budget) /
      (256ull * 1024ull * 1024ull);
  key << "conv2d-v1:" << d.vendor_id << ':' << d.device_id << ':'
      << d.driver_version << ':' << static_cast<uint32_t>(d.memory_topology)
      << ':' << (d.has_shader_f16 ? 1 : 0) << ':' << d.max_shared_memory
      << ':' << d.max_workgroup_invocations << ':'
      << d.max_workgroup_size_x << ':' << d.max_workgroup_size_y << ':'
      << d.max_workgroup_size_z << ':' << budget_bucket << ':' << B << ':' << C
      << ':' << O << ':' << H << ':' << W << ':' << Ho << ':' << Wo << ':'
      << KH << ':' << KW << ':' << SH << ':' << SW << ':' << DH << ':' << DW
      << ':' << PH << ':' << PW << ':' << G << ':' << (has_bias ? 1 : 0);
  return key.str();
}

static bool conv2d_cached_variant(
    const std::string& key, Conv2dVariant* variant, double* elapsed_ns) {
  auto& state = conv2d_tune_state();
  std::lock_guard<std::mutex> lock(state.mutex);
  load_conv2d_tune_cache_locked(state);
  auto it = state.entries.find(key);
  if (it == state.entries.end()) {
    ++state.cache_misses;
    return false;
  }
  ++state.cache_hits;
  if (variant) *variant = it->second.variant;
  if (elapsed_ns) *elapsed_ns = it->second.elapsed_ns;
  return true;
}

static bool conv2d_tuning_suppressed(const std::string& key) {
  auto& state = conv2d_tune_state();
  std::lock_guard<std::mutex> lock(state.mutex);
  return state.failed_keys.find(key) != state.failed_keys.end();
}

static void remember_conv2d_variant(
    const std::string& key, Conv2dVariant variant, double elapsed_ns) {
  auto& state = conv2d_tune_state();
  std::lock_guard<std::mutex> lock(state.mutex);
  load_conv2d_tune_cache_locked(state);
  if (state.entries.size() >= kAutotuneCacheMaxEntries &&
      state.entries.find(key) == state.entries.end()) {
    state.entries.erase(state.entries.begin());
  }
  state.entries[key] = Conv2dTuneEntry{variant, elapsed_ns};
  ++state.tune_runs;
  save_conv2d_tune_cache_locked(state);
}

static void note_conv2d_benchmark_dispatches(uint64_t count) {
  if (count == 0) return;
  auto& state = conv2d_tune_state();
  std::lock_guard<std::mutex> lock(state.mutex);
  state.benchmark_dispatches += count;
}

static void note_conv2d_tune_failure(const std::string& key) {
  auto& state = conv2d_tune_state();
  std::lock_guard<std::mutex> lock(state.mutex);
  if (!key.empty()) {
    if (state.failed_keys.size() >= kAutotuneCacheMaxEntries &&
        state.failed_keys.find(key) == state.failed_keys.end()) {
      state.failed_keys.erase(state.failed_keys.begin());
    }
    state.failed_keys.insert(key);
  }
  ++state.tune_failures;
}

static void conv2d_autotune_stats_impl(
    size_t* entries, uint64_t* cache_hits, uint64_t* cache_misses,
    uint64_t* tune_runs, uint64_t* tune_failures,
    uint64_t* benchmark_dispatches) {
  auto& state = conv2d_tune_state();
  std::lock_guard<std::mutex> lock(state.mutex);
  load_conv2d_tune_cache_locked(state);
  if (entries) *entries = state.entries.size();
  if (cache_hits) *cache_hits = state.cache_hits;
  if (cache_misses) *cache_misses = state.cache_misses;
  if (tune_runs) *tune_runs = state.tune_runs;
  if (tune_failures) *tune_failures = state.tune_failures;
  if (benchmark_dispatches) *benchmark_dispatches = state.benchmark_dispatches;
}

static bool conv2d_probe_output_bytes(
    int B, int O, int Ho, int Wo, VkDeviceSize* output_bytes) {
  if (!output_bytes || B <= 0 || O <= 0 || Ho <= 0 || Wo <= 0) return false;
  VkDeviceSize value = 2;
  constexpr VkDeviceSize limit = std::numeric_limits<VkDeviceSize>::max();
  for (int dimension : {B, O, Ho, Wo}) {
    const VkDeviceSize dim = static_cast<VkDeviceSize>(dimension);
    if (value > limit / dim) return false;
    value *= dim;
  }
  *output_bytes = value;
  return true;
}

static bool conv2d_tune_memory_ok(
    const Device& d, VkDeviceSize probe_output_bytes) {
  return matmul_tune_memory_ok(d, probe_output_bytes);
}

static void record_conv2d_variant_dispatch(
    Dispatch& disp, InputBuffer& dx, Buffer& dw, Buffer& db, Buffer& output,
    const Conv2dTiledParams& params, Conv2dVariant variant) {
  disp.begin(4);
  disp.bind(0, dx.get().buffer, dx.get().size);
  disp.bind(1, dw.buffer, dw.size);
  disp.bind(2, db.buffer, db.size);
  disp.bind(3, output.buffer, output.size);
  disp.run(
      variant == Conv2dVariant::C32 ? "conv2d_x32c32"
                                    : "conv2d_x32v4acc",
      &params, sizeof(params),
      (static_cast<uint32_t>(params.Wo) + 15u) / 16u,
      (static_cast<uint32_t>(params.Ho) + 15u) / 16u,
      static_cast<uint32_t>(params.B) *
          ((static_cast<uint32_t>(params.O) + 31u) / 32u));
}

struct Conv2dTuneMeasurement {
  Conv2dVariant variant = Conv2dVariant::V4Acc;
  double elapsed_ns = 0.0;
  bool valid = false;
  uint32_t valid_samples = 0;
};

static Conv2dTuneMeasurement tune_conv2d_dispatches(
    Device& d, ComputeManager& cm, InputBuffer& dx, Buffer& dw, Buffer& db,
    Buffer& out_v4acc, Buffer& out_c32, const Conv2dTiledParams& probe_params,
    Conv2dTuneMeasurement* c32_measurement) {
  Conv2dTuneMeasurement v4acc_measurement;
  if (!c32_measurement) return v4acc_measurement;
  constexpr uint32_t kRounds = 3;
  constexpr uint32_t kQueriesPerRound = 4;
  std::array<double, kRounds> samples_v4acc{};
  std::array<double, kRounds> samples_c32{};
  std::array<bool, kRounds> valid_v4acc{};
  std::array<bool, kRounds> valid_c32{};
  std::vector<std::unique_ptr<Dispatch>> recorded_dispatches;
  recorded_dispatches.reserve(6);
  uint64_t recorded_count = 0;
  auto cleanup_dispatches = [&]() {
    for (auto& dispatch : recorded_dispatches) dispatch->cleanup();
  };
  auto record_candidate = [&](Conv2dVariant variant) {
    recorded_dispatches.emplace_back(std::make_unique<Dispatch>(d, cm));
    record_conv2d_variant_dispatch(
        *recorded_dispatches.back(), dx, dw, db,
        variant == Conv2dVariant::C32 ? out_c32 : out_v4acc, probe_params,
        variant);
    ++recorded_count;
  };

  try {
    // Uploads share this command buffer with the probes. They must be visible
    // before query 0 so transfer time is excluded from both candidates.
    cm.barrier_transfer_to_compute();
    cm.reset_timestamps();
    for (uint32_t round = 0; round < kRounds; ++round) {
      const uint32_t base = round * kQueriesPerRound;
      const bool c32_first = round != 1;
      cm.write_timestamp(base);
      record_candidate(
          c32_first ? Conv2dVariant::C32 : Conv2dVariant::V4Acc);
      cm.write_timestamp(base + 1);
      cm.barrier_compute_to_compute();
      cm.write_timestamp(base + 2);
      record_candidate(
          c32_first ? Conv2dVariant::V4Acc : Conv2dVariant::C32);
      cm.write_timestamp(base + 3);
      if (round + 1 < kRounds) cm.barrier_compute_to_compute();
    }
    cm.submit_and_wait();
  } catch (...) {
    cleanup_dispatches();
    throw;
  }

  for (uint32_t round = 0; round < kRounds; ++round) {
    const uint32_t base = round * kQueriesPerRound;
    const bool c32_first = round != 1;
    double first_ns = 0.0;
    double second_ns = 0.0;
    const bool first_valid = cm.read_timestamp_pair(base, base + 1, &first_ns);
    const bool second_valid =
        cm.read_timestamp_pair(base + 2, base + 3, &second_ns);
    if (c32_first) {
      valid_c32[round] = first_valid;
      valid_v4acc[round] = second_valid;
      samples_c32[round] = first_ns;
      samples_v4acc[round] = second_ns;
    } else {
      valid_v4acc[round] = first_valid;
      valid_c32[round] = second_valid;
      samples_v4acc[round] = first_ns;
      samples_c32[round] = second_ns;
    }
  }
  v4acc_measurement.variant = Conv2dVariant::V4Acc;
  v4acc_measurement.valid_samples = static_cast<uint32_t>(
      std::count(valid_v4acc.begin(), valid_v4acc.end(), true));
  v4acc_measurement.valid = v4acc_measurement.valid_samples == kRounds;
  if (v4acc_measurement.valid) {
    v4acc_measurement.elapsed_ns = median_three(samples_v4acc);
  }
  c32_measurement->variant = Conv2dVariant::C32;
  c32_measurement->valid_samples = static_cast<uint32_t>(
      std::count(valid_c32.begin(), valid_c32.end(), true));
  c32_measurement->valid = c32_measurement->valid_samples == kRounds;
  if (c32_measurement->valid) {
    c32_measurement->elapsed_ns = median_three(samples_c32);
  }
  if (v4acc_measurement.valid && c32_measurement->valid) {
    note_conv2d_benchmark_dispatches(recorded_count);
  }
  cleanup_dispatches();
  return v4acc_measurement;
}

// All SSBOs store float16 (bandwidth halved). Data crosses as fp32 from
// Python; convert here.
static void upload_direct(const float* data, size_t numel, Buffer& dev) {
  const bool profile_enabled = RuntimeProfile::enabled();
  const auto profile_started = profile_enabled
      ? std::chrono::steady_clock::now()
      : std::chrono::steady_clock::time_point{};
  std::vector<uint16_t> halfs(numel);
  for (size_t i = 0; i < numel; i++) {
    float f = data[i];
    uint32_t bits;
    memcpy(&bits, &f, 4);
    // IEEE half conversion with round-to-nearest-even.
    uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t exp = (int32_t)((bits >> 23) & 0xff) - 127 + 15;
    uint32_t mant = bits & 0x7fffffu;
    uint16_t h;
    if (exp >= 31) {
      h = (uint16_t)(sign | 0x7c00u);
    } else if (exp <= 0) {
      if (exp < -10) {
        h = (uint16_t)sign;
      } else {
        mant |= 0x800000u;
        uint32_t shift = (uint32_t)(14 - exp);
        uint32_t m = (mant >> shift) & 0x3ffu;
        uint32_t rem = mant & ((1u << shift) - 1);
        uint32_t half = 1u << (shift - 1);
        if (rem > half || (rem == half && (m & 1u))) m++;
        h = (uint16_t)(sign | m);
      }
    } else {
      uint32_t m = (mant >> 13) & 0x3ffu;
      uint32_t rem = mant & 0x1fffu;
      if (rem > 0x1000u || (rem == 0x1000u && (m & 1u))) {
        m++;
        if (m == 0x400u) {
          m = 0;
          exp++;
          if (exp >= 31) {
            h = (uint16_t)(sign | 0x7c00u);
            halfs[i] = h;
            continue;
          }
        }
      }
      h = (uint16_t)(sign | ((uint32_t)exp << 10) | m);
    }
    halfs[i] = h;
  }
  const VkDeviceSize bytes = static_cast<VkDeviceSize>(numel * sizeof(uint16_t));
  if (dev.host_visible) {
    void* p = map_buffer(dev);
    memcpy(p, halfs.data(), static_cast<size_t>(bytes));
    flush_buffer(dev, 0, bytes);
    unmap_buffer(dev);
    if (profile_enabled) {
      runtime_profile().record_upload(
          runtime_profile_elapsed_ns(profile_started), bytes);
    }
    return;
  }

  // Discrete device-local allocations cannot be mapped. Stage the upload in
  // host-visible memory and keep the staging allocation alive until the
  // command buffer has completed.
  auto& cm = ComputeManager::get();
  Buffer staging = create_host_buffer(bytes);
  if (!staging.host_visible) {
    throw Error("Vulkan staging allocation is not host visible");
  }
  void* p = map_buffer(staging);
  memcpy(p, halfs.data(), static_cast<size_t>(bytes));
  flush_buffer(staging, 0, bytes);
  unmap_buffer(staging);
  VkBufferCopy region{};
  region.size = bytes;
  const bool use_async = Device::get().async_transfer_for(bytes);
  if (use_async) {
    cm.begin_transfer_if_needed();
    vkCmdCopyBuffer(cm.transfer_cmd, staging.buffer, dev.buffer, 1, &region);
  } else {
    vkCmdCopyBuffer(cm.cmd, staging.buffer, dev.buffer, 1, &region);
  }
  cm.retain_staging(std::move(staging));
  cm.record_staging_upload(bytes);
  cm.mark_transfer_pending(use_async);
  if (profile_enabled) {
    runtime_profile().record_upload(
        runtime_profile_elapsed_ns(profile_started), bytes);
  }
}

static Buffer* prepare_weight(uint64_t key, const float* data, size_t numel,
                              Buffer& owned) {
  const VkDeviceSize bytes = static_cast<VkDeviceSize>(numel * sizeof(uint16_t));
  if (key != 0 && WeightBufferCache::enabled()) {
    if (Buffer* cached = weight_cache().find(key, bytes)) return cached;
    Buffer candidate = create_weight_buffer(bytes);
    upload_direct(data, numel, candidate);
    if (Buffer* cached = weight_cache().insert(key, std::move(candidate))) {
      return cached;
    }
    owned = std::move(candidate);
    return &owned;
  }
  owned = create_device_buffer(bytes);
  upload_direct(data, numel, owned);
  return &owned;
}

static float* download_direct(Buffer& dev, size_t numel) {
  const bool profile_enabled = RuntimeProfile::enabled();
  const auto profile_started = profile_enabled
      ? std::chrono::steady_clock::now()
      : std::chrono::steady_clock::time_point{};
  float* out = (float*)malloc(numel * sizeof(float));
  const VkDeviceSize bytes = static_cast<VkDeviceSize>(numel * sizeof(uint16_t));
  Buffer staging;
  Buffer* source = &dev;
  if (!dev.host_visible) {
    staging = create_host_buffer(bytes);
    if (!staging.host_visible) {
      free(out);
      throw Error("Vulkan staging allocation is not host visible");
    }
    ComputeManager::get().copy_buffer_and_wait(dev.buffer, staging.buffer, bytes);
    source = &staging;
  }
  void* p = map_buffer(*source);
  invalidate_buffer(*source, 0, bytes);
  const uint16_t* halfs = (const uint16_t*)p;
  for (size_t i = 0; i < numel; i++) {
    uint16_t h = halfs[i];
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x3ffu;
    uint32_t bits;
    if (exp == 0) {
      if (mant == 0) {
        bits = sign;
      } else {
        int32_t e = -14;
        uint32_t m = mant;
        while (!(m & 0x400u)) { m <<= 1; e--; }
        m &= 0x3ffu;
        bits = sign | ((uint32_t)(e + 127) << 23) | (m << 13);
      }
    } else if (exp == 31) {
      bits = sign | 0x7f800000u | (mant << 13);
    } else {
      bits = sign | ((exp + 112u) << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &bits, 4);
    out[i] = f;
  }
  unmap_buffer(*source);
  if (profile_enabled) {
    runtime_profile().record_download(
        runtime_profile_elapsed_ns(profile_started), bytes);
  }
  return out;
}

static InputBuffer prepare_input(const float* host, size_t numel) {
  InputBuffer input;
  const VkDeviceSize bytes = static_cast<VkDeviceSize>(numel * sizeof(uint16_t));
  if (auto cached = resident_cache().find(host, bytes)) {
    input.resident = std::move(cached);
    return input;
  }
  input.owned = create_device_buffer(bytes);
  upload_direct(host, numel, input.owned);
  return input;
}

// Outputs are still materialized as CPU mirrors for the stock Vulkan bridge,
// but retain their SSBO allocation so a following SSBO operator can bind it
// without another upload. The cache is bounded and disabled with
// VKSSBO_RESIDENT_CACHE=0.
static float* download_resident(Buffer& dev, size_t numel) {
  float* out = download_direct(dev, numel);
  if (ResidentBufferCache::enabled()) {
    dev.recyclable = false;
    auto resident = std::make_shared<Buffer>(std::move(dev));
    resident_cache().insert(out, std::move(resident));
  }
  return out;
}

static float* download_f32_direct(Buffer& dev, size_t numel) {
  const bool profile_enabled = RuntimeProfile::enabled();
  const auto profile_started = profile_enabled
      ? std::chrono::steady_clock::now()
      : std::chrono::steady_clock::time_point{};
  float* out = (float*)malloc(numel * sizeof(float));
  const VkDeviceSize bytes = static_cast<VkDeviceSize>(numel * sizeof(float));
  Buffer staging;
  Buffer* source = &dev;
  if (!dev.host_visible) {
    staging = create_host_buffer(bytes);
    if (!staging.host_visible) {
      free(out);
      throw Error("Vulkan staging allocation is not host visible");
    }
    ComputeManager::get().copy_buffer_and_wait(dev.buffer, staging.buffer, bytes);
    source = &staging;
  }
  void* p = map_buffer(*source);
  invalidate_buffer(*source, 0, bytes);
  memcpy(out, p, static_cast<size_t>(bytes));
  unmap_buffer(*source);
  if (profile_enabled) {
    runtime_profile().record_download(
        runtime_profile_elapsed_ns(profile_started), bytes);
  }
  return out;
}

extern "C" {

VKSSBO_EXPORT void vkssbo_init(const char* spv_dir) {
  (void)spv_dir;  // SPIR-V is embedded; nothing to configure.
  Device::get();
}

VKSSBO_EXPORT void vkssbo_profile_reset() {
  runtime_profile().reset();
}

VKSSBO_EXPORT void vkssbo_profile_stats(
    uint32_t* schema_version, uint32_t* enabled,
    uint64_t* upload_host_ns, uint64_t* upload_bytes,
    uint64_t* upload_calls,
    uint64_t* submit_wait_ns, uint64_t* compute_submissions,
    uint64_t* transfer_submissions, uint64_t* copy_submissions,
    uint64_t* fence_waits, uint64_t* queue_wait_idle_count,
    uint64_t* gpu_command_ns, uint64_t* gpu_timestamp_submissions,
    uint64_t* gpu_timestamp_failures, uint64_t* dispatches,
    uint64_t* download_host_ns, uint64_t* download_transfer_wait_ns,
    uint64_t* download_bytes, uint64_t* download_submissions,
    uint64_t* download_calls) {
  const RuntimeProfileSnapshot stats = runtime_profile().read();
  if (schema_version) *schema_version = static_cast<uint32_t>(stats.schema_version);
  if (enabled) *enabled = RuntimeProfile::enabled() ? 1u : 0u;
  if (upload_host_ns) *upload_host_ns = stats.upload_host_ns;
  if (upload_bytes) *upload_bytes = stats.upload_bytes;
  if (upload_calls) *upload_calls = stats.upload_calls;
  if (submit_wait_ns) *submit_wait_ns = stats.submit_wait_ns;
  if (compute_submissions) *compute_submissions = stats.compute_submissions;
  if (transfer_submissions) {
    *transfer_submissions = stats.transfer_submissions;
  }
  if (copy_submissions) *copy_submissions = stats.copy_submissions;
  if (fence_waits) *fence_waits = stats.fence_waits;
  if (queue_wait_idle_count) {
    *queue_wait_idle_count = stats.queue_wait_idle_count;
  }
  if (gpu_command_ns) *gpu_command_ns = stats.gpu_command_ns;
  if (gpu_timestamp_submissions) {
    *gpu_timestamp_submissions = stats.gpu_timestamp_submissions;
  }
  if (gpu_timestamp_failures) {
    *gpu_timestamp_failures = stats.gpu_timestamp_failures;
  }
  if (dispatches) *dispatches = stats.dispatches;
  if (download_host_ns) *download_host_ns = stats.download_host_ns;
  if (download_transfer_wait_ns) {
    *download_transfer_wait_ns = stats.download_transfer_wait_ns;
  }
  if (download_bytes) *download_bytes = stats.download_bytes;
  if (download_submissions) {
    *download_submissions = stats.download_submissions;
  }
  if (download_calls) *download_calls = stats.download_calls;
}

VKSSBO_EXPORT void vkssbo_scheduler_stats(
    uint64_t* last_submitted, uint64_t* completed, uint64_t* in_flight,
    uint64_t* peak_in_flight, uint64_t* submits, uint64_t* polls,
    uint64_t* waits) {
  ComputeManager::get().scheduler_stats(
      last_submitted, completed, in_flight, peak_in_flight, submits, polls,
      waits);
}

VKSSBO_EXPORT void vkssbo_device_info(
    uint32_t* vendor_id, uint32_t* device_id, uint32_t* max_shared_memory,
    uint32_t* max_workgroup_invocations, uint32_t* shader_int8_supported,
    uint32_t* storage_buffer_8bit_supported,
    uint32_t* uniform_and_storage_buffer_8bit_supported,
    uint32_t* storage_push_constant_8bit_supported,
    uint32_t* memory_topology,
    uint32_t* has_device_local_only, uint32_t* has_host_visible_device_local,
    uint32_t* has_host_visible_system, uint64_t* device_local_budget,
    uint64_t* device_local_usage, uint64_t* device_local_only_budget,
    uint64_t* device_local_only_usage,
    uint64_t* host_visible_device_local_budget,
    uint64_t* host_visible_device_local_usage,
    uint64_t* host_visible_system_budget,
    uint64_t* host_visible_system_usage, char* name, size_t name_capacity,
    uint32_t* driver_version, uint32_t* timestamp_valid_bits,
    float* timestamp_period_ns, uint32_t* compute_queue_supported,
    uint32_t* compute_queue_family, uint32_t* compute_queue_flags,
    uint32_t* transfer_queue_supported, int32_t* transfer_queue_family,
    uint32_t* transfer_queue_flags,
    uint32_t* dedicated_transfer_queue_available,
    uint32_t* timeline_semaphore_supported,
    uint32_t* async_transfer_requested,
    uint32_t* async_transfer_eligible,
    uint32_t* async_transfer_enabled,
    uint32_t* async_transfer_resources_ready,
    uint64_t* async_transfer_min_bytes,
    uint32_t* physical_device_count,
    uint32_t* selected_device_index,
    char* device_selection,
    size_t device_selection_capacity,
    uint32_t* simulation_enabled,
    uint32_t* simulation_force_staging_requested,
    uint32_t* simulation_force_staging,
    uint32_t* simulation_strict_budget,
    uint64_t* simulation_transfer_latency_us,
    uint64_t* simulated_vram_budget,
    uint64_t* simulated_vram_usage,
    uint32_t* policy_memory_topology) {
  const auto& d = Device::get();
  if (vendor_id) *vendor_id = d.vendor_id;
  if (device_id) *device_id = d.device_id;
  if (shader_int8_supported) {
    *shader_int8_supported = d.has_shader_int8 ? 1u : 0u;
  }
  if (storage_buffer_8bit_supported) {
    *storage_buffer_8bit_supported = d.has_storage_buffer_8bit ? 1u : 0u;
  }
  if (uniform_and_storage_buffer_8bit_supported) {
    *uniform_and_storage_buffer_8bit_supported =
        d.has_uniform_and_storage_buffer_8bit ? 1u : 0u;
  }
  if (storage_push_constant_8bit_supported) {
    *storage_push_constant_8bit_supported =
        d.has_storage_push_constant_8bit ? 1u : 0u;
  }
  if (driver_version) *driver_version = d.driver_version;
  if (timestamp_valid_bits) *timestamp_valid_bits = d.timestamp_valid_bits;
  if (timestamp_period_ns) *timestamp_period_ns = d.timestamp_period_ns;
  if (compute_queue_supported) {
    *compute_queue_supported = d.compute_queue_supported ? 1u : 0u;
  }
  if (compute_queue_family) *compute_queue_family = d.queue_family;
  if (compute_queue_flags) {
    *compute_queue_flags = static_cast<uint32_t>(d.compute_queue_flags);
  }
  if (transfer_queue_supported) {
    *transfer_queue_supported = d.transfer_queue_supported ? 1u : 0u;
  }
  if (transfer_queue_family) {
    *transfer_queue_family = d.transfer_queue_supported
        ? static_cast<int32_t>(d.transfer_queue_family)
        : -1;
  }
  if (transfer_queue_flags) {
    *transfer_queue_flags = static_cast<uint32_t>(d.transfer_queue_flags);
  }
  if (dedicated_transfer_queue_available) {
    *dedicated_transfer_queue_available =
        d.dedicated_transfer_queue_available ? 1u : 0u;
  }
  if (timeline_semaphore_supported) {
    *timeline_semaphore_supported = d.timeline_semaphore_supported ? 1u : 0u;
  }
  if (async_transfer_requested) {
    *async_transfer_requested = d.async_transfer_requested ? 1u : 0u;
  }
  if (async_transfer_eligible) {
    *async_transfer_eligible = d.async_transfer_eligible ? 1u : 0u;
  }
  if (async_transfer_enabled) {
    *async_transfer_enabled = d.async_transfer_enabled ? 1u : 0u;
  }
  if (async_transfer_resources_ready) {
    *async_transfer_resources_ready =
        d.async_transfer_resources_ready ? 1u : 0u;
  }
  if (async_transfer_min_bytes) {
    *async_transfer_min_bytes = static_cast<uint64_t>(d.async_transfer_min_bytes);
  }
  if (physical_device_count) *physical_device_count = d.physical_device_count;
  if (selected_device_index) *selected_device_index = d.selected_device_index;
  if (device_selection && device_selection_capacity != 0) {
    std::strncpy(device_selection, d.device_selection.c_str(),
                 device_selection_capacity - 1);
    device_selection[device_selection_capacity - 1] = '\0';
  }
  if (max_shared_memory) *max_shared_memory = d.max_shared_memory;
  if (max_workgroup_invocations) {
    *max_workgroup_invocations = d.max_workgroup_invocations;
  }
  if (memory_topology) *memory_topology = static_cast<uint32_t>(d.memory_topology);
  if (has_device_local_only) *has_device_local_only = d.has_device_local_only ? 1u : 0u;
  if (has_host_visible_device_local) {
    *has_host_visible_device_local = d.has_host_visible_device_local ? 1u : 0u;
  }
  if (has_host_visible_system) *has_host_visible_system = d.has_host_visible_system ? 1u : 0u;
  if (device_local_budget) *device_local_budget = static_cast<uint64_t>(d.device_local_budget);
  if (device_local_usage) *device_local_usage = static_cast<uint64_t>(d.device_local_usage);
  if (device_local_only_budget) *device_local_only_budget = static_cast<uint64_t>(d.device_local_only_budget);
  if (device_local_only_usage) *device_local_only_usage = static_cast<uint64_t>(d.device_local_only_usage);
  if (host_visible_device_local_budget) {
    *host_visible_device_local_budget = static_cast<uint64_t>(d.host_visible_device_local_budget);
  }
  if (host_visible_device_local_usage) {
    *host_visible_device_local_usage = static_cast<uint64_t>(d.host_visible_device_local_usage);
  }
  if (host_visible_system_budget) {
    *host_visible_system_budget = static_cast<uint64_t>(d.host_visible_system_budget);
  }
  if (host_visible_system_usage) {
    *host_visible_system_usage = static_cast<uint64_t>(d.host_visible_system_usage);
  }
  if (name && name_capacity != 0) {
    std::strncpy(name, d.device_name, name_capacity - 1);
    name[name_capacity - 1] = '\0';
  }
  if (simulation_enabled) *simulation_enabled = d.simulation_enabled ? 1u : 0u;
  if (simulation_force_staging_requested) {
    *simulation_force_staging_requested =
        d.simulation_force_staging_requested ? 1u : 0u;
  }
  if (simulation_force_staging) {
    *simulation_force_staging = d.simulation_force_staging ? 1u : 0u;
  }
  if (simulation_strict_budget) {
    *simulation_strict_budget = d.simulation_strict_budget ? 1u : 0u;
  }
  if (simulation_transfer_latency_us) {
    *simulation_transfer_latency_us = d.simulation_transfer_latency_us;
  }
  if (simulated_vram_budget) {
    *simulated_vram_budget = static_cast<uint64_t>(d.simulated_vram_budget);
  }
  if (simulated_vram_usage) {
    *simulated_vram_usage = static_cast<uint64_t>(
        d.simulated_vram_usage.load(std::memory_order_relaxed));
  }
  if (policy_memory_topology) {
    *policy_memory_topology = static_cast<uint32_t>(d.policy_memory_topology());
  }
}

VKSSBO_EXPORT void vkssbo_free(float* p) {
  resident_cache().erase(p);
  free(p);
}

VKSSBO_EXPORT void vkssbo_memory_stats(
    size_t* heap_usage, size_t* heap_budget, size_t* pool_bytes,
    size_t* pool_limit, size_t* pool_entries, uint64_t* pool_hits,
    uint64_t* pool_misses, size_t* weight_bytes, size_t* weight_limit,
    size_t* weight_entries, uint64_t* weight_hits, uint64_t* weight_misses,
    int* rust_policy, size_t* resident_bytes, size_t* resident_limit,
    size_t* resident_entries, uint64_t* resident_hits,
    uint64_t* resident_misses, size_t* staging_bytes,
    size_t* staging_peak_bytes, uint64_t* staging_uploads,
    uint64_t* staging_downloads, size_t* staging_pool_bytes,
    size_t* staging_pool_entries, uint64_t* staging_pool_hits,
    uint64_t* staging_pool_misses) {
  auto budget = Device::get().query_memory_budget();
  size_t entries = 0;
  uint64_t hits = 0;
  uint64_t misses = 0;
  size_t cached_weights = 0;
  size_t weight_count = 0;
  uint64_t weight_cache_hits = 0;
  uint64_t weight_cache_misses = 0;
  size_t cached_resident = 0;
  size_t resident_count = 0;
  uint64_t resident_cache_hits = 0;
  uint64_t resident_cache_misses = 0;
  transient_pool_stats(pool_bytes, &entries, &hits, &misses);
  weight_cache_stats(&cached_weights, &weight_count, &weight_cache_hits,
                     &weight_cache_misses);
  resident_cache_stats(&cached_resident, &resident_count,
                        &resident_cache_hits, &resident_cache_misses);
  staging_stats(staging_bytes, staging_peak_bytes, staging_uploads,
                staging_downloads);
  staging_pool_stats(
      staging_pool_bytes, staging_pool_entries, staging_pool_hits,
      staging_pool_misses);
  if (heap_usage) *heap_usage = static_cast<size_t>(budget.usage);
  if (heap_budget) *heap_budget = static_cast<size_t>(budget.budget);
  if (pool_limit) *pool_limit = transient_pool_limit();
  if (pool_entries) *pool_entries = entries;
  if (pool_hits) *pool_hits = hits;
  if (pool_misses) *pool_misses = misses;
  if (weight_bytes) *weight_bytes = cached_weights;
  if (weight_limit) *weight_limit = weight_cache_limit();
  if (weight_entries) *weight_entries = weight_count;
  if (weight_hits) *weight_hits = weight_cache_hits;
  if (weight_misses) *weight_misses = weight_cache_misses;
  if (resident_bytes) *resident_bytes = cached_resident;
  if (resident_limit) *resident_limit = resident_cache_limit();
  if (resident_entries) *resident_entries = resident_count;
  if (resident_hits) *resident_hits = resident_cache_hits;
  if (resident_misses) *resident_misses = resident_cache_misses;
#ifdef VKSSBO_USE_RUST_MEMORY_POLICY
  if (rust_policy) *rust_policy =
      vkssbo_memory_policy_version() != 0 ? 1 : 0;
#else
  if (rust_policy) *rust_policy = 0;
#endif
}

VKSSBO_EXPORT void vkssbo_memory_trim() {
  transient_pool_trim();
  staging_pool_trim();
  weight_cache_trim();
  resident_cache_trim();
}

VKSSBO_EXPORT void vkssbo_matmul_autotune_stats(
    size_t* entries, uint64_t* cache_hits, uint64_t* cache_misses,
    uint64_t* tune_runs, uint64_t* tune_failures,
    uint64_t* benchmark_dispatches) {
  matmul_autotune_stats_impl(
      entries, cache_hits, cache_misses, tune_runs, tune_failures,
      benchmark_dispatches);
}

VKSSBO_EXPORT void vkssbo_conv2d_autotune_stats(
    size_t* entries, uint64_t* cache_hits, uint64_t* cache_misses,
    uint64_t* tune_runs, uint64_t* tune_failures,
    uint64_t* benchmark_dispatches) {
  conv2d_autotune_stats_impl(
      entries, cache_hits, cache_misses, tune_runs, tune_failures,
      benchmark_dispatches);
}

VKSSBO_EXPORT float* vkssbo_elementwise(int op, int mode, const float* a,
                                        const float* b, float scalar, int numel) {
  try {
  auto& d = Device::get();
  auto& cm = ComputeManager::get();

  cm.begin();
  Dispatch disp(d, cm);
  disp.begin(3);
  InputBuffer da = prepare_input(a, numel);
  disp.bind(0, da.get().buffer, da.get().size);
  Buffer db = create_device_buffer(numel  * 2);
  if (mode == 1) {
    upload_direct(b, numel, db);
  }
  disp.bind(1, db.buffer, db.size);
  Buffer dout = create_device_buffer(numel  * 2);
  disp.bind(2, dout.buffer, dout.size);

  struct P { int op, mode, numel; float scalar; } p{op, mode, numel, scalar};
  disp.run("elementwise", &p, sizeof(p), (numel + 255) / 256, 1, 1);
  cm.submit_and_wait();
  float* out = download_resident(dout, numel);
  disp.cleanup();
  return out;
  } catch (const std::exception& e) {
    abort_compute_manager();
    fprintf(stderr, "[vkssbo] elementwise error: %s\n", e.what());
    return nullptr;
  }
}

VKSSBO_EXPORT float* vkssbo_matmul(const float* a, const float* b, int B, int M,
                                   int K, int N, int transB,
                                   uint64_t b_key) {
  try {
  auto& d = Device::get();
  auto& cm = ComputeManager::get();

  cm.begin();
  InputBuffer da = prepare_input(a, (size_t)B * M * K);
  Buffer db_owned;
  Buffer* db = prepare_weight(b_key, b, (size_t)B * K * N, db_owned);
  Buffer dbias = create_device_buffer(2);
  Buffer dout;
  bool tuned_output_ready = false;

  MatmulVariant variant = MatmulVariant::Tile64;
  const bool explicit_variant = matmul_variant_override(&variant);
  const bool can_tune = d.has_shader_f16 && !explicit_variant &&
      matmul_autotune_enabled() && matmul_should_autotune(B, M, K, N) &&
      cm.timestamps_supported();
  if (can_tune) {
    const std::string key = matmul_tune_key(d, B, M, K, N, transB, false);
    double cached_ns = 0.0;
    if (!matmul_cached_variant(key, &variant, &cached_ns)) {
      const VkDeviceSize output_bytes = static_cast<VkDeviceSize>(B) *
          static_cast<VkDeviceSize>(M) * static_cast<VkDeviceSize>(N) * 2;
      if (!matmul_tune_memory_ok(d, output_bytes)) {
        note_matmul_tune_failure();
        dout = create_device_buffer(static_cast<size_t>(output_bytes));
      } else {
      Buffer out64 = create_device_buffer(static_cast<size_t>(output_bytes));
      Buffer out128 = create_device_buffer(static_cast<size_t>(output_bytes));
      MatmulTuneMeasurement second;
      MatmulTuneMeasurement first = tune_matmul_dispatches(
          d, cm, da, *db, out64, out128, dbias, B, M, K, N, transB, false,
          false, &second);
      if (first.valid && second.valid) {
        variant = second.elapsed_ns * 1.03 < first.elapsed_ns
            ? MatmulVariant::Tile128 : MatmulVariant::Tile64;
        remember_matmul_variant(
            key, variant, variant == MatmulVariant::Tile128
                ? second.elapsed_ns : first.elapsed_ns);
      } else {
        note_matmul_tune_failure();
        variant = MatmulVariant::Tile64;
      }
      dout = variant == MatmulVariant::Tile128 ? std::move(out128)
                                               : std::move(out64);
      tuned_output_ready = true;
      }
    } else {
      dout = create_device_buffer((size_t)B * M * N * 2);
    }
  } else {
    if (!d.has_shader_f16) variant = MatmulVariant::Tile64;
    dout = create_device_buffer((size_t)B * M * N * 2);
  }

  if (tuned_output_ready) {
    float* out = download_resident(dout, (size_t)B * M * N);
    return out;
  }

  Dispatch disp(d, cm);
  if (d.has_shader_f16) {
    record_matmul_dispatch(disp, da, *db, dout, dbias, B, M, K, N, transB,
                           false, false, variant);
  } else {
    disp.begin(4);
    disp.bind(0, da.get().buffer, da.get().size);
    disp.bind(1, db->buffer, db->size);
    disp.bind(2, dout.buffer, dout.size);
    disp.bind(3, dbias.buffer, dbias.size);
    struct P { int B, M, K, N, transB; } p{B, M, K, N, transB};
    disp.run("matmul", &p, sizeof(p), (uint32_t)(N + 7) / 8,
             (uint32_t)(B * M + 7) / 8, 1);
  }
  cm.submit_and_wait();
  float* out = download_resident(dout, (size_t)B * M * N);
  disp.cleanup();
  return out;
  } catch (const std::exception& e) {
    abort_compute_manager();
    fprintf(stderr, "[vkssbo] matmul error: %s\n", e.what());
    return nullptr;
  }
}

// Fused matmul+bias entry point used by aten::linear.  The matmul shader
// already has a fourth bias binding and a hasBias push-constant flag; keeping
// this as a separate ABI preserves the existing matmul callers.  The same
// dispatch optionally applies SiLU after the bias so the bridge can expose an
// explicit Linear+Bias+SiLU building block without changing ordinary linear.
static float* vkssbo_matmul_bias_activation(
    const float* a, const float* b, const float* bias, int B, int M, int K,
    int N, int transB, uint64_t b_key, uint64_t bias_key, int activation) {
  try {
  auto& d = Device::get();
  const char* fusion_env = std::getenv(
      activation == 1 ? "VKSSBO_MATMUL_SILU_FUSION" :
      activation == 2 ? "VKSSBO_MATMUL_GELU_FUSION" :
                        "VKSSBO_MATMUL_BIAS_FUSION");
  const bool fusion_enabled = !fusion_env || fusion_env[0] != '0';
  if (!d.has_shader_f16 || !fusion_enabled) {
    float* out = vkssbo_matmul(a, b, B, M, K, N, transB, b_key);
    if (!out) return out;
    // The fallback mutates the CPU mirror, so its pre-bias GPU allocation is
    // no longer a valid resident representation of the result.
    resident_cache().erase(out);
    for (size_t row = 0; row < (size_t)B * M; ++row)
      for (int n = 0; n < N; ++n) {
        float v = out[row * (size_t)N + n];
        if (bias) v += bias[n];
        if (activation == 1) {
          v = v / (1.0f + std::exp(-v));
        } else if (activation == 2) {
          v = 0.5f * v * (1.0f + std::tanh(
              0.7978845608028654f * (v + 0.044715f * v * v * v)));
        }
        out[row * (size_t)N + n] = v;
      }
    return out;
  }
  auto& cm = ComputeManager::get();
  cm.begin();
  InputBuffer da = prepare_input(a, (size_t)B * M * K);
  Buffer db_owned;
  Buffer* db = prepare_weight(b_key, b, (size_t)B * K * N, db_owned);
  Buffer dbias_owned;
  Buffer* dbias = nullptr;
  const bool has_bias = bias != nullptr;
  if (has_bias) {
    dbias = prepare_weight(bias_key, bias, (size_t)N, dbias_owned);
  } else {
    dbias_owned = create_device_buffer(2);
    dbias = &dbias_owned;
  }
  Buffer dout;
  bool tuned_output_ready = false;
  MatmulVariant variant = MatmulVariant::Tile64;
  const bool explicit_variant = matmul_variant_override(&variant);
  const bool can_tune = !explicit_variant && matmul_autotune_enabled() &&
      matmul_should_autotune(B, M, K, N) && cm.timestamps_supported();
  if (can_tune) {
    const std::string key = matmul_tune_key(
        d, B, M, K, N, transB, has_bias, activation);
    double cached_ns = 0.0;
    if (!matmul_cached_variant(key, &variant, &cached_ns)) {
      const VkDeviceSize output_bytes = static_cast<VkDeviceSize>(B) *
          static_cast<VkDeviceSize>(M) * static_cast<VkDeviceSize>(N) * 2;
      if (!matmul_tune_memory_ok(d, output_bytes)) {
        note_matmul_tune_failure();
        dout = create_device_buffer(static_cast<size_t>(output_bytes));
      } else {
      Buffer out64 = create_device_buffer(static_cast<size_t>(output_bytes));
      Buffer out128 = create_device_buffer(static_cast<size_t>(output_bytes));
      MatmulTuneMeasurement second;
      MatmulTuneMeasurement first = tune_matmul_dispatches(
          d, cm, da, *db, out64, out128, *dbias, B, M, K, N, transB, has_bias,
          activation, &second);
      if (first.valid && second.valid) {
        variant = second.elapsed_ns * 1.03 < first.elapsed_ns
            ? MatmulVariant::Tile128 : MatmulVariant::Tile64;
        remember_matmul_variant(
            key, variant, variant == MatmulVariant::Tile128
                ? second.elapsed_ns : first.elapsed_ns);
      } else {
        note_matmul_tune_failure();
        variant = MatmulVariant::Tile64;
      }
      dout = variant == MatmulVariant::Tile128 ? std::move(out128)
                                               : std::move(out64);
      tuned_output_ready = true;
      }
    } else {
      dout = create_device_buffer((size_t)B * M * N * 2);
    }
  } else {
    dout = create_device_buffer((size_t)B * M * N * 2);
  }
  if (tuned_output_ready) {
    float* out = download_resident(dout, (size_t)B * M * N);
    return out;
  }
  Dispatch disp(d, cm);
  record_matmul_dispatch(disp, da, *db, dout, *dbias, B, M, K, N, transB,
                         has_bias, activation, variant);
  cm.submit_and_wait();
  float* out = download_resident(dout, (size_t)B * M * N);
  disp.cleanup();
  return out;
  } catch (const std::exception& e) {
    abort_compute_manager();
    fprintf(stderr, "[vkssbo] matmul%s error: %s\n",
            activation == 1 ? "_bias_silu" :
            activation == 2 ? "_bias_gelu" : "_bias", e.what());
    return nullptr;
  }
}

extern "C" {
VKSSBO_EXPORT float* vkssbo_matmul_bias(
    const float* a, const float* b, const float* bias, int B, int M, int K,
    int N, int transB, uint64_t b_key, uint64_t bias_key) {
  return vkssbo_matmul_bias_activation(
      a, b, bias, B, M, K, N, transB, b_key, bias_key, 0);
}

VKSSBO_EXPORT float* vkssbo_matmul_bias_silu(
    const float* a, const float* b, const float* bias, int B, int M, int K,
    int N, int transB, uint64_t b_key, uint64_t bias_key) {
  return vkssbo_matmul_bias_activation(
      a, b, bias, B, M, K, N, transB, b_key, bias_key, 1);
}

VKSSBO_EXPORT float* vkssbo_matmul_bias_gelu(
    const float* a, const float* b, const float* bias, int B, int M, int K,
    int N, int transB, uint64_t b_key, uint64_t bias_key) {
  return vkssbo_matmul_bias_activation(
      a, b, bias, B, M, K, N, transB, b_key, bias_key, 2);
}
} // extern "C"

// Fused attention score path: scores = softmax(query @ key^T * scale).
// Both dispatches stay in one command buffer and the score matrix remains in
// fp16 device-local storage; only the final softmax result is downloaded.
VKSSBO_EXPORT float* vkssbo_matmul_softmax(
    const float* query, const float* key, int B, int M, int K, int N,
    float scale, uint64_t key_key) {
  try {
  auto& d = Device::get();
  if (!d.has_shader_f16) return nullptr;
  auto& cm = ComputeManager::get();

  cm.begin();
  InputBuffer dq = prepare_input(query, (size_t)B * M * K);
  Buffer dk_owned;
  // The key is supplied as [B, N, K]; transB=1 in the matmul shader reads
  // that layout as [B, K, N] without a CPU-side transpose or copy.
  Buffer* dk = prepare_weight(key_key, key, (size_t)B * N * K, dk_owned);
  Buffer dummy = create_device_buffer(2);
  Buffer scores = create_device_buffer((size_t)B * M * N * 2);
  Buffer output = create_device_buffer((size_t)B * M * N * 2);

  Dispatch matmul_disp(d, cm);
  record_matmul_dispatch(
      matmul_disp, dq, *dk, scores, dummy, B, M, K, N, 1, false, 0,
      MatmulVariant::Tile64, scale);
  // Intel and other drivers used by the supported Vulkan matrix require an
  // explicit compute barrier between a producer dispatch and softmax.
  cm.barrier_compute_to_compute();

  Dispatch softmax_disp(d, cm);
  softmax_disp.begin(2);
  softmax_disp.bind(0, scores.buffer, scores.size);
  softmax_disp.bind(1, output.buffer, output.size);
  struct P { int rows, cols; } p{B * M, N};
  softmax_disp.run("softmax", &p, sizeof(p), (uint32_t)(B * M), 1, 1);

  cm.submit_and_wait();
  float* out = download_resident(output, (size_t)B * M * N);
  matmul_disp.cleanup();
  softmax_disp.cleanup();
  return out;
  } catch (const std::exception& e) {
    abort_compute_manager();
    fprintf(stderr, "[vkssbo] matmul_softmax error: %s\n", e.what());
    return nullptr;
  } catch (...) {
    abort_compute_manager();
    fprintf(stderr, "[vkssbo] matmul_softmax unknown exception\n");
    return nullptr;
  }
}

VKSSBO_EXPORT float* vkssbo_softmax(const float* a, int rows, int cols) {
  try {
  auto& d = Device::get();
  auto& cm = ComputeManager::get();

  cm.begin();
  Dispatch disp(d, cm);
  disp.begin(2);
  InputBuffer da = prepare_input(a, (size_t)rows * cols);
  disp.bind(0, da.get().buffer, da.get().size);
  Buffer dout = create_device_buffer((size_t)rows * cols  * 2);
  disp.bind(1, dout.buffer, dout.size);

  struct P { int rows, cols; } p{rows, cols};
  disp.run("softmax", &p, sizeof(p), rows, 1, 1);
  cm.submit_and_wait();
  float* out = download_resident(dout, (size_t)rows * cols);
  disp.cleanup();
  return out;
  } catch (const std::exception& e) {
    abort_compute_manager();
    fprintf(stderr, "[vkssbo] softmax error: %s\n", e.what());
    return nullptr;
  }
}

VKSSBO_EXPORT float* vkssbo_layernorm(const float* a, const float* w,
                                      const float* b, int rows, int cols,
                                      float eps, int has_w, int has_b,
                                      uint64_t w_key, uint64_t b_key) {
  try {
  auto& d = Device::get();
  auto& cm = ComputeManager::get();

  cm.begin();
  Dispatch disp(d, cm);
  disp.begin(4);
  InputBuffer da = prepare_input(a, (size_t)rows * cols);
  disp.bind(0, da.get().buffer, da.get().size);
  Buffer dout = create_device_buffer((size_t)rows * cols  * 2);
  disp.bind(1, dout.buffer, dout.size);
  Buffer dw_owned;
  Buffer* dw = nullptr;
  if (has_w) {
    dw = prepare_weight(w_key, w, (size_t)cols, dw_owned);
  } else {
    dw_owned = create_device_buffer(2);
    dw = &dw_owned;
  }
  disp.bind(2, dw->buffer, dw->size);
  Buffer dbb_owned;
  Buffer* dbb = nullptr;
  if (has_b) {
    dbb = prepare_weight(b_key, b, (size_t)cols, dbb_owned);
  } else {
    dbb_owned = create_device_buffer(2);
    dbb = &dbb_owned;
  }
  disp.bind(3, dbb->buffer, dbb->size);

  struct P { int rows, cols; float eps; int has_w, has_b; } p{rows, cols, eps, has_w, has_b};
  disp.run("layernorm", &p, sizeof(p), rows, 1, 1);
  cm.submit_and_wait();
  float* out = download_resident(dout, (size_t)rows * cols);
  disp.cleanup();
  return out;
  } catch (const std::exception& e) {
    abort_compute_manager();
    fprintf(stderr, "[vkssbo] layernorm error: %s\n", e.what());
    return nullptr;
  }
}

static float* vkssbo_conv2d_impl(
    const float* x, const float* w, const float* bias, int B, int C, int O,
    int H, int W, int Ho, int Wo, int KH, int KW, int SH, int SW, int DH,
    int DW, int PH, int PW, int G, uint64_t w_key, uint64_t bias_key,
    bool fuse_silu) {
  try {
  auto& d = Device::get();
  auto& cm = ComputeManager::get();

  cm.begin();
  InputBuffer dx = prepare_input(x, (size_t)B * C * H * W);
  if (std::getenv("VK_TRACE")) {
    float* check = download_direct(dx.get(), (size_t)B * C * H * W);
    fprintf(stderr, "[conv-upload] x[0]=%.4f x[last]=%.4f numel=%zu\n",
            check[0], check[(size_t)B * C * H * W - 1], (size_t)B * C * H * W);
    free(check);
  }
  Buffer dw_owned;
  Buffer* dw = prepare_weight(w_key, w, (size_t)O * (C / G) * KH * KW,
                              dw_owned);
  Buffer db_owned;
  Buffer* db = prepare_weight(bias_key, bias, (size_t)O, db_owned);
  const char* x32_env = std::getenv("VKSSBO_CONV_X32");
  const char* x32c32_env = std::getenv("VKSSBO_CONV_C32");
  const char* x32v4_env = std::getenv("VKSSBO_CONV_V4");
  const char* x32v4acc_env = std::getenv("VKSSBO_CONV_V4ACC");
  // The shared-tile kernel is only selected for channel-aligned inputs.  The
  // generic path remains the correctness fallback for tail C chunks on
  // drivers where partially populated shared tiles are unreliable.
  const bool use_x32 =
      (!x32_env || x32_env[0] != '0') &&
      d.has_shader_f16 &&
      d.max_shared_memory >= 16u * 1024u &&
      d.max_workgroup_invocations >= 256u &&
      d.max_workgroup_size_x >= 16u && d.max_workgroup_size_y >= 16u &&
      G == 1 && O >= 32 && (C % 16) == 0 && KH == 3 && KW == 3 &&
      SH == 1 && SW == 1 &&
      DH == 1 && DW == 1 && PH >= 0 && PW >= 0 && PH <= 1 && PW <= 1;

  Conv2dVariant selected_variant = Conv2dVariant::V4Acc;
  bool tune_submitted = false;
  const bool explicit_variant = conv2d_has_explicit_variant_override();
  const bool can_autotune = !fuse_silu && !explicit_variant &&
      conv2d_autotune_enabled() &&
      use_x32 && conv2d_autotune_eligible(
          d, B, C, O, H, W, Ho, Wo, KH, KW, SH, SW, DH, DW, PH, PW, G) &&
      cm.timestamps_supported();
  if (can_autotune) {
    const std::string key = conv2d_tune_key(
        d, B, C, O, H, W, Ho, Wo, KH, KW, SH, SW, DH, DW, PH, PW, G,
        bias_key != 0);
    double cached_ns = 0.0;
    if (conv2d_tuning_suppressed(key)) {
      selected_variant = Conv2dVariant::V4Acc;
    } else if (!conv2d_cached_variant(key, &selected_variant, &cached_ns)) {
      const int probe_ho = std::min(Ho, 16);
      const int probe_wo = std::min(Wo, 16);
      VkDeviceSize probe_output_bytes = 0;
      if (!conv2d_probe_output_bytes(
              B, O, probe_ho, probe_wo, &probe_output_bytes) ||
          !conv2d_tune_memory_ok(d, probe_output_bytes)) {
        note_conv2d_tune_failure(key);
      } else {
        bool probes_allocated = false;
        Buffer probe_v4acc;
        Buffer probe_c32;
        try {
          probe_v4acc = create_device_buffer(probe_output_bytes);
          probe_c32 = create_device_buffer(probe_output_bytes);
          probes_allocated = true;
        } catch (...) {
          // A/B probing is optional. Keep the original command buffer (and
          // its pending uploads) intact and run the conservative full kernel.
          note_conv2d_tune_failure(key);
        }
        if (probes_allocated) {
          Conv2dTiledParams probe_params{
              B, C, O, H, W, probe_ho, probe_wo, KH, KW, SH, SW, PH, PW, G};
          try {
            Conv2dTuneMeasurement c32_measurement;
            Conv2dTuneMeasurement v4acc_measurement =
                tune_conv2d_dispatches(
                    d, cm, dx, *dw, *db, probe_v4acc, probe_c32,
                    probe_params, &c32_measurement);
            tune_submitted = true;
            if (v4acc_measurement.valid && c32_measurement.valid) {
              selected_variant =
                  c32_measurement.elapsed_ns * 1.03 <
                      v4acc_measurement.elapsed_ns
                  ? Conv2dVariant::C32
                  : Conv2dVariant::V4Acc;
              remember_conv2d_variant(
                  key, selected_variant,
                  selected_variant == Conv2dVariant::C32
                      ? c32_measurement.elapsed_ns
                      : v4acc_measurement.elapsed_ns);
            } else {
              note_conv2d_tune_failure(key);
              selected_variant = Conv2dVariant::V4Acc;
            }
          } catch (...) {
            // If recording or submission failed, reset the command buffer and
            // restore all uploads before attempting the conservative path.
            note_conv2d_tune_failure(key);
            selected_variant = Conv2dVariant::V4Acc;
            abort_compute_manager();
            cm.begin();
            upload_direct(x, (size_t)B * C * H * W, dx.get());
            upload_direct(w, (size_t)O * (C / G) * KH * KW, *dw);
            upload_direct(bias, (size_t)O, *db);
            tune_submitted = false;
          }
        }
      }
    }
  }
  // Probes are deliberately bounded and never reused as the user-visible
  // output. A successful tune therefore starts a fresh full-shape dispatch.
  if (tune_submitted) cm.begin();

  // Explicit C32 remains available for manual A/B; otherwise the autotuner
  // may promote it only after clearing the 3% stability threshold.
  const bool use_x32c32 =
      ((x32c32_env && x32c32_env[0] == '1') &&
       use_x32 && d.max_shared_memory >= 40u * 1024u && (C % 32) == 0) ||
      (can_autotune && selected_variant == Conv2dVariant::C32);
  // The vec4 weight staging path has the same shared-memory footprint as the
  // scalar x32 kernel and is the default. Set VKSSBO_CONV_V4=0 for fallback
  // A/B tests on drivers with unusual 16-bit vector behavior.
  const bool use_x32v4 =
      (!x32v4_env || x32v4_env[0] != '0') &&
      use_x32 && !use_x32c32;
  const bool use_x32v4acc =
      (!x32v4acc_env || x32v4acc_env[0] != '0') && use_x32v4;
  if (fuse_silu && !use_x32v4acc) {
    // The public fused entry point treats nullptr as "use the unfused
    // fallback". Do not silently claim fusion on a different kernel.  The
    // setup above may already have begun recording uploads, so abort before
    // returning; otherwise the caller's unfused fallback would inherit a
    // command buffer that is still in the recording state.
    abort_compute_manager();
    return nullptr;
  }
  const bool use_tiled = !use_x32 &&
      G == 1 && SH == 1 && SW == 1 && DH == 1 && DW == 1 && KH <= 3 && KW <= 3;
  Buffer dout = create_device_buffer(
      (size_t)B * O * Ho * Wo * ((use_tiled || use_x32 || use_x32c32)
          ? sizeof(uint16_t) : sizeof(float)));
  Dispatch disp(d, cm);
  disp.begin(4);
  disp.bind(0, dx.get().buffer, dx.get().size);
  disp.bind(1, dw->buffer, dw->size);
  disp.bind(2, db->buffer, db->size);
  disp.bind(3, dout.buffer, dout.size);

  Conv2dTiledParams tiled_p{
      B, C, O, H, W, Ho, Wo, KH, KW, SH, SW, PH, PW, G};
  struct SimpleP {
    int B, C, O, H, W, Ho, Wo, KH, KW, SH, SW, DH, DW, PH, PW, G;
  } simple_p{B, C, O, H, W, Ho, Wo, KH, KW, SH, SW, DH, DW, PH, PW, G};
  if (std::getenv("VK_TRACE")) {
    fprintf(stderr,
            "[conv] kernel=%s B=%d C=%d O=%d H=%d W=%d Ho=%d Wo=%d "
            "KH=%d KW=%d SH=%d SW=%d DH=%d DW=%d PH=%d PW=%d G=%d\n",
            use_x32c32 ? "x32c32" : (use_x32v4acc ? "x32v4acc" : (use_x32v4 ? "x32v4" : (use_x32 ? "x32" : (use_tiled ? "tiled" : "simple")))), B, C, O, H, W, Ho, Wo, KH,
            KW, SH, SW, DH, DW, PH, PW, G);
    fprintf(stderr, "[conv-select] shared=%u x32=%d c32=%d v4=%d "
                    "env_x32=%s env_c32=%s env_v4=%s\n",
            d.max_shared_memory, use_x32 ? 1 : 0, use_x32c32 ? 1 : 0,
            use_x32v4 ? 1 : 0, x32_env ? x32_env : "<unset>",
            x32c32_env ? x32c32_env : "<unset>",
            x32v4_env ? x32v4_env : "<unset>");
  }
  if (use_x32c32) {
    disp.run("conv2d_x32c32", &tiled_p, sizeof(tiled_p), (Wo + 15) / 16,
             (Ho + 15) / 16, B * ((O + 31) / 32));
  } else if (use_x32v4acc) {
    disp.run(
             fuse_silu ? "conv2d_x32v4acc_silu" : "conv2d_x32v4acc",
             &tiled_p, sizeof(tiled_p), (Wo + 15) / 16,
             (Ho + 15) / 16, B * ((O + 31) / 32));
  } else if (use_x32v4) {
    disp.run("conv2d_x32v4", &tiled_p, sizeof(tiled_p), (Wo + 15) / 16,
             (Ho + 15) / 16, B * ((O + 31) / 32));
  } else if (use_x32) {
    disp.run("conv2d_x32", &tiled_p, sizeof(tiled_p), (Wo + 15) / 16,
             (Ho + 15) / 16, B * ((O + 31) / 32));
  } else if (use_tiled) {
    disp.run("conv2d", &tiled_p, sizeof(tiled_p), (Wo + 7) / 8,
             (Ho + 7) / 8, B * O);
  } else {
    disp.run("conv2d_simple_f32o", &simple_p, sizeof(simple_p), (Wo + 7) / 8,
             (Ho + 7) / 8, B * O);
  }
  cm.submit_and_wait();
  float* out = (use_tiled || use_x32 || use_x32c32)
      ? download_resident(dout, (size_t)B * O * Ho * Wo)
      : download_f32_direct(dout, (size_t)B * O * Ho * Wo);
  disp.cleanup();
  return out;
  } catch (const std::exception& e) {
    abort_compute_manager();
    fprintf(stderr, "[vkssbo] conv2d error: %s\n", e.what());
    return nullptr;
  }
}

VKSSBO_EXPORT float* vkssbo_conv2d(
    const float* x, const float* w, const float* bias, int B, int C, int O,
    int H, int W, int Ho, int Wo, int KH, int KW, int SH, int SW, int DH,
    int DW, int PH, int PW, int G, uint64_t w_key, uint64_t bias_key) {
  return vkssbo_conv2d_impl(
      x, w, bias, B, C, O, H, W, Ho, Wo, KH, KW, SH, SW, DH, DW, PH, PW,
      G, w_key, bias_key, false);
}

VKSSBO_EXPORT float* vkssbo_conv2d_silu(
    const float* x, const float* w, const float* bias, int B, int C, int O,
    int H, int W, int Ho, int Wo, int KH, int KW, int SH, int SW, int DH,
    int DW, int PH, int PW, int G, uint64_t w_key, uint64_t bias_key) {
  return vkssbo_conv2d_impl(
      x, w, bias, B, C, O, H, W, Ho, Wo, KH, KW, SH, SW, DH, DW, PH, PW,
      G, w_key, bias_key, true);
}

VKSSBO_EXPORT float* vkssbo_conv3d(const float* x, const float* w,
                                   const float* bias, int B, int C, int O,
                                   int D, int H, int W, int Do, int Ho, int Wo,
                                   int KD, int KH, int KW,
                                   int SD, int SH, int SW,
                                   int PD, int PH, int PW,
                                   int DD, int DH, int DW, int G,
                                   uint64_t w_key, uint64_t bias_key) {
  try {
  auto& d = Device::get();
  auto& cm = ComputeManager::get();

  cm.begin();
  Dispatch disp(d, cm);
  disp.begin(4);
  InputBuffer dx = prepare_input(x, (size_t)B * C * D * H * W);
  disp.bind(0, dx.get().buffer, dx.get().size);
  Buffer dw_owned;
  Buffer* dw = prepare_weight(w_key, w, (size_t)O * (C / G) * KD * KH * KW,
                              dw_owned);
  disp.bind(1, dw->buffer, dw->size);
  Buffer db_owned;
  Buffer* db = prepare_weight(bias_key, bias, (size_t)O, db_owned);
  disp.bind(2, db->buffer, db->size);
  Buffer dout = create_device_buffer((size_t)B * O * Do * Ho * Wo * 2);
  disp.bind(3, dout.buffer, dout.size);

  struct PFast {
    int B, C, O, D, H, W, Do, Ho, Wo, KD, KH, KW, SD, SH, SW, PD, PH, PW, G;
  } pfast{B, C, O, D, H, W, Do, Ho, Wo, KD, KH, KW, SD, SH, SW, PD, PH, PW, G};
  struct PDilated {
    int B, C, O, D, H, W, Do, Ho, Wo, KD, KH, KW, SD, SH, SW, PD, PH, PW,
        DD, DH, DW, G;
  } pdilated{B, C, O, D, H, W, Do, Ho, Wo, KD, KH, KW, SD, SH, SW, PD, PH, PW,
      DD, DH, DW, G};
  fprintf(stderr, "[conv3d] B=%d C=%d O=%d D=%d H=%d W=%d Do=%d Ho=%d Wo=%d K=%dx%dx%d\n",
          B, C, O, D, H, W, Do, Ho, Wo, KD, KH, KW);
  if (DD == 1 && DH == 1 && DW == 1) {
    disp.run("conv3d", &pfast, sizeof(pfast), (Wo + 7) / 8,
             (Ho * Do + 7) / 8, B * O);
  } else {
    disp.run("conv3d_dilated", &pdilated, sizeof(pdilated), (Wo + 7) / 8,
             (Ho * Do + 7) / 8, B * O);
  }
  cm.submit_and_wait();
  float* out = download_resident(dout, (size_t)B * O * Do * Ho * Wo);
  disp.cleanup();
  return out;
  } catch (const std::exception& e) {
    abort_compute_manager();
    fprintf(stderr, "[vkssbo] conv3d error: %s\n", e.what());
    return nullptr;
  }
}

VKSSBO_EXPORT float* vkssbo_conv1d(const float* x, const float* w,
                                   const float* bias, int B, int C, int O,
                                   int L, int Lo, int KL, int SL, int PL,
                                   int DL, int G, uint64_t w_key,
                                   uint64_t bias_key) {
  // (B,C,L) and (O,C,KL) are layout-identical to (B,C,1,L) / (O,C,1,KL).
  // Fake H dim: size 1, no padding.
  return vkssbo_conv2d(x, w, bias, B, C, O, 1, L, 1, Lo, 1, KL, 1, SL,
                       1, DL, 0, PL, G, w_key, bias_key);
}

VKSSBO_EXPORT float* vkssbo_groupnorm(const float* a, const float* w,                                      const float* b, int B, int C, int H,
                                      int W, int G, float eps, int has_w,
                                      int has_b, uint64_t w_key,
                                      uint64_t b_key) {
  try {
  auto& d = Device::get();
  auto& cm = ComputeManager::get();

  cm.begin();
  Dispatch disp(d, cm);
  disp.begin(4);
  InputBuffer da = prepare_input(a, (size_t)B * C * H * W);
  disp.bind(0, da.get().buffer, da.get().size);
  Buffer dout = create_device_buffer((size_t)B * C * H * W  * 2);
  disp.bind(1, dout.buffer, dout.size);
  Buffer dw_owned;
  Buffer* dw = nullptr;
  if (has_w) {
    dw = prepare_weight(w_key, w, (size_t)C, dw_owned);
  } else {
    dw_owned = create_device_buffer(2);
    dw = &dw_owned;
  }
  disp.bind(2, dw->buffer, dw->size);
  Buffer dbb_owned;
  Buffer* dbb = nullptr;
  if (has_b) {
    dbb = prepare_weight(b_key, b, (size_t)C, dbb_owned);
  } else {
    dbb_owned = create_device_buffer(2);
    dbb = &dbb_owned;
  }
  disp.bind(3, dbb->buffer, dbb->size);

  struct P { int B, C, H, W, G; float eps; int has_w, has_b; } p{B, C, H, W, G, eps, has_w, has_b};
  disp.run("groupnorm", &p, sizeof(p), (uint32_t)(B * C + 255) / 256, 1, 1);
  cm.submit_and_wait();
  float* out = download_resident(dout, (size_t)B * C * H * W);
  disp.cleanup();
  return out;
  } catch (const std::exception& e) {
    abort_compute_manager();
    fprintf(stderr, "[vkssbo] groupnorm error: %s\n", e.what());
    return nullptr;
  }
}

VKSSBO_EXPORT float* vkssbo_upsample(const float* a, int B, int C, int H, int W,
                                     int Ho, int Wo, int mode) {
  try {
  auto& d = Device::get();
  auto& cm = ComputeManager::get();

  cm.begin();
  Dispatch disp(d, cm);
  disp.begin(2);
  InputBuffer da = prepare_input(a, (size_t)B * C * H * W);
  disp.bind(0, da.get().buffer, da.get().size);
  Buffer dout = create_device_buffer((size_t)B * C * Ho * Wo  * 2);
  disp.bind(1, dout.buffer, dout.size);

  struct P { int B, C, H, W, Ho, Wo, mode; } p{B, C, H, W, Ho, Wo, mode};
  disp.run("upsample_v2", &p, sizeof(p), (uint32_t)(Wo + 7) / 8,
           (uint32_t)(Ho + 7) / 8, (uint32_t)(B * C));
  cm.submit_and_wait();
  float* out = download_resident(dout, (size_t)B * C * Ho * Wo);
  disp.cleanup();
  return out;
  } catch (const std::exception& e) {
    abort_compute_manager();
    fprintf(stderr, "[vkssbo] upsample error: %s\n", e.what());
    return nullptr;
  }
}

VKSSBO_EXPORT float* vkssbo_cat4(const float* a0, const float* a1,
                                 const float* a2, const float* a3, int B,
                                 int H, int W, int C0, int C1, int C2, int C3,
                                 int O) {
  try {
  auto& d = Device::get();
  auto& cm = ComputeManager::get();

  cm.begin();
  Dispatch disp(d, cm);
  disp.begin(5);
  size_t base = (size_t)B * H * W;
  InputBuffer d0;
  if (C0 > 0) d0 = prepare_input(a0, base * C0);
  else d0.owned = create_device_buffer(2);
  disp.bind(0, d0.get().buffer, d0.get().size);
  InputBuffer d1;
  if (C1 > 0) d1 = prepare_input(a1, base * C1);
  else d1.owned = create_device_buffer(2);
  disp.bind(1, d1.get().buffer, d1.get().size);
  InputBuffer d2;
  if (C2 > 0) d2 = prepare_input(a2, base * C2);
  else d2.owned = create_device_buffer(2);
  disp.bind(2, d2.get().buffer, d2.get().size);
  InputBuffer d3;
  if (C3 > 0) d3 = prepare_input(a3, base * C3);
  else d3.owned = create_device_buffer(2);
  disp.bind(3, d3.get().buffer, d3.get().size);
  Buffer dout = create_device_buffer(base * O  * 2);
  disp.bind(4, dout.buffer, dout.size);

  struct P { int B, H, W, C0, C1, C2, C3, O; } p{B, H, W, C0, C1, C2, C3, O};
  disp.run("cat4", &p, sizeof(p), (uint32_t)(W + 7) / 8, (uint32_t)(H + 7) / 8, (uint32_t)(B * O));
  cm.submit_and_wait();
  float* out = download_resident(dout, base * O);
  disp.cleanup();
  return out;
  } catch (const std::exception& e) {
    abort_compute_manager();
    fprintf(stderr, "[vkssbo] cat4 error: %s\n", e.what());
    return nullptr;
  }
}

} // extern "C"
