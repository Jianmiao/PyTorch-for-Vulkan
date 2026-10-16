// SSBO Vulkan backend: bridges the stock torch Vulkan device kernels to the
// SSBO compute backend (data downloaded to CPU, computed on GPU via SSBO,
// uploaded back). Fallbacks return undefined tensors.
#include "SsboBackend.h"

#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/NativeBuffer.h>
#include <torch/library.h>

#include <cmath>
#include <algorithm>
#include <atomic>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

extern "C" {
void vkssbo_init(const char* spv_dir);
float* vkssbo_elementwise(int op, int mode, const float* a, const float* b,
                           float scalar, int numel);
float* vkssbo_matmul(const float* a, const float* b, int B, int M, int K, int N,
                     int transB, uint64_t b_key);
float* vkssbo_matmul_bias(const float* a, const float* b, const float* bias,
                          int B, int M, int K, int N, int transB,
                          uint64_t b_key, uint64_t bias_key);
float* vkssbo_matmul_bias_silu(
    const float* a, const float* b, const float* bias,
    int B, int M, int K, int N, int transB,
    uint64_t b_key, uint64_t bias_key);
float* vkssbo_matmul_bias_gelu(
    const float* a, const float* b, const float* bias,
    int B, int M, int K, int N, int transB,
    uint64_t b_key, uint64_t bias_key);
float* vkssbo_matmul_softmax(
    const float* query, const float* key, int B, int M, int K, int N,
    float scale, uint64_t key_key);
float* vkssbo_softmax(const float* a, int rows, int cols);
float* vkssbo_layernorm(const float* a, const float* w, const float* b,
                        int rows, int cols, float eps, int has_w, int has_b,
                        uint64_t w_key, uint64_t b_key);
float* vkssbo_groupnorm(const float* a, const float* w, const float* b,
                        int B, int C, int H, int W, int G, float eps,
                        int has_w, int has_b, uint64_t w_key, uint64_t b_key);
float* vkssbo_conv2d(const float* x, const float* w, const float* bias,
                     int B, int C, int O, int H, int W, int Ho, int Wo,
                     int KH, int KW, int SH, int SW, int DH, int DW, int PH,
                     int PW, int G,
                     uint64_t w_key, uint64_t bias_key);
float* vkssbo_conv2d_silu(const float* x, const float* w, const float* bias,
                          int B, int C, int O, int H, int W, int Ho, int Wo,
                          int KH, int KW, int SH, int SW, int DH, int DW,
                          int PH, int PW, int G,
                          uint64_t w_key, uint64_t bias_key);
float* vkssbo_conv3d(const float* x, const float* w, const float* bias,
                     int B, int C, int O, int D, int H, int W, int Do, int Ho, int Wo,
                     int KD, int KH, int KW, int SD, int SH, int SW,
                     int PD, int PH, int PW, int DD, int DH, int DW, int G,
                     uint64_t w_key,
                     uint64_t bias_key);
float* vkssbo_conv1d(const float* x, const float* w, const float* bias,
                     int B, int C, int O, int L, int Lo, int KL, int SL, int PL,
                     int DL, int G, uint64_t w_key, uint64_t bias_key);
float* vkssbo_upsample(const float* a, int B, int C, int H, int W, int Ho, int Wo, int mode);
float* vkssbo_cat4(const float* a, const float* b, const float* c, const float* d,
                   int B, int H, int W, int C0, int C1, int C2, int C3, int O);
void vkssbo_free(float* p);
void vkssbo_device_info(uint32_t* vendor_id, uint32_t* device_id,
                        uint32_t* max_shared_memory,
                        uint32_t* max_workgroup_invocations,
                        uint32_t* shader_int8_supported,
                        uint32_t* storage_buffer_8bit_supported,
                        uint32_t* uniform_and_storage_buffer_8bit_supported,
                        uint32_t* storage_push_constant_8bit_supported,
                        uint32_t* memory_topology,
                        uint32_t* has_device_local_only,
                        uint32_t* has_host_visible_device_local,
                        uint32_t* has_host_visible_system,
                        uint64_t* device_local_budget,
                        uint64_t* device_local_usage,
                        uint64_t* device_local_only_budget,
                        uint64_t* device_local_only_usage,
                        uint64_t* host_visible_device_local_budget,
                        uint64_t* host_visible_device_local_usage,
                        uint64_t* host_visible_system_budget,
                        uint64_t* host_visible_system_usage,
                         char* name, size_t name_capacity,
                         uint32_t* driver_version,
                         uint32_t* timestamp_valid_bits,
                         float* timestamp_period_ns,
                         uint32_t* compute_queue_supported,
                         uint32_t* compute_queue_family,
                         uint32_t* compute_queue_flags,
                         uint32_t* transfer_queue_supported,
                         int32_t* transfer_queue_family,
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
                         uint32_t* policy_memory_topology);
void vkssbo_memory_stats(size_t* heap_usage, size_t* heap_budget,
                         size_t* pool_bytes, size_t* pool_limit,
                         size_t* pool_entries, uint64_t* pool_hits,
                         uint64_t* pool_misses, size_t* weight_bytes,
                         size_t* weight_limit, size_t* weight_entries,
                         uint64_t* weight_hits, uint64_t* weight_misses,
                         int* rust_policy, size_t* resident_bytes,
                         size_t* resident_limit, size_t* resident_entries,
                         uint64_t* resident_hits, uint64_t* resident_misses,
                         size_t* staging_bytes, size_t* staging_peak_bytes,
                         uint64_t* staging_uploads, uint64_t* staging_downloads,
                         size_t* staging_pool_bytes,
                         size_t* staging_pool_entries,
                         uint64_t* staging_pool_hits,
                         uint64_t* staging_pool_misses);
void vkssbo_memory_trim();
void vkssbo_matmul_autotune_stats(
    size_t* entries, uint64_t* cache_hits, uint64_t* cache_misses,
    uint64_t* tune_runs, uint64_t* tune_failures,
    uint64_t* benchmark_dispatches);
void vkssbo_conv2d_autotune_stats(
    size_t* entries, uint64_t* cache_hits, uint64_t* cache_misses,
    uint64_t* tune_runs, uint64_t* tune_failures,
    uint64_t* benchmark_dispatches);
void vkssbo_profile_reset();
void vkssbo_profile_stats(
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
    uint64_t* download_calls);
void vkssbo_scheduler_stats(
    uint64_t* last_submitted, uint64_t* completed, uint64_t* in_flight,
    uint64_t* peak_in_flight, uint64_t* submits, uint64_t* polls,
    uint64_t* waits);
}

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace ssbo {

namespace {

constexpr int OP_ADD = 0;
constexpr int OP_SUB = 1;
constexpr int OP_MUL = 2;
constexpr int OP_DIV = 3;
constexpr int OP_SILU = 5;
constexpr int OP_GELU = 6;
constexpr int OP_SIGMOID = 7;
constexpr int OP_TANH = 8;
constexpr int OP_SQRT = 9;
constexpr int OP_EXP = 10;

bool g_enabled = true;

using WeakTensorImpl =
    c10::weak_intrusive_ptr<c10::TensorImpl, c10::UndefinedTensorImpl>;

// SSBO results are downloaded into host-visible memory before they are
// materialized as stock Vulkan tensors. Keep that allocation alive alongside
// the Vulkan Tensor so a following SSBO operator can consume it directly,
// avoiding an otherwise redundant stock Vulkan download. The weak TensorImpl
// key makes the cache non-owning; stale entries are pruned on lookup/insert.
struct HostLease {
  WeakTensorImpl weak;
  std::shared_ptr<float> data;
  int64_t numel = 0;
  uint64_t version = 0;
};

constexpr size_t kMaxHostLeases = 65536;
std::mutex g_host_lease_mutex;
std::unordered_map<c10::TensorImpl*, HostLease> g_cpu_host_leases;
std::unordered_map<c10::TensorImpl*, HostLease> g_vulkan_host_leases;

enum class ProducerKind : uint8_t {
  None = 0,
  Conv1d,
  Conv2d,
  Conv2dSilu,
  Conv3d,
  Matmul,
  LinearSilu,
  LinearGelu,
  MatmulSoftmax,
  Softmax,
  LayerNorm,
  GroupNorm,
  Silu,
  Gelu,
  Sigmoid,
  Tanh,
  Binary,
  Scalar,
  Upsample,
  Cat,
};

struct ProducerRecord {
  WeakTensorImpl weak;
  ProducerKind kind = ProducerKind::None;
  int64_t numel = 0;
  uint64_t version = 0;
};

constexpr size_t kMaxProducerRecords = 65536;
std::mutex g_producer_mutex;
std::unordered_map<c10::TensorImpl*, ProducerRecord> g_producer_records;

struct BridgeCounters {
  std::atomic<uint64_t> host_lease_hits{0};
  std::atomic<uint64_t> host_lease_misses{0};
  std::atomic<uint64_t> host_lease_hit_bytes{0};
  std::atomic<uint64_t> host_lease_miss_bytes{0};
  std::atomic<uint64_t> ssbo_to_cpu_materializations{0};
  std::atomic<uint64_t> ssbo_to_cpu_bytes{0};
  std::atomic<uint64_t> stock_vulkan_to_cpu_materializations{0};
  std::atomic<uint64_t> stock_vulkan_to_cpu_bytes{0};
  std::atomic<uint64_t> cpu_to_stock_vulkan_materializations{0};
  std::atomic<uint64_t> cpu_to_stock_vulkan_bytes{0};
  std::atomic<uint64_t> stock_vulkan_copy_calls{0};
  std::atomic<uint64_t> stock_vulkan_copy_from_ssbo{0};
};

BridgeCounters g_bridge_counters;

const char* producer_kind_name(ProducerKind kind) {
  switch (kind) {
    case ProducerKind::Conv1d: return "conv1d";
    case ProducerKind::Conv2d: return "conv2d";
    case ProducerKind::Conv2dSilu: return "conv2d_silu";
    case ProducerKind::Conv3d: return "conv3d";
    case ProducerKind::Matmul: return "matmul";
    case ProducerKind::LinearSilu: return "linear_silu";
    case ProducerKind::LinearGelu: return "linear_gelu";
    case ProducerKind::MatmulSoftmax: return "matmul_softmax";
    case ProducerKind::Softmax: return "softmax";
    case ProducerKind::LayerNorm: return "layernorm";
    case ProducerKind::GroupNorm: return "groupnorm";
    case ProducerKind::Silu: return "silu";
    case ProducerKind::Gelu: return "gelu";
    case ProducerKind::Sigmoid: return "sigmoid";
    case ProducerKind::Tanh: return "tanh";
    case ProducerKind::Binary: return "binary";
    case ProducerKind::Scalar: return "scalar";
    case ProducerKind::Upsample: return "upsample";
    case ProducerKind::Cat: return "cat";
    default: return "none";
  }
}

bool fallback_trace_enabled() {
  const char* env = std::getenv("VKSSBO_TRACE_FALLBACK");
  return env && env[0] != '\0' && env[0] != '0';
}

void trace_tensor_shape(const Tensor& tensor) {
  std::fputc('[', stderr);
  if (tensor.defined()) {
    for (int64_t i = 0; i < tensor.dim(); ++i) {
      if (i != 0) {
        std::fputc(',', stderr);
      }
      std::fprintf(stderr, "%lld", static_cast<long long>(tensor.size(i)));
    }
  }
  std::fputc(']', stderr);
}

void trace_matmul_softmax_fallback(
    const char* reason,
    const Tensor& query,
    const Tensor& key,
    double scale,
    bool transpose_key,
    const char* detail = nullptr) {
  if (!fallback_trace_enabled()) {
    return;
  }
  const char* dtype = "unknown";
  if (query.defined()) {
    dtype = query.scalar_type() == at::kHalf ? "f16"
        : query.scalar_type() == at::kFloat ? "f32" : "other";
  }
  std::fprintf(stderr, "[vkssbo] fallback op=matmul_softmax reason=%s q_shape=",
               reason);
  trace_tensor_shape(query);
  std::fputs(" k_shape=", stderr);
  trace_tensor_shape(key);
  std::fprintf(stderr, " dtype=%s transpose_key=%d scale=%.9g",
               dtype, transpose_key ? 1 : 0, scale);
  if (detail && detail[0] != '\0') {
    std::fprintf(stderr, " detail=%s", detail);
  }
  std::fputc('\n', stderr);
  std::fflush(stderr);
}

bool host_leases_enabled() {
  const char* env = std::getenv("VKSSBO_HOST_LEASE");
  // Resident SSBO reuse needs the CPU mirror pointer to survive the Vulkan
  // bridge. Lease tracking is version-checked and weak-TensorImpl-backed, so
  // it is safe by default; set VKSSBO_HOST_LEASE=0 for the legacy copy path.
  return !env || env[0] != '0';
}

uint64_t host_lease_version(const Tensor& t) {
  return t.is_inference() ? 0 : static_cast<uint64_t>(t._version());
}

void prune_host_leases_locked(
    std::unordered_map<c10::TensorImpl*, HostLease>& leases) {
  for (auto it = leases.begin(); it != leases.end();) {
    if (it->second.weak.expired()) {
      it = leases.erase(it);
    } else {
      ++it;
    }
  }
  if (leases.size() > kMaxHostLeases) {
    leases.clear();
  }
}

void prune_producer_records_locked() {
  for (auto it = g_producer_records.begin(); it != g_producer_records.end();) {
    if (it->second.weak.expired()) {
      it = g_producer_records.erase(it);
    } else {
      ++it;
    }
  }
  if (g_producer_records.size() > kMaxProducerRecords) {
    g_producer_records.clear();
  }
}

void remember_producer(const Tensor& t, ProducerKind kind) {
  if (!t.defined() || !t.is_vulkan()) return;
  std::lock_guard<std::mutex> lock(g_producer_mutex);
  prune_producer_records_locked();
  g_producer_records.insert_or_assign(
      t.unsafeGetTensorImpl(),
      ProducerRecord{
          WeakTensorImpl(t.getIntrusivePtr()), kind, t.numel(),
          t.is_inference() ? 0 : static_cast<uint64_t>(t._version())});
}

void remember_cpu_host_lease(const Tensor& t, std::shared_ptr<float> data,
                             int64_t numel) {
  if (!host_leases_enabled() || !t.defined() || !data) return;
  std::lock_guard<std::mutex> lock(g_host_lease_mutex);
  prune_host_leases_locked(g_cpu_host_leases);
  g_cpu_host_leases.insert_or_assign(
      t.unsafeGetTensorImpl(),
      HostLease{WeakTensorImpl(t.getIntrusivePtr()), std::move(data), numel,
                host_lease_version(t)});
}

std::shared_ptr<float> lookup_vulkan_host_lease(const Tensor& t) {
  if (!host_leases_enabled() || !t.defined() || !t.is_vulkan()) return {};
  std::lock_guard<std::mutex> lock(g_host_lease_mutex);
  auto it = g_vulkan_host_leases.find(t.unsafeGetTensorImpl());
  if (it == g_vulkan_host_leases.end()) {
    g_bridge_counters.host_lease_misses.fetch_add(1, std::memory_order_relaxed);
    g_bridge_counters.host_lease_miss_bytes.fetch_add(
        static_cast<uint64_t>(std::max<int64_t>(0, t.numel())) * sizeof(float),
        std::memory_order_relaxed);
    return {};
  }
  if (it->second.weak.expired() ||
      (!t.is_inference() && host_lease_version(t) != it->second.version)) {
    g_vulkan_host_leases.erase(it);
    g_bridge_counters.host_lease_misses.fetch_add(1, std::memory_order_relaxed);
    g_bridge_counters.host_lease_miss_bytes.fetch_add(
        static_cast<uint64_t>(std::max<int64_t>(0, t.numel())) * sizeof(float),
        std::memory_order_relaxed);
    return {};
  }
  g_bridge_counters.host_lease_hits.fetch_add(1, std::memory_order_relaxed);
  g_bridge_counters.host_lease_hit_bytes.fetch_add(
      static_cast<uint64_t>(std::max<int64_t>(0, t.numel())) * sizeof(float),
      std::memory_order_relaxed);
  return it->second.data;
}

std::shared_ptr<float> take_cpu_host_lease(const Tensor& t) {
  if (!host_leases_enabled() || !t.defined()) return {};
  std::lock_guard<std::mutex> lock(g_host_lease_mutex);
  auto it = g_cpu_host_leases.find(t.unsafeGetTensorImpl());
  if (it == g_cpu_host_leases.end()) return {};
  auto data = it->second.data;
  g_cpu_host_leases.erase(it);
  return data;
}

void remember_vulkan_host_lease(const Tensor& t, std::shared_ptr<float> data,
                                int64_t numel) {
  if (!host_leases_enabled() || !t.defined() || !t.is_vulkan() || !data) return;
  std::lock_guard<std::mutex> lock(g_host_lease_mutex);
  prune_host_leases_locked(g_vulkan_host_leases);
  g_vulkan_host_leases.insert_or_assign(
      t.unsafeGetTensorImpl(),
      HostLease{WeakTensorImpl(t.getIntrusivePtr()), std::move(data), numel,
                host_lease_version(t)});
}

struct TensorCacheIdentity {
  WeakTensorImpl weak;
  uint64_t serial;
};

constexpr size_t kMaxTensorCacheIdentities = 65536;
std::mutex g_tensor_cache_identity_mutex;
std::unordered_map<c10::TensorImpl*, TensorCacheIdentity>
    g_tensor_cache_identities;
std::atomic<uint64_t> g_next_tensor_cache_serial{1};

uint64_t mix_cache_key(uint64_t value) {
  value += 0x9e3779b97f4a7c15ull;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ull;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebull;
  return value ^ (value >> 31);
}

Tensor cpu_float(const Tensor& t) {
  Tensor x = t;
  if (x.is_vulkan()) {
    if (auto data = lookup_vulkan_host_lease(x)) {
      auto keepalive = data;
      x = at::from_blob(
          data.get(), x.sizes(),
          [keepalive](void*) mutable {},
          at::TensorOptions().dtype(at::kFloat));
    } else {
      g_bridge_counters.stock_vulkan_to_cpu_materializations.fetch_add(
          1, std::memory_order_relaxed);
      g_bridge_counters.stock_vulkan_to_cpu_bytes.fetch_add(
          static_cast<uint64_t>(std::max<int64_t>(0, x.numel())) * sizeof(float),
          std::memory_order_relaxed);
      x = x.to("cpu");
    }
  }
  if (x.dtype() != at::kFloat) x = x.to(at::kFloat);
  return x.contiguous();
}

Tensor to_vulkan(Tensor x) {
  Tensor contiguous = x.contiguous();
  std::shared_ptr<float> host_data = take_cpu_host_lease(x);
  if (!host_data && contiguous.unsafeGetTensorImpl() != x.unsafeGetTensorImpl()) {
    host_data = take_cpu_host_lease(contiguous);
  }
  Tensor out = contiguous.to(at::Device(at::kVulkan));
  g_bridge_counters.cpu_to_stock_vulkan_materializations.fetch_add(
      1, std::memory_order_relaxed);
  g_bridge_counters.cpu_to_stock_vulkan_bytes.fetch_add(
      static_cast<uint64_t>(std::max<int64_t>(0, out.numel())) * sizeof(float),
      std::memory_order_relaxed);
  if (host_data) {
    remember_vulkan_host_lease(out, std::move(host_data), out.numel());
  }
  return out;
}

Tensor to_vulkan_with_producer(Tensor x, ProducerKind kind) {
  Tensor out = to_vulkan(std::move(x));
  remember_producer(out, kind);
  return out;
}

uint64_t tensor_cache_key(const Tensor& t, uint64_t salt = 0) {
  c10::TensorImpl* const impl = t.unsafeGetTensorImpl();
  uint64_t serial = 0;
  {
    const std::lock_guard<std::mutex> lock(g_tensor_cache_identity_mutex);
    auto it = g_tensor_cache_identities.find(impl);
    if (it == g_tensor_cache_identities.end() || it->second.weak.expired()) {
      if (it != g_tensor_cache_identities.end()) {
        g_tensor_cache_identities.erase(it);
      }
      if (g_tensor_cache_identities.size() >= kMaxTensorCacheIdentities) {
        for (auto candidate = g_tensor_cache_identities.begin();
             candidate != g_tensor_cache_identities.end();) {
          if (candidate->second.weak.expired()) {
            candidate = g_tensor_cache_identities.erase(candidate);
          } else {
            ++candidate;
          }
        }
        if (g_tensor_cache_identities.size() >= kMaxTensorCacheIdentities) {
          g_tensor_cache_identities.clear();
        }
      }
      serial = g_next_tensor_cache_serial.fetch_add(1, std::memory_order_relaxed);
      g_tensor_cache_identities.emplace(
          impl,
          TensorCacheIdentity{WeakTensorImpl(t.getIntrusivePtr()), serial});
    } else {
      serial = it->second.serial;
    }
  }
  // Inference tensors intentionally omit a version counter.  They are common
  // for Vulkan inference workloads. The weak-reference-backed serial remains
  // unique even after a TensorImpl address is recycled.
  const uint64_t version = t.is_inference()
      ? 0
      : static_cast<uint64_t>(t._version());
  return mix_cache_key(serial) ^ mix_cache_key(version) ^ mix_cache_key(salt);
}

Tensor from_cpu(float* p, int64_t numel, IntArrayRef shape) {
  if (!p) return Tensor();
  // Share ownership with the lease cache. The temporary CPU Tensor and the
  // registered Vulkan Tensor each hold a reference, so the SSBO download is
  // released only after both the upload and all direct SSBO consumers finish.
  std::shared_ptr<float> owner(p, [](float* data) { vkssbo_free(data); });
  g_bridge_counters.ssbo_to_cpu_materializations.fetch_add(
      1, std::memory_order_relaxed);
  g_bridge_counters.ssbo_to_cpu_bytes.fetch_add(
      static_cast<uint64_t>(std::max<int64_t>(0, numel)) * sizeof(float),
      std::memory_order_relaxed);
  auto keepalive = owner;
  Tensor out = at::from_blob(
      p, shape,
      [keepalive](void*) mutable {},
      at::TensorOptions().dtype(at::kFloat));
  remember_cpu_host_lease(out, std::move(owner), numel);
  return out;
}

bool is_float_input(const Tensor& x) {
  return x.scalar_type() == at::kFloat || x.scalar_type() == at::kHalf;
}

} // namespace

bool enabled() {
  return g_enabled;
}

ProducerMetadata producer_metadata(const Tensor& tensor) {
  ProducerMetadata out;
  if (!tensor.defined() || !tensor.is_vulkan()) return out;
  std::lock_guard<std::mutex> lock(g_producer_mutex);
  auto it = g_producer_records.find(tensor.unsafeGetTensorImpl());
  if (it == g_producer_records.end()) return out;
  const ProducerRecord& record = it->second;
  const uint64_t version = tensor.is_inference()
      ? 0
      : static_cast<uint64_t>(tensor._version());
  if (record.weak.expired() || record.version != version) {
    g_producer_records.erase(it);
    return out;
  }
  out.valid = 1;
  out.kind = producer_kind_name(record.kind);
  out.numel = record.numel;
  out.version = static_cast<int64_t>(record.version);
  return out;
}

BridgeStats bridge_stats() {
  BridgeStats out;
  out.host_lease_enabled = host_leases_enabled() ? 1 : 0;
  // SSBO and stock Vulkan currently use independent VkDevice ownership.
  // Keep these fields explicit so a future compatible path cannot be enabled
  // accidentally by inferring compatibility from matching device properties.
  out.direct_copy_eligible = 0;
  out.direct_copy_enabled = 0;
  out.host_lease_hits = static_cast<int64_t>(
      g_bridge_counters.host_lease_hits.load(std::memory_order_relaxed));
  out.host_lease_misses = static_cast<int64_t>(
      g_bridge_counters.host_lease_misses.load(std::memory_order_relaxed));
  out.host_lease_hit_bytes = static_cast<int64_t>(
      g_bridge_counters.host_lease_hit_bytes.load(std::memory_order_relaxed));
  out.host_lease_miss_bytes = static_cast<int64_t>(
      g_bridge_counters.host_lease_miss_bytes.load(std::memory_order_relaxed));
  out.ssbo_to_cpu_materializations = static_cast<int64_t>(
      g_bridge_counters.ssbo_to_cpu_materializations.load(std::memory_order_relaxed));
  out.ssbo_to_cpu_bytes = static_cast<int64_t>(
      g_bridge_counters.ssbo_to_cpu_bytes.load(std::memory_order_relaxed));
  out.stock_vulkan_to_cpu_materializations = static_cast<int64_t>(
      g_bridge_counters.stock_vulkan_to_cpu_materializations.load(
          std::memory_order_relaxed));
  out.stock_vulkan_to_cpu_bytes = static_cast<int64_t>(
      g_bridge_counters.stock_vulkan_to_cpu_bytes.load(std::memory_order_relaxed));
  out.cpu_to_stock_vulkan_materializations = static_cast<int64_t>(
      g_bridge_counters.cpu_to_stock_vulkan_materializations.load(std::memory_order_relaxed));
  out.cpu_to_stock_vulkan_bytes = static_cast<int64_t>(
      g_bridge_counters.cpu_to_stock_vulkan_bytes.load(std::memory_order_relaxed));
  out.stock_vulkan_copy_calls = static_cast<int64_t>(
      g_bridge_counters.stock_vulkan_copy_calls.load(std::memory_order_relaxed));
  out.stock_vulkan_copy_from_ssbo = static_cast<int64_t>(
      g_bridge_counters.stock_vulkan_copy_from_ssbo.load(std::memory_order_relaxed));
  return out;
}

void bridge_reset() {
  g_bridge_counters.host_lease_hits.store(0, std::memory_order_relaxed);
  g_bridge_counters.host_lease_misses.store(0, std::memory_order_relaxed);
  g_bridge_counters.host_lease_hit_bytes.store(0, std::memory_order_relaxed);
  g_bridge_counters.host_lease_miss_bytes.store(0, std::memory_order_relaxed);
  g_bridge_counters.ssbo_to_cpu_materializations.store(0, std::memory_order_relaxed);
  g_bridge_counters.ssbo_to_cpu_bytes.store(0, std::memory_order_relaxed);
  g_bridge_counters.stock_vulkan_to_cpu_materializations.store(
      0, std::memory_order_relaxed);
  g_bridge_counters.stock_vulkan_to_cpu_bytes.store(0, std::memory_order_relaxed);
  g_bridge_counters.cpu_to_stock_vulkan_materializations.store(0, std::memory_order_relaxed);
  g_bridge_counters.cpu_to_stock_vulkan_bytes.store(0, std::memory_order_relaxed);
  g_bridge_counters.stock_vulkan_copy_calls.store(0, std::memory_order_relaxed);
  g_bridge_counters.stock_vulkan_copy_from_ssbo.store(0, std::memory_order_relaxed);
}

void record_stock_vulkan_copy(const Tensor& src, const Tensor& dst) {
  if (!src.defined() || !dst.defined() || !src.is_vulkan() || !dst.is_vulkan()) {
    return;
  }
  g_bridge_counters.stock_vulkan_copy_calls.fetch_add(1, std::memory_order_relaxed);
  const ProducerMetadata metadata = producer_metadata(src);
  if (metadata.valid) {
    g_bridge_counters.stock_vulkan_copy_from_ssbo.fetch_add(
        1, std::memory_order_relaxed);
  }
}

void init() {
  static std::once_flag once;
  std::call_once(once, [] {
    vkssbo_init(nullptr);
  });
}

MemoryStats memory_stats() {
  init();
  size_t heap_usage = 0;
  size_t heap_budget = 0;
  size_t pool_bytes = 0;
  size_t pool_limit = 0;
  size_t pool_entries = 0;
  uint64_t pool_hits = 0;
  uint64_t pool_misses = 0;
  size_t weight_bytes = 0;
  size_t weight_limit = 0;
  size_t weight_entries = 0;
  uint64_t weight_hits = 0;
  uint64_t weight_misses = 0;
  size_t resident_bytes = 0;
  size_t resident_limit = 0;
  size_t resident_entries = 0;
  uint64_t resident_hits = 0;
  uint64_t resident_misses = 0;
  size_t staging_bytes = 0;
  size_t staging_peak_bytes = 0;
  uint64_t staging_uploads = 0;
  uint64_t staging_downloads = 0;
  size_t staging_pool_bytes = 0;
  size_t staging_pool_entries = 0;
  uint64_t staging_pool_hits = 0;
  uint64_t staging_pool_misses = 0;
  int rust_policy = 0;
  vkssbo_memory_stats(&heap_usage, &heap_budget, &pool_bytes, &pool_limit,
                      &pool_entries, &pool_hits, &pool_misses, &weight_bytes,
                      &weight_limit, &weight_entries, &weight_hits,
                      &weight_misses, &rust_policy, &resident_bytes,
                      &resident_limit, &resident_entries, &resident_hits,
                      &resident_misses, &staging_bytes, &staging_peak_bytes,
                      &staging_uploads, &staging_downloads, &staging_pool_bytes,
                      &staging_pool_entries, &staging_pool_hits,
                      &staging_pool_misses);
  return MemoryStats{
      static_cast<int64_t>(heap_usage), static_cast<int64_t>(heap_budget),
      static_cast<int64_t>(pool_bytes), static_cast<int64_t>(pool_limit),
      static_cast<int64_t>(pool_entries), static_cast<int64_t>(pool_hits),
      static_cast<int64_t>(pool_misses), static_cast<int64_t>(weight_bytes),
      static_cast<int64_t>(weight_limit), static_cast<int64_t>(weight_entries),
      static_cast<int64_t>(weight_hits), static_cast<int64_t>(weight_misses),
      static_cast<int64_t>(rust_policy), static_cast<int64_t>(resident_bytes),
      static_cast<int64_t>(resident_limit), static_cast<int64_t>(resident_entries),
      static_cast<int64_t>(resident_hits), static_cast<int64_t>(resident_misses),
      static_cast<int64_t>(staging_bytes), static_cast<int64_t>(staging_peak_bytes),
      static_cast<int64_t>(staging_uploads), static_cast<int64_t>(staging_downloads),
      static_cast<int64_t>(staging_pool_bytes),
      static_cast<int64_t>(staging_pool_entries),
      static_cast<int64_t>(staging_pool_hits),
      static_cast<int64_t>(staging_pool_misses)};
}

DeviceInfo device_info() {
  init();
  uint32_t vendor_id = 0;
  uint32_t device_id = 0;
  uint32_t max_shared_memory = 0;
  uint32_t max_workgroup_invocations = 0;
  uint32_t shader_int8_supported = 0;
  uint32_t storage_buffer_8bit_supported = 0;
  uint32_t uniform_and_storage_buffer_8bit_supported = 0;
  uint32_t storage_push_constant_8bit_supported = 0;
  uint32_t memory_topology = 0;
  uint32_t has_device_local_only = 0;
  uint32_t has_host_visible_device_local = 0;
  uint32_t has_host_visible_system = 0;
  uint64_t device_local_budget = 0;
  uint64_t device_local_usage = 0;
  uint64_t device_local_only_budget = 0;
  uint64_t device_local_only_usage = 0;
  uint64_t host_visible_device_local_budget = 0;
  uint64_t host_visible_device_local_usage = 0;
  uint64_t host_visible_system_budget = 0;
  uint64_t host_visible_system_usage = 0;
  uint32_t driver_version = 0;
  uint32_t timestamp_valid_bits = 0;
  float timestamp_period_ns = 0.0f;
  uint32_t compute_queue_supported = 0;
  uint32_t compute_queue_family = 0;
  uint32_t compute_queue_flags = 0;
  uint32_t transfer_queue_supported = 0;
  int32_t transfer_queue_family = -1;
  uint32_t transfer_queue_flags = 0;
  uint32_t dedicated_transfer_queue_available = 0;
  uint32_t timeline_semaphore_supported = 0;
  uint32_t async_transfer_requested = 0;
  uint32_t async_transfer_eligible = 0;
  uint32_t async_transfer_enabled = 0;
  uint32_t async_transfer_resources_ready = 0;
  uint64_t async_transfer_min_bytes = 0;
  uint32_t physical_device_count = 0;
  uint32_t selected_device_index = 0;
  char device_selection[64]{};
  uint32_t simulation_enabled = 0;
  uint32_t simulation_force_staging_requested = 0;
  uint32_t simulation_force_staging = 0;
  uint32_t simulation_strict_budget = 0;
  uint64_t simulation_transfer_latency_us = 0;
  uint64_t simulated_vram_budget = 0;
  uint64_t simulated_vram_usage = 0;
  uint32_t policy_memory_topology = 0;
  char name[256]{};
  vkssbo_device_info(&vendor_id, &device_id, &max_shared_memory,
                     &max_workgroup_invocations, &shader_int8_supported,
                     &storage_buffer_8bit_supported,
                     &uniform_and_storage_buffer_8bit_supported,
                     &storage_push_constant_8bit_supported,
                     &memory_topology,
                     &has_device_local_only, &has_host_visible_device_local,
                     &has_host_visible_system, &device_local_budget,
                     &device_local_usage, &device_local_only_budget,
                     &device_local_only_usage,
                     &host_visible_device_local_budget,
                     &host_visible_device_local_usage,
                     &host_visible_system_budget, &host_visible_system_usage,
                     name, sizeof(name), &driver_version,
                     &timestamp_valid_bits, &timestamp_period_ns,
                     &compute_queue_supported, &compute_queue_family,
                     &compute_queue_flags, &transfer_queue_supported,
                     &transfer_queue_family, &transfer_queue_flags,
                     &dedicated_transfer_queue_available,
                     &timeline_semaphore_supported,
                     &async_transfer_requested, &async_transfer_eligible,
                     &async_transfer_enabled,
                     &async_transfer_resources_ready,
                     &async_transfer_min_bytes,
                     &physical_device_count, &selected_device_index,
                     device_selection, sizeof(device_selection),
                     &simulation_enabled,
                     &simulation_force_staging_requested,
                     &simulation_force_staging,
                     &simulation_strict_budget,
                     &simulation_transfer_latency_us,
                     &simulated_vram_budget,
                     &simulated_vram_usage,
                     &policy_memory_topology);
  DeviceInfo out;
  out.vendor_id = static_cast<int64_t>(vendor_id);
  out.device_id = static_cast<int64_t>(device_id);
  out.driver_version = static_cast<int64_t>(driver_version);
  out.timestamp_valid_bits = static_cast<int64_t>(timestamp_valid_bits);
  out.timestamp_period_ns = static_cast<double>(timestamp_period_ns);
  out.compute_queue_supported = static_cast<int64_t>(compute_queue_supported);
  out.compute_queue_family = static_cast<int64_t>(compute_queue_family);
  out.compute_queue_flags = static_cast<int64_t>(compute_queue_flags);
  out.transfer_queue_supported = static_cast<int64_t>(transfer_queue_supported);
  out.transfer_queue_family = static_cast<int64_t>(transfer_queue_family);
  out.transfer_queue_flags = static_cast<int64_t>(transfer_queue_flags);
  out.dedicated_transfer_queue_available =
      static_cast<int64_t>(dedicated_transfer_queue_available);
  out.timeline_semaphore_supported =
      static_cast<int64_t>(timeline_semaphore_supported);
  out.async_transfer_requested = static_cast<int64_t>(async_transfer_requested);
  out.async_transfer_eligible = static_cast<int64_t>(async_transfer_eligible);
  out.async_transfer_enabled = static_cast<int64_t>(async_transfer_enabled);
  out.async_transfer_resources_ready =
      static_cast<int64_t>(async_transfer_resources_ready);
  out.async_transfer_min_bytes =
      static_cast<int64_t>(async_transfer_min_bytes);
  out.physical_device_count = static_cast<int64_t>(physical_device_count);
  out.selected_device_index = static_cast<int64_t>(selected_device_index);
  out.device_selection = std::string(device_selection);
  out.max_shared_memory = static_cast<int64_t>(max_shared_memory);
  out.max_workgroup_invocations = static_cast<int64_t>(max_workgroup_invocations);
  out.shader_int8_supported = static_cast<int64_t>(shader_int8_supported);
  out.storage_buffer_8bit_supported =
      static_cast<int64_t>(storage_buffer_8bit_supported);
  out.uniform_and_storage_buffer_8bit_supported =
      static_cast<int64_t>(uniform_and_storage_buffer_8bit_supported);
  out.storage_push_constant_8bit_supported =
      static_cast<int64_t>(storage_push_constant_8bit_supported);
  out.int8_weight_only_enabled = 0;
  out.memory_topology = static_cast<int64_t>(memory_topology);
  out.has_device_local_only = static_cast<int64_t>(has_device_local_only);
  out.has_host_visible_device_local = static_cast<int64_t>(has_host_visible_device_local);
  out.has_host_visible_system = static_cast<int64_t>(has_host_visible_system);
  out.device_local_budget = static_cast<int64_t>(device_local_budget);
  out.device_local_usage = static_cast<int64_t>(device_local_usage);
  out.device_local_only_budget = static_cast<int64_t>(device_local_only_budget);
  out.device_local_only_usage = static_cast<int64_t>(device_local_only_usage);
  out.host_visible_device_local_budget = static_cast<int64_t>(host_visible_device_local_budget);
  out.host_visible_device_local_usage = static_cast<int64_t>(host_visible_device_local_usage);
  out.host_visible_system_budget = static_cast<int64_t>(host_visible_system_budget);
  out.host_visible_system_usage = static_cast<int64_t>(host_visible_system_usage);
  out.simulation_enabled = static_cast<int64_t>(simulation_enabled);
  out.simulation_force_staging_requested =
      static_cast<int64_t>(simulation_force_staging_requested);
  out.simulation_force_staging = static_cast<int64_t>(simulation_force_staging);
  out.simulation_strict_budget = static_cast<int64_t>(simulation_strict_budget);
  out.simulation_transfer_latency_us =
      static_cast<int64_t>(simulation_transfer_latency_us);
  out.simulated_vram_budget = static_cast<int64_t>(simulated_vram_budget);
  out.simulated_vram_usage = static_cast<int64_t>(simulated_vram_usage);
  out.policy_memory_topology = static_cast<int64_t>(policy_memory_topology);
  out.device_name = std::string(name);
  return out;
}

AutotuneStats matmul_autotune_stats() {
  size_t entries = 0;
  uint64_t cache_hits = 0;
  uint64_t cache_misses = 0;
  uint64_t tune_runs = 0;
  uint64_t tune_failures = 0;
  uint64_t benchmark_dispatches = 0;
  vkssbo_matmul_autotune_stats(
      &entries, &cache_hits, &cache_misses, &tune_runs, &tune_failures,
      &benchmark_dispatches);
  return AutotuneStats{
      static_cast<int64_t>(entries), static_cast<int64_t>(cache_hits),
      static_cast<int64_t>(cache_misses), static_cast<int64_t>(tune_runs),
      static_cast<int64_t>(tune_failures),
      static_cast<int64_t>(benchmark_dispatches)};
}

AutotuneStats conv2d_autotune_stats() {
  size_t entries = 0;
  uint64_t cache_hits = 0;
  uint64_t cache_misses = 0;
  uint64_t tune_runs = 0;
  uint64_t tune_failures = 0;
  uint64_t benchmark_dispatches = 0;
  vkssbo_conv2d_autotune_stats(
      &entries, &cache_hits, &cache_misses, &tune_runs, &tune_failures,
      &benchmark_dispatches);
  return AutotuneStats{
      static_cast<int64_t>(entries), static_cast<int64_t>(cache_hits),
      static_cast<int64_t>(cache_misses), static_cast<int64_t>(tune_runs),
      static_cast<int64_t>(tune_failures),
      static_cast<int64_t>(benchmark_dispatches)};
}

ProfileStats profile_stats() {
  uint32_t schema_version = 0;
  uint32_t enabled = 0;
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
  vkssbo_profile_stats(
      &schema_version, &enabled, &upload_host_ns, &upload_bytes, &upload_calls,
      &submit_wait_ns, &compute_submissions, &transfer_submissions,
      &copy_submissions, &fence_waits, &queue_wait_idle_count,
      &gpu_command_ns, &gpu_timestamp_submissions, &gpu_timestamp_failures,
      &dispatches,
      &download_host_ns, &download_transfer_wait_ns, &download_bytes,
      &download_submissions, &download_calls);
  return ProfileStats{
      static_cast<int64_t>(schema_version), static_cast<int64_t>(enabled),
      static_cast<int64_t>(upload_host_ns), static_cast<int64_t>(upload_bytes),
      static_cast<int64_t>(upload_calls), static_cast<int64_t>(submit_wait_ns),
      static_cast<int64_t>(compute_submissions),
      static_cast<int64_t>(transfer_submissions),
      static_cast<int64_t>(copy_submissions), static_cast<int64_t>(fence_waits),
      static_cast<int64_t>(queue_wait_idle_count),
      static_cast<int64_t>(gpu_command_ns),
      static_cast<int64_t>(gpu_timestamp_submissions),
      static_cast<int64_t>(gpu_timestamp_failures),
      static_cast<int64_t>(dispatches),
      static_cast<int64_t>(download_host_ns),
      static_cast<int64_t>(download_transfer_wait_ns),
      static_cast<int64_t>(download_bytes),
      static_cast<int64_t>(download_submissions),
      static_cast<int64_t>(download_calls)};
}

void profile_reset() {
  vkssbo_profile_reset();
}

SchedulerStats scheduler_stats() {
  uint64_t last_submitted = 0;
  uint64_t completed = 0;
  uint64_t in_flight = 0;
  uint64_t peak_in_flight = 0;
  uint64_t submits = 0;
  uint64_t polls = 0;
  uint64_t waits = 0;
  vkssbo_scheduler_stats(
      &last_submitted, &completed, &in_flight, &peak_in_flight, &submits,
      &polls, &waits);
  return SchedulerStats{
      static_cast<int64_t>(last_submitted), static_cast<int64_t>(completed),
      static_cast<int64_t>(in_flight), static_cast<int64_t>(peak_in_flight),
      static_cast<int64_t>(submits), static_cast<int64_t>(polls),
      static_cast<int64_t>(waits)};
}

void empty_cache() {
  init();
  vkssbo_memory_trim();
}

Tensor convolution(
    const Tensor& input,
    const Tensor& weight,
    const std::optional<Tensor>& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups) {
  if (!enabled() || !is_float_input(input) || input.dim() != 4 ||
      groups <= 0 || stride.size() != 2 || padding.size() != 2 ||
      dilation.size() != 2 || stride[0] <= 0 || stride[1] <= 0 ||
      dilation[0] <= 0 || dilation[1] <= 0) {
    return Tensor();
  }
  init();
  try {
    Tensor x = cpu_float(input);
    Tensor w = cpu_float(weight);
    Tensor b;
    if (bias && bias->defined()) {
      b = cpu_float(*bias);
    } else {
      b = at::zeros({w.size(0)}, at::kFloat);
    }
    int64_t B = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);
    int64_t O = w.size(0), KH = w.size(2), KW = w.size(3);
    if (w.dim() != 4 || C % groups != 0 || O % groups != 0 ||
        w.size(1) != C / groups ||
        (bias && bias->defined() && bias->numel() != O)) {
      return Tensor();
    }
    int64_t Ho =
        (H + 2 * padding[0] - dilation[0] * (KH - 1) - 1) / stride[0] + 1;
    int64_t Wo =
        (W + 2 * padding[1] - dilation[1] * (KW - 1) - 1) / stride[1] + 1;
    float* p = vkssbo_conv2d(
        x.data_ptr<float>(), w.data_ptr<float>(), b.data_ptr<float>(),
        (int)B, (int)C, (int)O, (int)H, (int)W, (int)Ho, (int)Wo,
        (int)KH, (int)KW, (int)stride[0], (int)stride[1],
        (int)dilation[0], (int)dilation[1], (int)padding[0],
        (int)padding[1], (int)groups,
        tensor_cache_key(weight),
        bias && bias->defined() ? tensor_cache_key(*bias) : 0);
    return to_vulkan_with_producer(
        from_cpu(p, B * O * Ho * Wo, {B, O, Ho, Wo}), ProducerKind::Conv2d);
  } catch (const std::exception&) {
    return Tensor();
  }
}

Tensor conv2d_silu(
    const Tensor& input,
    const Tensor& weight,
    const std::optional<Tensor>& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups) {
  if (!enabled() || !is_float_input(input) || input.dim() != 4 ||
      weight.dim() != 4 || groups <= 0 || stride.size() != 2 ||
      padding.size() != 2 || dilation.size() != 2 || stride[0] <= 0 ||
      stride[1] <= 0 || dilation[0] <= 0 || dilation[1] <= 0) {
    return Tensor();
  }
  init();
  try {
    Tensor x = cpu_float(input);
    Tensor w = cpu_float(weight);
    Tensor b = bias && bias->defined()
        ? cpu_float(*bias)
        : at::zeros({w.size(0)}, at::kFloat);
    const int64_t B = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);
    const int64_t O = w.size(0), KH = w.size(2), KW = w.size(3);
    if (C % groups != 0 || O % groups != 0 || w.size(1) != C / groups ||
        (bias && bias->defined() && bias->numel() != O)) {
      return Tensor();
    }
    const int64_t Ho =
        (H + 2 * padding[0] - dilation[0] * (KH - 1) - 1) / stride[0] + 1;
    const int64_t Wo =
        (W + 2 * padding[1] - dilation[1] * (KW - 1) - 1) / stride[1] + 1;
    float* p = vkssbo_conv2d_silu(
        x.data_ptr<float>(), w.data_ptr<float>(), b.data_ptr<float>(),
        static_cast<int>(B), static_cast<int>(C), static_cast<int>(O),
        static_cast<int>(H), static_cast<int>(W), static_cast<int>(Ho),
        static_cast<int>(Wo), static_cast<int>(KH), static_cast<int>(KW),
        static_cast<int>(stride[0]), static_cast<int>(stride[1]),
        static_cast<int>(dilation[0]), static_cast<int>(dilation[1]),
        static_cast<int>(padding[0]), static_cast<int>(padding[1]),
        static_cast<int>(groups), tensor_cache_key(weight),
        bias && bias->defined() ? tensor_cache_key(*bias) : 0);
    if (p) {
      return to_vulkan_with_producer(
          from_cpu(p, B * O * Ho * Wo, {B, O, Ho, Wo}),
          ProducerKind::Conv2dSilu);
    }
  } catch (const std::exception&) {
    // Fall through to the existing, independently verified unfused path.
  }
  Tensor conv = convolution(input, weight, bias, stride, padding, dilation, groups);
  return conv.defined()
      ? at::native::vulkan::ops::ssbo::silu(conv)
      : Tensor();
}

Tensor conv1d(
    const Tensor& input,
    const Tensor& weight,
    const std::optional<Tensor>& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups) {
  if (!enabled() || !is_float_input(input) || input.dim() != 3) return Tensor();
  init();
  try {
    Tensor x = cpu_float(input);
    Tensor w = cpu_float(weight);
    Tensor b = bias && bias->defined() ? cpu_float(*bias)
                                       : at::zeros({w.size(0)}, at::kFloat);
    if (stride.size() != 1 || padding.size() != 1 || dilation.size() != 1 ||
        stride[0] <= 0 || dilation[0] <= 0 || groups <= 0) {
      return Tensor();
    }
    int64_t B = x.size(0), C = x.size(1), L = x.size(2);
    int64_t O = w.size(0), KL = w.size(2);
    if (w.dim() != 3 || C % groups != 0 || O % groups != 0 ||
        w.size(1) != C / groups ||
        (bias && bias->defined() && bias->numel() != O)) {
      return Tensor();
    }
    int64_t Lo =
        (L + 2 * padding[0] - dilation[0] * (KL - 1) - 1) / stride[0] + 1;
    float* p = vkssbo_conv1d(
        x.data_ptr<float>(), w.data_ptr<float>(), b.data_ptr<float>(),
        (int)B, (int)C, (int)O, (int)L, (int)Lo, (int)KL,
        (int)stride[0], (int)padding[0], (int)dilation[0], (int)groups,
        tensor_cache_key(weight),
        bias && bias->defined() ? tensor_cache_key(*bias) : 0);
    return to_vulkan_with_producer(
        from_cpu(p, B * O * Lo, {B, O, Lo}), ProducerKind::Conv1d);
  } catch (const std::exception&) {
    return Tensor();
  }
}

Tensor conv3d(
    const Tensor& input,
    const Tensor& weight,
    const std::optional<Tensor>& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups) {
  if (!enabled() || !is_float_input(input) || input.dim() != 5) return Tensor();
  init();
  try {
    Tensor x = cpu_float(input);
    Tensor w = cpu_float(weight);
    Tensor b = bias && bias->defined() ? cpu_float(*bias)
                                       : at::zeros({w.size(0)}, at::kFloat);
    if ((stride.size() != 1 && stride.size() != 3) ||
        (padding.size() != 1 && padding.size() != 3) ||
        (dilation.size() != 1 && dilation.size() != 3)) return Tensor();
    auto expand3 = [](IntArrayRef values) {
      return std::array<int64_t, 3>{values[0], values.size() == 1 ? values[0] : values[1],
          values.size() == 1 ? values[0] : values[2]};
    };
    const auto s = expand3(stride), pp = expand3(padding), dd = expand3(dilation);
    for (int64_t value : s) if (value <= 0) return Tensor();
    for (int64_t value : dd) if (value <= 0) return Tensor();
    if (groups <= 0 || w.dim() != 5) return Tensor();
    int64_t sd = s[0], sh = s[1], sw = s[2];
    int64_t pd = pp[0], ph = pp[1], pw = pp[2];
    int64_t dil_d = dd[0], dil_h = dd[1], dil_w = dd[2];
    int64_t B = x.size(0), C = x.size(1), D = x.size(2), H = x.size(3), W = x.size(4);
    int64_t O = w.size(0), KD = w.size(2), KH = w.size(3), KW = w.size(4);
    if (C % groups != 0 || O % groups != 0 || w.size(1) != C / groups ||
        (bias && bias->defined() && bias->numel() != O)) {
      return Tensor();
    }
    int64_t Do = (D + 2 * pd - dil_d * (KD - 1) - 1) / sd + 1;
    int64_t Ho = (H + 2 * ph - dil_h * (KH - 1) - 1) / sh + 1;
    int64_t Wo = (W + 2 * pw - dil_w * (KW - 1) - 1) / sw + 1;
    float* p = vkssbo_conv3d(
        x.data_ptr<float>(), w.data_ptr<float>(), b.data_ptr<float>(),
        (int)B, (int)C, (int)O, (int)D, (int)H, (int)W,
        (int)Do, (int)Ho, (int)Wo,
        (int)KD, (int)KH, (int)KW,
        (int)sd, (int)sh, (int)sw, (int)pd, (int)ph, (int)pw,
        (int)dil_d, (int)dil_h, (int)dil_w, (int)groups,
        tensor_cache_key(weight),
        bias && bias->defined() ? tensor_cache_key(*bias) : 0);
    return to_vulkan_with_producer(
        from_cpu(p, B * O * Do * Ho * Wo, {B, O, Do, Ho, Wo}),
        ProducerKind::Conv3d);
  } catch (const std::exception& e) {
    if (std::getenv("VK_TRACE")) {
      std::fprintf(stderr, "[vkssbo] aten conv3d bridge failed: %s\n", e.what());
    }
    return Tensor();
  }
}

Tensor linear(const Tensor& input, const Tensor& weight, const std::optional<Tensor>& bias) {
  if (!enabled() || !is_float_input(input)) return Tensor();
  init();
  try {
    Tensor x = cpu_float(input);
    Tensor w = cpu_float(weight);
    Tensor b = bias && bias->defined() ? cpu_float(*bias) : Tensor();
    if (x.dim() < 2 || w.dim() != 2 || x.size(-1) != w.size(1)) {
      return Tensor();
    }
    // The SSBO matmul entry point is batched, while aten::linear accepts any
    // rank >= 2. Flatten all leading dimensions into rows and restore the
    // original rank with the output feature dimension below.
    int64_t B = 1;
    int64_t K = x.size(-1);
    int64_t M = x.numel() / K;
    int64_t N = w.size(0);
    float* p = nullptr;
    const uint64_t weight_key = tensor_cache_key(weight, 0x4d4dull);
    if (b.defined() && b.numel() != N) return Tensor();
    p = vkssbo_matmul_bias(
        x.data_ptr<float>(), w.data_ptr<float>(),
        b.defined() ? b.data_ptr<float>() : nullptr, (int)B, (int)M,
        (int)K, (int)N, 2, weight_key,
        b.defined() ? tensor_cache_key(*bias, 0x62696173ull) : 0);
    std::vector<int64_t> out_shape(x.sizes().begin(), x.sizes().end());
    out_shape.back() = N;
    Tensor out = from_cpu(p, B * M * N, out_shape);
    if (!out.defined()) return Tensor();
    return to_vulkan(out);
  } catch (const std::exception&) {
    return Tensor();
  }
}

Tensor linear_silu(
    const Tensor& input,
    const Tensor& weight,
    const std::optional<Tensor>& bias) {
  if (!enabled() || !is_float_input(input)) return Tensor();
  init();
  try {
    Tensor x = cpu_float(input);
    Tensor w = cpu_float(weight);
    Tensor b = bias && bias->defined() ? cpu_float(*bias) : Tensor();
    if (x.dim() < 2 || w.dim() != 2 || x.size(-1) != w.size(1)) {
      return Tensor();
    }
    const int64_t B = 1;
    const int64_t K = x.size(-1);
    const int64_t M = x.numel() / K;
    const int64_t N = w.size(0);
    if (b.defined() && b.numel() != N) return Tensor();

    const char* fusion_env = std::getenv("VKSSBO_MATMUL_SILU_FUSION");
    const bool fusion_requested = !fusion_env || fusion_env[0] != '0';
    if (!fusion_requested) {
      Tensor projected = ssbo::linear(input, weight, bias);
      return projected.defined() ? ssbo::silu(projected) : Tensor();
    }

    float* p = vkssbo_matmul_bias_silu(
        x.data_ptr<float>(), w.data_ptr<float>(),
        b.defined() ? b.data_ptr<float>() : nullptr,
        static_cast<int>(B), static_cast<int>(M), static_cast<int>(K),
        static_cast<int>(N), 2, tensor_cache_key(weight, 0x6c73696cull),
        b.defined() ? tensor_cache_key(*bias, 0x6c736962ull) : 0);
    std::vector<int64_t> out_shape(x.sizes().begin(), x.sizes().end());
    out_shape.back() = N;
    return to_vulkan_with_producer(
        from_cpu(p, B * M * N, out_shape), ProducerKind::LinearSilu);
  } catch (const std::exception&) {
    return Tensor();
  }
}

Tensor linear_gelu(
    const Tensor& input,
    const Tensor& weight,
    const std::optional<Tensor>& bias) {
  if (!enabled() || !is_float_input(input)) return Tensor();
  init();
  try {
    Tensor x = cpu_float(input);
    Tensor w = cpu_float(weight);
    Tensor b = bias && bias->defined() ? cpu_float(*bias) : Tensor();
    if (x.dim() < 2 || w.dim() != 2 || x.size(-1) != w.size(1)) {
      return Tensor();
    }
    const int64_t B = 1;
    const int64_t K = x.size(-1);
    const int64_t M = x.numel() / K;
    const int64_t N = w.size(0);
    if (b.defined() && b.numel() != N) return Tensor();

    const char* fusion_env = std::getenv("VKSSBO_MATMUL_GELU_FUSION");
    const bool fusion_requested = !fusion_env || fusion_env[0] != '0';
    if (!fusion_requested) {
      Tensor projected = ssbo::linear(input, weight, bias);
      return projected.defined() ? ssbo::gelu(projected) : Tensor();
    }

    float* p = vkssbo_matmul_bias_gelu(
        x.data_ptr<float>(), w.data_ptr<float>(),
        b.defined() ? b.data_ptr<float>() : nullptr,
        static_cast<int>(B), static_cast<int>(M), static_cast<int>(K),
        static_cast<int>(N), 2, tensor_cache_key(weight, 0x6c67656cull),
        b.defined() ? tensor_cache_key(*bias, 0x6c676562ull) : 0);
    std::vector<int64_t> out_shape(x.sizes().begin(), x.sizes().end());
    out_shape.back() = N;
    return to_vulkan_with_producer(
        from_cpu(p, B * M * N, out_shape), ProducerKind::LinearGelu);
  } catch (const std::exception&) {
    return Tensor();
  }
}

Tensor matmul_softmax(
    const Tensor& query,
    const Tensor& key,
    double scale,
    bool transpose_key) {
  if (!enabled()) {
    trace_matmul_softmax_fallback(
        "input_guard", query, key, scale, transpose_key,
        "backend_disabled");
    return Tensor();
  }
  if (!query.defined() || !key.defined() || !is_float_input(query) ||
      !is_float_input(key)) {
    trace_matmul_softmax_fallback(
        "input_guard", query, key, scale, transpose_key,
        "dtype_or_device");
    return Tensor();
  }
  if (query.dim() != 3 || key.dim() != 3) {
    trace_matmul_softmax_fallback(
        "input_guard", query, key, scale, transpose_key,
        "rank");
    return Tensor();
  }
  if (!transpose_key) {
    trace_matmul_softmax_fallback(
        "input_guard", query, key, scale, transpose_key,
        "transpose_key_required");
    return Tensor();
  }
  if (query.size(0) != key.size(0)) {
    trace_matmul_softmax_fallback(
        "input_guard", query, key, scale, transpose_key,
        "batch_mismatch");
    return Tensor();
  }
  if (query.size(2) != key.size(2)) {
    trace_matmul_softmax_fallback(
        "input_guard", query, key, scale, transpose_key,
        "feature_mismatch");
    return Tensor();
  }
  init();
  try {
    Tensor q = cpu_float(query);
    Tensor k = cpu_float(key);
    const int64_t B = q.size(0), M = q.size(1), K = q.size(2), N = k.size(1);
    if (scale == 0.0 || !std::isfinite(scale)) {
      trace_matmul_softmax_fallback(
          "scale_guard", query, key, scale, transpose_key,
          "non_finite_or_zero");
      return Tensor();
    }
    const char* env = std::getenv("VKSSBO_MATMUL_SOFTMAX_FUSION");
    const bool requested = !env || env[0] != '0';
    if (!requested) {
      Tensor kt = k.transpose(1, 2).contiguous();
      Tensor scores = ssbo::matmul(query, to_vulkan(kt));
      if (!scores.defined()) {
        trace_matmul_softmax_fallback(
            "native_undefined", query, key, scale, transpose_key,
            "unfused_matmul");
        return Tensor();
      }
      Tensor scaled = ssbo::elementwise_scalar(scores, scale, 2);
      if (!scaled.defined()) {
        trace_matmul_softmax_fallback(
            "native_undefined", query, key, scale, transpose_key,
            "unfused_scale");
        return Tensor();
      }
      Tensor result = ssbo::softmax(scaled, -1);
      if (!result.defined()) {
        trace_matmul_softmax_fallback(
            "native_undefined", query, key, scale, transpose_key,
            "unfused_softmax");
      }
      return result;
    }
    float* p = vkssbo_matmul_softmax(
        q.data_ptr<float>(), k.data_ptr<float>(), static_cast<int>(B),
        static_cast<int>(M), static_cast<int>(K), static_cast<int>(N),
        static_cast<float>(scale), tensor_cache_key(key, 0x6d736dull));
    if (!p) {
      trace_matmul_softmax_fallback(
          "native_null_fallback", query, key, scale, transpose_key,
          "fused_kernel_returned_null");
      Tensor kt = k.transpose(1, 2).contiguous();
      Tensor scores = ssbo::matmul(query, to_vulkan(kt));
      if (!scores.defined()) {
        trace_matmul_softmax_fallback(
            "native_undefined", query, key, scale, transpose_key,
            "unfused_matmul");
        return Tensor();
      }
      Tensor scaled = ssbo::elementwise_scalar(scores, scale, 2);
      if (!scaled.defined()) {
        trace_matmul_softmax_fallback(
            "native_undefined", query, key, scale, transpose_key,
            "unfused_scale");
        return Tensor();
      }
      Tensor result = ssbo::softmax(scaled, -1);
      if (!result.defined()) {
        trace_matmul_softmax_fallback(
            "native_undefined", query, key, scale, transpose_key,
            "unfused_softmax");
      }
      return result;
    }
    return to_vulkan_with_producer(
        from_cpu(p, B * M * N, {B, M, N}), ProducerKind::MatmulSoftmax);
  } catch (const std::exception& e) {
    trace_matmul_softmax_fallback(
        "exception_std", query, key, scale, transpose_key, e.what());
    return Tensor();
  } catch (...) {
    trace_matmul_softmax_fallback(
        "unknown_exception_rethrow", query, key, scale, transpose_key);
    throw;
  }
}

Tensor mm(const Tensor& a, const Tensor& b) {
  if (!enabled() || !is_float_input(a) || a.dim() != 2) return Tensor();
  init();
  try {
    Tensor x = cpu_float(a);
    Tensor w = cpu_float(b);
    int64_t M = x.size(0), K = x.size(1), N = w.size(1);
    float* p = vkssbo_matmul(x.data_ptr<float>(), w.data_ptr<float>(), 1,
                             (int)M, (int)K, (int)N, 0,
                             tensor_cache_key(b, 0x6d6dull));
    return to_vulkan_with_producer(
        from_cpu(p, M * N, {M, N}), ProducerKind::Matmul);
  } catch (const std::exception&) {
    return Tensor();
  }
}

Tensor bmm(const Tensor& a, const Tensor& b) {
  if (!enabled() || !is_float_input(a) || a.dim() != 3) return Tensor();
  init();
  try {
    Tensor x = cpu_float(a);
    Tensor w = cpu_float(b);
    int64_t B = x.size(0), M = x.size(1), K = x.size(2), N = w.size(2);
    float* p = vkssbo_matmul(x.data_ptr<float>(), w.data_ptr<float>(),
                             (int)B, (int)M, (int)K, (int)N, 0,
                             tensor_cache_key(b, 0x626d6dull));
    return to_vulkan_with_producer(
        from_cpu(p, B * M * N, {B, M, N}), ProducerKind::Matmul);
  } catch (const std::exception&) {
    return Tensor();
  }
}

Tensor matmul(const Tensor& a, const Tensor& b) {
  if (!enabled() || !is_float_input(a)) return Tensor();
  if (a.dim() == 2 && b.dim() == 2) return ssbo::mm(a, b);
  if (a.dim() == 3 && b.dim() == 3 && a.size(0) == b.size(0)) return ssbo::bmm(a, b);
  return Tensor();
}

Tensor softmax(const Tensor& x, int64_t dim) {
  if (!enabled() || !is_float_input(x)) return Tensor();
  init();
  try {
    Tensor t = cpu_float(x);
    int64_t ndim = t.dim();
    int64_t d = dim < 0 ? dim + ndim : dim;
    if (d != ndim - 1) return Tensor();
    int64_t cols = t.size(d);
    Tensor t2 = t.reshape({-1, cols});
    int64_t rows = t2.size(0);
    float* p = vkssbo_softmax(t2.data_ptr<float>(), (int)rows, (int)cols);
    return to_vulkan_with_producer(
        from_cpu(p, rows * cols, x.sizes()), ProducerKind::Softmax);
  } catch (const std::exception&) {
    return Tensor();
  }
}

Tensor layer_norm(
    const Tensor& x,
    IntArrayRef normalized_shape,
    const std::optional<Tensor>& w,
    const std::optional<Tensor>& b,
    double eps) {
  if (!enabled() || !is_float_input(x)) return Tensor();
  init();
  try {
    Tensor t = cpu_float(x);
    int64_t cols = 1;
    for (int64_t s : normalized_shape) cols *= s;
    if (cols <= 0 || (w && w->defined() && w->numel() != cols) ||
        (b && b->defined() && b->numel() != cols)) {
      return Tensor();
    }
    Tensor t2 = t.reshape({-1, cols});
    int64_t rows = t2.size(0);
    Tensor ww = w && w->defined() ? cpu_float(*w).contiguous() : Tensor();
    Tensor bb = b && b->defined() ? cpu_float(*b).contiguous() : Tensor();
    float* p = vkssbo_layernorm(
        t2.data_ptr<float>(),
        ww.defined() ? ww.data_ptr<float>() : nullptr,
        bb.defined() ? bb.data_ptr<float>() : nullptr,
        (int)rows, (int)cols, (float)eps, ww.defined() ? 1 : 0,
        bb.defined() ? 1 : 0,
        ww.defined() ? tensor_cache_key(*w) : 0,
        bb.defined() ? tensor_cache_key(*b) : 0);
    return to_vulkan_with_producer(
        from_cpu(p, rows * cols, x.sizes()), ProducerKind::LayerNorm);
  } catch (const std::exception&) {
    return Tensor();
  }
}

Tensor group_norm(
    const Tensor& x,
    int64_t num_groups,
    const std::optional<Tensor>& w,
    const std::optional<Tensor>& b,
    double eps) {
  if (!enabled() || !is_float_input(x)) return Tensor();
  init();
  try {
    Tensor t = cpu_float(x);
    if (t.dim() < 2) return Tensor();
    int64_t B = t.size(0), C = t.size(1);
    int64_t spatial = t.numel() / (B * C);
    if (num_groups <= 0 || C % num_groups != 0 ||
        (w && w->defined() && w->numel() != C) ||
        (b && b->defined() && b->numel() != C) ||
        spatial > std::numeric_limits<int>::max()) {
      return Tensor();
    }
    Tensor ww = w && w->defined() ? cpu_float(*w).contiguous() : Tensor();
    Tensor bb = b && b->defined() ? cpu_float(*b).contiguous() : Tensor();
    float* p = vkssbo_groupnorm(
        t.data_ptr<float>(),
        ww.defined() ? ww.data_ptr<float>() : nullptr,
        bb.defined() ? bb.data_ptr<float>() : nullptr,
        (int)B, (int)C, 1, (int)spatial, (int)num_groups, (float)eps,
        ww.defined() ? 1 : 0, bb.defined() ? 1 : 0,
        ww.defined() ? tensor_cache_key(*w) : 0,
        bb.defined() ? tensor_cache_key(*b) : 0);
    return to_vulkan_with_producer(
        from_cpu(p, t.numel(), x.sizes()), ProducerKind::GroupNorm);
  } catch (const std::exception&) {
    return Tensor();
  }
}

std::tuple<Tensor, Tensor, Tensor> native_group_norm(
    const Tensor& x,
    const std::optional<Tensor>& w,
    const std::optional<Tensor>& b,
    int64_t batch,
    int64_t channels,
    int64_t spatial,
    int64_t num_groups,
    double eps) {
  Tensor t = cpu_float(x);
  if (t.dim() < 2 || t.size(0) != batch || t.size(1) != channels ||
      t.numel() != batch * channels * spatial ||
      channels % num_groups != 0) {
    return std::make_tuple(Tensor(), Tensor(), Tensor());
  }

  Tensor output =
      at::native::vulkan::ops::ssbo::group_norm(x, num_groups, w, b, eps);
  if (!output.defined()) {
    return std::make_tuple(Tensor(), Tensor(), Tensor());
  }

  Tensor mean = at::empty({batch, num_groups}, t.options());
  Tensor rstd = at::empty({batch, num_groups}, t.options());
  const float* input = t.data_ptr<float>();
  float* mean_data = mean.data_ptr<float>();
  float* rstd_data = rstd.data_ptr<float>();
  const int64_t values_per_group = channels / num_groups * spatial;

  for (int64_t n = 0; n < batch; ++n) {
    for (int64_t g = 0; g < num_groups; ++g) {
      const int64_t stats_index = n * num_groups + g;
      const int64_t offset = stats_index * values_per_group;
      double sum = 0.0;
      double sum_sq = 0.0;
      for (int64_t i = 0; i < values_per_group; ++i) {
        const double value = input[offset + i];
        sum += value;
        sum_sq += value * value;
      }
      const double group_mean = sum / values_per_group;
      const double variance =
          std::max(0.0, sum_sq / values_per_group - group_mean * group_mean);
      mean_data[stats_index] = static_cast<float>(group_mean);
      rstd_data[stats_index] =
          static_cast<float>(1.0 / std::sqrt(variance + eps));
    }
  }

  return std::make_tuple(output, to_vulkan(mean), to_vulkan(rstd));
}

Tensor silu(const Tensor& x) {
  if (is_native_buffer(x)) return native_buffer_unary(x, OP_SILU);
  if (!enabled() || !is_float_input(x)) return Tensor();
  init();
  try {
    Tensor t = cpu_float(x);
    float* p = vkssbo_elementwise(OP_SILU, 0, t.data_ptr<float>(), nullptr, 0.0f, (int)t.numel());
    return to_vulkan_with_producer(
        from_cpu(p, t.numel(), x.sizes()), ProducerKind::Silu);
  } catch (const std::exception&) {
    return Tensor();
  }
}

Tensor gelu(const Tensor& x) {
  if (!enabled() || !is_float_input(x)) return Tensor();
  init();
  try {
    Tensor t = cpu_float(x);
    float* p = vkssbo_elementwise(OP_GELU, 0, t.data_ptr<float>(), nullptr, 0.0f, (int)t.numel());
    return to_vulkan_with_producer(
        from_cpu(p, t.numel(), x.sizes()), ProducerKind::Gelu);
  } catch (const std::exception&) {
    return Tensor();
  }
}

Tensor sigmoid(const Tensor& x) {
  if (is_native_buffer(x)) return native_buffer_unary(x, OP_SIGMOID);
  if (!enabled() || !is_float_input(x)) return Tensor();
  init();
  try {
    Tensor t = cpu_float(x);
    float* p = vkssbo_elementwise(OP_SIGMOID, 0, t.data_ptr<float>(), nullptr, 0.0f, (int)t.numel());
    return to_vulkan_with_producer(
        from_cpu(p, t.numel(), x.sizes()), ProducerKind::Sigmoid);
  } catch (const std::exception&) {
    return Tensor();
  }
}

Tensor tanh(const Tensor& x) {
  if (!enabled() || !is_float_input(x)) return Tensor();
  init();
  try {
    Tensor t = cpu_float(x);
    float* p = vkssbo_elementwise(OP_TANH, 0, t.data_ptr<float>(), nullptr, 0.0f, (int)t.numel());
    return to_vulkan_with_producer(
        from_cpu(p, t.numel(), x.sizes()), ProducerKind::Tanh);
  } catch (const std::exception&) {
    return Tensor();
  }
}

Tensor elementwise_binary(const Tensor& a, const Tensor& b, int64_t op) {
  if (!enabled() || !is_float_input(a) || a.numel() != b.numel()) return Tensor();
  init();
  try {
    Tensor x = cpu_float(a);
    Tensor y = cpu_float(b);
    float* p = vkssbo_elementwise((int)op, 1, x.data_ptr<float>(), y.data_ptr<float>(), 0.0f, (int)x.numel());
    return to_vulkan_with_producer(
        from_cpu(p, x.numel(), a.sizes()), ProducerKind::Binary);
  } catch (const std::exception&) {
    return Tensor();
  }
}

Tensor elementwise_scalar(const Tensor& a, double scalar, int64_t op) {
  if (!enabled() || !is_float_input(a)) return Tensor();
  init();
  try {
    Tensor x = cpu_float(a);
    float* p = vkssbo_elementwise((int)op, 0, x.data_ptr<float>(), nullptr, (float)scalar, (int)x.numel());
    return to_vulkan_with_producer(
        from_cpu(p, x.numel(), a.sizes()), ProducerKind::Scalar);
  } catch (const std::exception&) {
    return Tensor();
  }
}

Tensor upsample(
    const Tensor& x,
    IntArrayRef output_size,
    bool bilinear,
    bool align_corners) {
  if (!enabled() || !is_float_input(x) || x.dim() != 4 ||
      output_size.size() != 2) {
    return Tensor();
  }
  init();
  try {
    Tensor t = cpu_float(x);
    int64_t B = t.size(0), C = t.size(1), H = t.size(2), W = t.size(3);
    int64_t Ho = output_size[0], Wo = output_size[1];
    const int mode = bilinear ? (align_corners ? 2 : 1) : 0;
    float* p = vkssbo_upsample(t.data_ptr<float>(), (int)B, (int)C, (int)H, (int)W,
                               (int)Ho, (int)Wo, mode);
    return to_vulkan_with_producer(
        from_cpu(p, B * C * Ho * Wo, {B, C, Ho, Wo}), ProducerKind::Upsample);
  } catch (const std::exception&) {
    return Tensor();
  }
}

Tensor cat(const TensorList& tensors, int64_t dim) {
  if (!enabled() || dim != 1) return Tensor();
  init();
  try {
    std::vector<Tensor> ts;
    int64_t B = -1, H = -1, W = -1;
    for (const Tensor& t : tensors) {
      Tensor x = cpu_float(t);
      if (x.dim() != 4) return Tensor();
      if (B == -1) {
        B = x.size(0);
        H = x.size(2);
        W = x.size(3);
      } else if (x.size(0) != B || x.size(2) != H || x.size(3) != W) {
        return Tensor();
      }
      ts.push_back(x);
    }
    if (ts.empty() || ts.size() > 4) return Tensor();
    int64_t C0 = ts.size() > 0 ? ts[0].size(1) : 0;
    int64_t C1 = ts.size() > 1 ? ts[1].size(1) : 0;
    int64_t C2 = ts.size() > 2 ? ts[2].size(1) : 0;
    int64_t C3 = ts.size() > 3 ? ts[3].size(1) : 0;
    int64_t O = C0 + C1 + C2 + C3;
    const float* pa = ts.size() > 0 ? ts[0].data_ptr<float>() : nullptr;
    const float* pb = ts.size() > 1 ? ts[1].data_ptr<float>() : nullptr;
    const float* pc = ts.size() > 2 ? ts[2].data_ptr<float>() : nullptr;
    const float* pd = ts.size() > 3 ? ts[3].data_ptr<float>() : nullptr;
    float* p = vkssbo_cat4(pa, pb, pc, pd, (int)B, (int)H, (int)W,
                           (int)C0, (int)C1, (int)C2, (int)C3, (int)O);
    return to_vulkan_with_producer(
        from_cpu(p, B * O * H * W, {B, O, H, W}), ProducerKind::Cat);
  } catch (const std::exception&) {
    return Tensor();
  }
}

namespace {

template <typename FN>
Tensor try_elem(const Tensor& x, int64_t op, FN&& fail) {
  if (!enabled() || !is_float_input(x)) return fail();
  return elementwise_scalar(x, 0.0, op);
}

Tensor wrap_elem_binary(const Tensor& a, const Tensor& b, int64_t op) {
  return elementwise_binary(a, b, op);
}

} // namespace

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl("native_group_norm", [](
      const Tensor& x,
      const std::optional<Tensor>& w,
      const std::optional<Tensor>& b,
      c10::SymInt batch,
      c10::SymInt channels,
      c10::SymInt spatial,
      int64_t num_groups,
      double eps) -> std::tuple<Tensor, Tensor, Tensor> {
    return ssbo::native_group_norm(
        x,
        w,
        b,
        batch.expect_int(),
        channels.expect_int(),
        spatial.expect_int(),
        num_groups,
        eps);
  });
  m.impl("silu", [](const Tensor& x) -> Tensor { return ssbo::silu(x); });
  m.impl("conv3d", [](const Tensor& x, const Tensor& w, const std::optional<Tensor>& b,
                      IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation,
                      int64_t groups) -> Tensor {
    return ssbo::conv3d(x, w, b, stride, padding, dilation, groups);
  });
}

TORCH_LIBRARY(vulkan_ssbo, m) {
  m.def("native_buffer_stats() -> Dict(str, int)", []() {
    const auto& counters = at::native::vulkan::ops::native_buffer_counters();
    c10::Dict<std::string, int64_t> stats;
    stats.insert("uploads", counters.uploads.load());
    stats.insert("downloads", counters.downloads.load());
    stats.insert("unary_dispatches", counters.unary_dispatches.load());
    stats.insert("device_copies", counters.device_copies.load());
    return stats;
  });
  // B3 capability declaration.  This is deliberately metadata-only: it must
  // not create a Vulkan context or expose an opaque handle as a Storage.
  // Applications can query it before moving a tensor to Vulkan and choose a
  // supported path instead of discovering an OpaqueTensorImpl limitation via
  // a late storage()/data_ptr() exception.
  m.def("native_buffer_capabilities() -> Dict(str, str)", []() {
    c10::Dict<std::string, std::string> capabilities;
    const bool enabled =
        at::native::vulkan::ops::native_buffer_requested();
    capabilities.insert("enabled", enabled ? "1" : "0");
    capabilities.insert("representation", enabled ? "linear_fp32_buffer"
                                                   : "legacy_texture");
    capabilities.insert("device", "vulkan");
    capabilities.insert("dtype", "float32");
    capabilities.insert("rank", "1..4");
    capabilities.insert("max_numel", "65535");
    capabilities.insert("strides", "positive");
    capabilities.insert("storage", "opaque");
    capabilities.insert("storage_api", "unsupported");
    capabilities.insert("untyped_storage_api", "unsupported");
    capabilities.insert("data_ptr", "unsupported");
    capabilities.insert("requires_grad", "unsupported");
    capabilities.insert("backward", "unsupported");
    capabilities.insert("copy", "fp32_positive_stride");
    capabilities.insert("clone", "contiguous_only_for_strided");
    capabilities.insert("as_strided", "unsupported");
    return capabilities;
  });
  m.def("is_native_buffer(Tensor x) -> bool", [](const Tensor& x) {
    return at::native::vulkan::ops::is_native_buffer(x);
  });
  m.def("memory_stats() -> Dict(str, int)");
  m.def("device_info() -> Dict(str, str)");
  m.def("matmul_autotune_stats() -> Dict(str, int)");
  m.def("conv2d_autotune_stats() -> Dict(str, int)");
  m.def("profile_stats() -> Dict(str, int)");
  m.def("profile_reset() -> ()");
  m.def("scheduler_stats() -> Dict(str, int)");
  m.def("producer_metadata(Tensor tensor) -> Dict(str, str)");
  m.def("bridge_stats() -> Dict(str, int)");
  m.def("bridge_reset() -> ()");
  m.def(
      "conv2d_silu(Tensor input, Tensor weight, Tensor? bias, int[] stride, "
      "int[] padding, int[] dilation, int groups) -> Tensor");
  m.def("linear_silu(Tensor input, Tensor weight, Tensor? bias=None) -> Tensor");
  m.def("linear_gelu(Tensor input, Tensor weight, Tensor? bias=None) -> Tensor");
  m.def("matmul_softmax(Tensor query, Tensor key, float scale, bool transpose_key=True) -> Tensor");
  m.def("empty_cache() -> ()");
  m.impl("memory_stats", []() {
    const MemoryStats s = ssbo::memory_stats();
    c10::Dict<std::string, int64_t> out;
    out.insert("heap_usage", s.heap_usage);
    out.insert("heap_budget", s.heap_budget);
    out.insert("pool_bytes", s.pool_bytes);
    out.insert("pool_limit", s.pool_limit);
    out.insert("pool_entries", s.pool_entries);
    out.insert("pool_hits", s.pool_hits);
    out.insert("pool_misses", s.pool_misses);
    out.insert("weight_bytes", s.weight_bytes);
    out.insert("weight_limit", s.weight_limit);
    out.insert("weight_entries", s.weight_entries);
    out.insert("weight_hits", s.weight_hits);
    out.insert("weight_misses", s.weight_misses);
    out.insert("rust_policy", s.rust_policy);
    out.insert("resident_bytes", s.resident_bytes);
    out.insert("resident_limit", s.resident_limit);
    out.insert("resident_entries", s.resident_entries);
    out.insert("resident_hits", s.resident_hits);
    out.insert("resident_misses", s.resident_misses);
    out.insert("staging_bytes", s.staging_bytes);
    out.insert("staging_peak_bytes", s.staging_peak_bytes);
    out.insert("staging_uploads", s.staging_uploads);
    out.insert("staging_downloads", s.staging_downloads);
    out.insert("staging_pool_bytes", s.staging_pool_bytes);
    out.insert("staging_pool_entries", s.staging_pool_entries);
    out.insert("staging_pool_hits", s.staging_pool_hits);
    out.insert("staging_pool_misses", s.staging_pool_misses);
    return out;
  });
  m.impl("device_info", []() {
    const DeviceInfo info = ssbo::device_info();
    c10::Dict<std::string, std::string> out;
    out.insert("vendor_id", std::to_string(info.vendor_id));
    out.insert("device_id", std::to_string(info.device_id));
    out.insert("driver_version", std::to_string(info.driver_version));
    out.insert("timestamp_valid_bits", std::to_string(info.timestamp_valid_bits));
    out.insert("timestamp_period_ns", std::to_string(info.timestamp_period_ns));
    out.insert("compute_queue_supported",
               std::to_string(info.compute_queue_supported));
    out.insert("compute_queue_family", std::to_string(info.compute_queue_family));
    out.insert("compute_queue_flags", std::to_string(info.compute_queue_flags));
    out.insert("transfer_queue_supported",
               std::to_string(info.transfer_queue_supported));
    out.insert("transfer_queue_family",
               std::to_string(info.transfer_queue_family));
    out.insert("transfer_queue_flags", std::to_string(info.transfer_queue_flags));
    out.insert("dedicated_transfer_queue_available",
               std::to_string(info.dedicated_transfer_queue_available));
    out.insert("timeline_semaphore_supported",
               std::to_string(info.timeline_semaphore_supported));
    out.insert("async_transfer_requested",
               std::to_string(info.async_transfer_requested));
    out.insert("async_transfer_eligible",
               std::to_string(info.async_transfer_eligible));
    out.insert("async_transfer_enabled",
               std::to_string(info.async_transfer_enabled));
    out.insert("async_transfer_resources_ready",
               std::to_string(info.async_transfer_resources_ready));
    out.insert("async_transfer_min_bytes",
               std::to_string(info.async_transfer_min_bytes));
    out.insert("physical_device_count",
               std::to_string(info.physical_device_count));
    out.insert("selected_device_index",
               std::to_string(info.selected_device_index));
    out.insert("device_selection", info.device_selection);
    out.insert("max_shared_memory", std::to_string(info.max_shared_memory));
    out.insert("max_workgroup_invocations",
               std::to_string(info.max_workgroup_invocations));
    out.insert("shader_int8_supported",
               std::to_string(info.shader_int8_supported));
    out.insert("storage_buffer_8bit_supported",
               std::to_string(info.storage_buffer_8bit_supported));
    out.insert("uniform_and_storage_buffer_8bit_supported",
               std::to_string(info.uniform_and_storage_buffer_8bit_supported));
    out.insert("storage_push_constant_8bit_supported",
               std::to_string(info.storage_push_constant_8bit_supported));
    out.insert("int8_weight_only_enabled",
               std::to_string(info.int8_weight_only_enabled));
    static const char* topology_names[] = {
        "unknown", "uma", "discrete", "mixed"};
    const int64_t topology =
        info.memory_topology >= 0 && info.memory_topology <= 3
        ? info.memory_topology : 0;
    out.insert("memory_topology", topology_names[topology]);
    out.insert("has_device_local_only",
               std::to_string(info.has_device_local_only));
    out.insert("has_host_visible_device_local",
               std::to_string(info.has_host_visible_device_local));
    out.insert("has_host_visible_system",
               std::to_string(info.has_host_visible_system));
    out.insert("device_local_budget",
               std::to_string(info.device_local_budget));
    out.insert("device_local_usage", std::to_string(info.device_local_usage));
    out.insert("device_local_only_budget",
               std::to_string(info.device_local_only_budget));
    out.insert("device_local_only_usage",
               std::to_string(info.device_local_only_usage));
    out.insert("host_visible_device_local_budget",
               std::to_string(info.host_visible_device_local_budget));
    out.insert("host_visible_device_local_usage",
               std::to_string(info.host_visible_device_local_usage));
    out.insert("host_visible_system_budget",
               std::to_string(info.host_visible_system_budget));
    out.insert("host_visible_system_usage",
               std::to_string(info.host_visible_system_usage));
    out.insert("simulation_enabled",
               std::to_string(info.simulation_enabled));
    out.insert("simulation_force_staging_requested",
               std::to_string(info.simulation_force_staging_requested));
    out.insert("simulation_force_staging",
               std::to_string(info.simulation_force_staging));
    out.insert("simulation_strict_budget",
               std::to_string(info.simulation_strict_budget));
    out.insert("simulation_transfer_latency_us",
               std::to_string(info.simulation_transfer_latency_us));
    out.insert("simulated_vram_budget",
               std::to_string(info.simulated_vram_budget));
    out.insert("simulated_vram_usage",
               std::to_string(info.simulated_vram_usage));
    static const char* policy_topology_names[] = {
        "unknown", "uma", "discrete", "mixed"};
    const int64_t policy_topology =
        info.policy_memory_topology >= 0 && info.policy_memory_topology <= 3
        ? info.policy_memory_topology : 0;
    out.insert("policy_memory_topology",
               policy_topology_names[policy_topology]);
    out.insert("device_name", info.device_name);
    return out;
  });
  m.impl("matmul_autotune_stats", []() {
    const AutotuneStats s = ssbo::matmul_autotune_stats();
    c10::Dict<std::string, int64_t> out;
    out.insert("entries", s.entries);
    out.insert("cache_hits", s.cache_hits);
    out.insert("cache_misses", s.cache_misses);
    out.insert("tune_runs", s.tune_runs);
    out.insert("tune_failures", s.tune_failures);
    out.insert("benchmark_dispatches", s.benchmark_dispatches);
    return out;
  });
  m.impl("conv2d_autotune_stats", []() {
    const AutotuneStats s = ssbo::conv2d_autotune_stats();
    c10::Dict<std::string, int64_t> out;
    out.insert("entries", s.entries);
    out.insert("cache_hits", s.cache_hits);
    out.insert("cache_misses", s.cache_misses);
    out.insert("tune_runs", s.tune_runs);
    out.insert("tune_failures", s.tune_failures);
    out.insert("benchmark_dispatches", s.benchmark_dispatches);
    return out;
  });
  m.impl("profile_stats", []() {
    const ProfileStats s = ssbo::profile_stats();
    c10::Dict<std::string, int64_t> out;
    out.insert("schema_version", s.schema_version);
    out.insert("enabled", s.enabled);
    out.insert("upload_host_ns", s.upload_host_ns);
    out.insert("upload_bytes", s.upload_bytes);
    out.insert("upload_calls", s.upload_calls);
    out.insert("submit_wait_ns", s.submit_wait_ns);
    out.insert("compute_submissions", s.compute_submissions);
    out.insert("transfer_submissions", s.transfer_submissions);
    out.insert("copy_submissions", s.copy_submissions);
    out.insert("fence_waits", s.fence_waits);
    out.insert("queue_wait_idle_count", s.queue_wait_idle_count);
    out.insert("gpu_command_ns", s.gpu_command_ns);
    out.insert("gpu_timestamp_submissions", s.gpu_timestamp_submissions);
    out.insert("gpu_timestamp_failures", s.gpu_timestamp_failures);
    out.insert("dispatches", s.dispatches);
    out.insert("download_host_ns", s.download_host_ns);
    out.insert("download_transfer_wait_ns", s.download_transfer_wait_ns);
    out.insert("download_bytes", s.download_bytes);
    out.insert("download_submissions", s.download_submissions);
    out.insert("download_calls", s.download_calls);
    return out;
  });
  m.impl("profile_reset", []() {
    ssbo::profile_reset();
  });
  m.impl("scheduler_stats", []() {
    const SchedulerStats s = ssbo::scheduler_stats();
    c10::Dict<std::string, int64_t> out;
    out.insert("last_submitted", s.last_submitted);
    out.insert("completed", s.completed);
    out.insert("in_flight", s.in_flight);
    out.insert("peak_in_flight", s.peak_in_flight);
    out.insert("submits", s.submits);
    out.insert("polls", s.polls);
    out.insert("waits", s.waits);
    return out;
  });
  m.impl("bridge_stats", []() {
    const BridgeStats s = ssbo::bridge_stats();
    c10::Dict<std::string, int64_t> out;
    out.insert("schema_version", s.schema_version);
    out.insert("host_lease_enabled", s.host_lease_enabled);
    out.insert("direct_copy_eligible", s.direct_copy_eligible);
    out.insert("direct_copy_enabled", s.direct_copy_enabled);
    out.insert("host_lease_hits", s.host_lease_hits);
    out.insert("host_lease_misses", s.host_lease_misses);
    out.insert("host_lease_hit_bytes", s.host_lease_hit_bytes);
    out.insert("host_lease_miss_bytes", s.host_lease_miss_bytes);
    out.insert("ssbo_to_cpu_materializations", s.ssbo_to_cpu_materializations);
    out.insert("ssbo_to_cpu_bytes", s.ssbo_to_cpu_bytes);
    out.insert("stock_vulkan_to_cpu_materializations",
               s.stock_vulkan_to_cpu_materializations);
    out.insert("stock_vulkan_to_cpu_bytes", s.stock_vulkan_to_cpu_bytes);
    out.insert("cpu_to_stock_vulkan_materializations",
               s.cpu_to_stock_vulkan_materializations);
    out.insert("cpu_to_stock_vulkan_bytes", s.cpu_to_stock_vulkan_bytes);
    out.insert("stock_vulkan_copy_calls", s.stock_vulkan_copy_calls);
    out.insert("stock_vulkan_copy_from_ssbo", s.stock_vulkan_copy_from_ssbo);
    return out;
  });
  m.impl("bridge_reset", []() { ssbo::bridge_reset(); });
  m.impl("producer_metadata", [](const Tensor& tensor) {
    const ProducerMetadata info = ssbo::producer_metadata(tensor);
    c10::Dict<std::string, std::string> out;
    out.insert("valid", std::to_string(info.valid));
    out.insert("kind", info.kind);
    out.insert("numel", std::to_string(info.numel));
    out.insert("version", std::to_string(info.version));
    return out;
  });
  m.impl("conv2d_silu", [](const Tensor& input, const Tensor& weight,
                            const std::optional<Tensor>& bias,
                            IntArrayRef stride, IntArrayRef padding,
                            IntArrayRef dilation, int64_t groups) {
    return ssbo::conv2d_silu(
        input, weight, bias, stride, padding, dilation, groups);
  });
  m.impl("linear_silu", [](const Tensor& input, const Tensor& weight,
                            const std::optional<Tensor>& bias) {
    return ssbo::linear_silu(input, weight, bias);
  });
  m.impl("linear_gelu", [](const Tensor& input, const Tensor& weight,
                            const std::optional<Tensor>& bias) {
    return ssbo::linear_gelu(input, weight, bias);
  });
  m.impl("matmul_softmax", [](const Tensor& query, const Tensor& key,
                               double scale, bool transpose_key) {
    return ssbo::matmul_softmax(query, key, scale, transpose_key);
  });
  m.impl("empty_cache", []() { ssbo::empty_cache(); });
}

} // namespace ssbo
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
