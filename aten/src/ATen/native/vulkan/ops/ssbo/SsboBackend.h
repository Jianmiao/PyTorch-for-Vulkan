// SSBO Vulkan backend bridge for the stock torch Vulkan device kernels.
// Each entry returns a defined tensor when the SSBO path succeeded, or an
// undefined tensor so the caller can fall back to the stock texture path.
#pragma once
#include <ATen/ATen.h>
#include <string>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace ssbo {

struct MemoryStats {
  int64_t heap_usage = 0;
  int64_t heap_budget = 0;
  int64_t pool_bytes = 0;
  int64_t pool_limit = 0;
  int64_t pool_entries = 0;
  int64_t pool_hits = 0;
  int64_t pool_misses = 0;
  int64_t weight_bytes = 0;
  int64_t weight_limit = 0;
  int64_t weight_entries = 0;
  int64_t weight_hits = 0;
  int64_t weight_misses = 0;
  int64_t rust_policy = 0;
  int64_t resident_bytes = 0;
  int64_t resident_limit = 0;
  int64_t resident_entries = 0;
  int64_t resident_hits = 0;
  int64_t resident_misses = 0;
  int64_t staging_bytes = 0;
  int64_t staging_peak_bytes = 0;
  int64_t staging_uploads = 0;
  int64_t staging_downloads = 0;
  int64_t staging_pool_bytes = 0;
  int64_t staging_pool_entries = 0;
  int64_t staging_pool_hits = 0;
  int64_t staging_pool_misses = 0;
};

struct DeviceInfo {
  int64_t vendor_id = 0;
  int64_t device_id = 0;
  int64_t driver_version = 0;
  int64_t timestamp_valid_bits = 0;
  double timestamp_period_ns = 0.0;
  int64_t compute_queue_supported = 0;
  int64_t compute_queue_family = -1;
  int64_t compute_queue_flags = 0;
  int64_t transfer_queue_supported = 0;
  int64_t transfer_queue_family = -1;
  int64_t transfer_queue_flags = 0;
  int64_t dedicated_transfer_queue_available = 0;
  int64_t timeline_semaphore_supported = 0;
  int64_t async_transfer_requested = 0;
  int64_t async_transfer_eligible = 0;
  int64_t async_transfer_enabled = 0;
  int64_t async_transfer_resources_ready = 0;
  int64_t async_transfer_min_bytes = 0;
  int64_t physical_device_count = 0;
  int64_t selected_device_index = 0;
  std::string device_selection = "first";
  int64_t max_shared_memory = 0;
  int64_t max_workgroup_invocations = 0;
  int64_t shader_int8_supported = 0;
  int64_t storage_buffer_8bit_supported = 0;
  int64_t uniform_and_storage_buffer_8bit_supported = 0;
  int64_t storage_push_constant_8bit_supported = 0;
  int64_t int8_weight_only_enabled = 0;
  int64_t memory_topology = 0;
  int64_t has_device_local_only = 0;
  int64_t has_host_visible_device_local = 0;
  int64_t has_host_visible_system = 0;
  int64_t device_local_budget = 0;
  int64_t device_local_usage = 0;
  int64_t device_local_only_budget = 0;
  int64_t device_local_only_usage = 0;
  int64_t host_visible_device_local_budget = 0;
  int64_t host_visible_device_local_usage = 0;
  int64_t host_visible_system_budget = 0;
  int64_t host_visible_system_usage = 0;
  int64_t simulation_enabled = 0;
  int64_t simulation_force_staging_requested = 0;
  int64_t simulation_force_staging = 0;
  int64_t simulation_strict_budget = 0;
  int64_t simulation_transfer_latency_us = 0;
  int64_t simulated_vram_budget = 0;
  int64_t simulated_vram_usage = 0;
  int64_t policy_memory_topology = 0;
  std::string device_name;
};

struct AutotuneStats {
  int64_t entries = 0;
  int64_t cache_hits = 0;
  int64_t cache_misses = 0;
  int64_t tune_runs = 0;
  int64_t tune_failures = 0;
  int64_t benchmark_dispatches = 0;
};

// Opt-in structured profiling for the SSBO bridge.
struct ProfileStats {
  int64_t schema_version = 2;
  int64_t enabled = 0;
  int64_t upload_host_ns = 0;
  int64_t upload_bytes = 0;
  int64_t upload_calls = 0;
  int64_t submit_wait_ns = 0;
  int64_t compute_submissions = 0;
  int64_t transfer_submissions = 0;
  int64_t copy_submissions = 0;
  int64_t fence_waits = 0;
  int64_t queue_wait_idle_count = 0;
  int64_t gpu_command_ns = 0;
  int64_t gpu_timestamp_submissions = 0;
  int64_t gpu_timestamp_failures = 0;
  int64_t dispatches = 0;
  int64_t download_host_ns = 0;
  int64_t download_transfer_wait_ns = 0;
  int64_t download_bytes = 0;
  int64_t download_submissions = 0;
  int64_t download_calls = 0;
};

// Completion-value lifecycle counters for the SSBO command scheduler.
// These are diagnostic-only and intentionally do not expose Vulkan handles.
struct SchedulerStats {
  int64_t last_submitted = 0;
  int64_t completed = 0;
  int64_t in_flight = 0;
  int64_t peak_in_flight = 0;
  int64_t submits = 0;
  int64_t polls = 0;
  int64_t waits = 0;
};

struct ProducerMetadata {
  int64_t valid = 0;
  std::string kind = "none";
  int64_t numel = 0;
  int64_t version = 0;
};

// Observable boundary between the standalone SSBO device and stock Vulkan
// tensor storage. The two runtimes currently own different VkDevice objects,
// so direct storage sharing remains explicitly disabled.
struct BridgeStats {
  int64_t schema_version = 2;
  int64_t host_lease_enabled = 0;
  int64_t direct_copy_eligible = 0;
  int64_t direct_copy_enabled = 0;
  int64_t host_lease_hits = 0;
  int64_t host_lease_misses = 0;
  int64_t host_lease_hit_bytes = 0;
  int64_t host_lease_miss_bytes = 0;
  int64_t ssbo_to_cpu_materializations = 0;
  int64_t ssbo_to_cpu_bytes = 0;
  int64_t stock_vulkan_to_cpu_materializations = 0;
  int64_t stock_vulkan_to_cpu_bytes = 0;
  int64_t cpu_to_stock_vulkan_materializations = 0;
  int64_t cpu_to_stock_vulkan_bytes = 0;
  int64_t stock_vulkan_copy_calls = 0;
  int64_t stock_vulkan_copy_from_ssbo = 0;
};

bool enabled();
void init();

Tensor convolution(
    const Tensor& input,
    const Tensor& weight,
    const std::optional<Tensor>& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups);
Tensor conv2d_silu(
    const Tensor& input,
    const Tensor& weight,
    const std::optional<Tensor>& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups);
Tensor conv1d(
    const Tensor& input,
    const Tensor& weight,
    const std::optional<Tensor>& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups);
Tensor conv3d(
    const Tensor& input,
    const Tensor& weight,
    const std::optional<Tensor>& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups);
Tensor linear(const Tensor& input, const Tensor& weight, const std::optional<Tensor>& bias);
Tensor linear_silu(
    const Tensor& input,
    const Tensor& weight,
    const std::optional<Tensor>& bias);
Tensor linear_gelu(
    const Tensor& input,
    const Tensor& weight,
    const std::optional<Tensor>& bias);
Tensor matmul_softmax(
    const Tensor& query,
    const Tensor& key,
    double scale,
    bool transpose_key = true);
Tensor mm(const Tensor& a, const Tensor& b);
Tensor bmm(const Tensor& a, const Tensor& b);
Tensor matmul(const Tensor& a, const Tensor& b);
Tensor softmax(const Tensor& x, int64_t dim);
Tensor layer_norm(
    const Tensor& x,
    IntArrayRef normalized_shape,
    const std::optional<Tensor>& w,
    const std::optional<Tensor>& b,
    double eps);
Tensor group_norm(
    const Tensor& x,
    int64_t num_groups,
    const std::optional<Tensor>& w,
    const std::optional<Tensor>& b,
    double eps);
std::tuple<Tensor, Tensor, Tensor> native_group_norm(
    const Tensor& x,
    const std::optional<Tensor>& w,
    const std::optional<Tensor>& b,
    int64_t batch,
    int64_t channels,
    int64_t spatial,
    int64_t num_groups,
    double eps);
Tensor silu(const Tensor& x);
Tensor gelu(const Tensor& x);
Tensor sigmoid(const Tensor& x);
Tensor tanh(const Tensor& x);
Tensor elementwise_binary(const Tensor& a, const Tensor& b, int64_t op);
Tensor elementwise_scalar(const Tensor& a, double scalar, int64_t op);
Tensor upsample(
    const Tensor& x,
    IntArrayRef output_size,
    bool bilinear,
    bool align_corners = false);
Tensor cat(const TensorList& tensors, int64_t dim);
MemoryStats memory_stats();
DeviceInfo device_info();
AutotuneStats matmul_autotune_stats();
AutotuneStats conv2d_autotune_stats();
ProfileStats profile_stats();
void profile_reset();
SchedulerStats scheduler_stats();
ProducerMetadata producer_metadata(const Tensor& tensor);
BridgeStats bridge_stats();
void bridge_reset();
void record_stock_vulkan_copy(const Tensor& src, const Tensor& dst);
void empty_cache();


} // namespace ssbo
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
