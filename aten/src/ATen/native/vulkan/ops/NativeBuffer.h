#pragma once

#include <ATen/native/vulkan/ops/Common.h>
#include <atomic>
#include <cstdlib>
#include <cstring>

namespace at::native::vulkan::ops {

struct NativeBufferCounters {
  std::atomic<int64_t> uploads{0};
  std::atomic<int64_t> downloads{0};
  std::atomic<int64_t> unary_dispatches{0};
  std::atomic<int64_t> device_copies{0};
};

inline NativeBufferCounters& native_buffer_counters() {
  static NativeBufferCounters counters;
  return counters;
}

inline bool native_buffer_requested() {
  static const bool requested = [] {
    const char* value = std::getenv("VK_NATIVE_BUFFER");
    return value && std::strcmp(value, "1") == 0;
  }();
  return requested;
}

inline bool is_native_buffer(const Tensor& tensor) {
  return native_buffer_requested() && tensor.is_vulkan() &&
      convert(tensor).storage_type() == api::StorageType::BUFFER;
}

inline bool is_native_buffer_storage(const vTensor& tensor) {
  return native_buffer_requested() && tensor.storage_type() == api::StorageType::BUFFER;
}

inline std::vector<int64_t> native_contiguous_strides(const std::vector<int64_t>& sizes) {
  std::vector<int64_t> strides(sizes.size());
  int64_t stride = 1;
  for (size_t i = sizes.size(); i-- > 0;) {
    strides[i] = stride;
    stride *= sizes[i];
  }
  return strides;
}

// Copy/gather/scatter and unary kernels share a logical-index metadata ABI.
// Callers establish resource barriers before passing the buffers here.
inline bool native_strided_dispatch(
    api::Context* context, api::PipelineBarrier& barrier,
    api::VulkanBuffer& dst, const api::VulkanBuffer& src,
    const std::vector<int64_t>& sizes, const std::vector<int64_t>& dst_strides,
    const std::vector<int64_t>& src_strides, int64_t dst_offset, int64_t src_offset,
    uint32_t operation = 0, VkFence fence = VK_NULL_HANDLE) {
  using namespace api::utils;
  TORCH_CHECK(!sizes.empty() && sizes.size() <= 4 &&
                  sizes.size() == src_strides.size() && sizes.size() == dst_strides.size(),
              "Vulkan native strided dispatch requires ranks 1 through 4");
  const auto count = multiply_integers(sizes);
  TORCH_CHECK(count > 0 && count <= 65535, "Vulkan native strided dispatch exceeds limit");
  struct Block {
    uvec4 sizes;
    uvec4 input_strides;
    uvec4 output_strides;
    uvec4 info;
  };
  api::UniformParamsBuffer params(context, Block{
      make_whcn_uvec4(sizes), make_whcn_uvec4(src_strides), make_whcn_uvec4(dst_strides),
      {safe_downcast<uint32_t>(count), safe_downcast<uint32_t>(src_offset),
       safe_downcast<uint32_t>(dst_offset), operation}});
  return context->submit_compute_job(VK_KERNEL(native_buffer_strided), barrier,
      uvec3{static_cast<uint32_t>(count), 1u, 1u}, uvec3{64u, 1u, 1u}, fence,
      dst, src, params.buffer());
}

// FP32 buffers owned by the stock context, with validated positive strides.
// No standalone SSBO runtime or CPU mirror is used. Submission and waits
// still follow the existing stock context policy.
inline Tensor native_buffer_unary(const Tensor& input, uint32_t operation) {
  TORCH_CHECK(input.scalar_type() == kFloat && !input.requires_grad(),
              "Vulkan native buffer unary supports FP32 inference only");
  TORCH_CHECK(operation == 5 || operation == 7,
              "Vulkan native buffer unary operation is not supported");
  const vTensor& src = convert(input);
  TORCH_CHECK(src.gpu_memory_layout() == api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
              "Vulkan native buffer unary requires contiguous layout");
  TORCH_CHECK(src.gpu_numel() > 0 && src.gpu_numel() <= 65535,
              "Vulkan native buffer unary exceeds the bounded B1 limit");
  api::Context* context = api::context();
  vTensor output{context, src.sizes(), src.dtype(), api::StorageType::BUFFER,
                 src.gpu_memory_layout()};
  api::PipelineBarrier barrier{};
  auto& destination = output.buffer(barrier, api::PipelineStage::COMPUTE,
                                     api::MemoryAccessType::WRITE);
  const auto& source = src.buffer(barrier, api::PipelineStage::COMPUTE);
  native_strided_dispatch(context, barrier, destination, source, src.sizes(),
                          output.strides(), src.strides(), 0, src.storage_offset(), operation);
  native_buffer_counters().unary_dispatches++;
  return convert(output);
}

} // namespace at::native::vulkan::ops
