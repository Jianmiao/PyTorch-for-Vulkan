#include <ATen/native/vulkan/ops/Factory.h>
#include <ATen/native/vulkan/ops/NativeBuffer.h>
#include <c10/core/Contiguity.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

Tensor _empty_affine_quantized(
    const IntArrayRef sizes,
    const std::optional<ScalarType> dtype,
    const std::optional<c10::Layout> layout,
    const std::optional<Device> device,
    const std::optional<bool> pin_memory,
    const double scale,
    const int64_t zero_point,
    const std::optional<MemoryFormat> memory_format) {
  api::StorageType storage_type = api::StorageType::TEXTURE_3D;
  return convert_quantized(vTensor{
      api::context(),
      sizes.vec(),
      scale,
      zero_point,
      convert_dtype(dtype ? *dtype : c10::kFloat),
      storage_type,
      memory_format ? get_gpu_memory_layout(storage_type, *memory_format)
                    : api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED,
  });
}

static Tensor empty_memory_format(
    const IntArrayRef sizes,
    const std::optional<ScalarType> dtype,
    const std::optional<c10::Layout> layout,
    const std::optional<Device> device,
    const std::optional<bool> pin_memory,
    const std::optional<MemoryFormat> memory_format) {
  // Texture storage encodes NCHW in a 3D image and is limited to four
  // logical dimensions.  Keep higher-rank tensors in a linear Vulkan buffer;
  // the buffer metadata ABI folds leading dimensions, so SSBO operators can
  // still preserve the original logical shape.
  if (native_buffer_requested()) {
    TORCH_CHECK(dtype.value_or(kFloat) == kFloat,
                "Vulkan native buffer B1 only supports float32");
    TORCH_CHECK(layout.value_or(kStrided) == kStrided && !pin_memory.value_or(false),
                "Vulkan native buffer requires strided, unpinned storage");
    TORCH_CHECK(!device || device->index() <= 0,
                "Vulkan native buffer B1 supports the default device only");
    TORCH_CHECK(memory_format.value_or(MemoryFormat::Contiguous) == MemoryFormat::Contiguous,
                "Vulkan native buffer B1 only supports contiguous memory format");
    int64_t count = 1;
    TORCH_CHECK(!sizes.empty() && sizes.size() <= 4,
                "Vulkan native buffer B1 supports ranks 1 through 4");
    for (const int64_t size : sizes) {
      TORCH_CHECK(size > 0 && size <= 65535 / count,
                  "Vulkan native buffer B1 requires 1..65535 elements");
      count *= size;
    }
  }
  const bool use_buffer = native_buffer_requested() || sizes.size() > 4;
  api::StorageType storage_type =
      use_buffer ? api::StorageType::BUFFER : api::StorageType::TEXTURE_3D;
  const c10::MemoryFormat effective_memory_format =
      use_buffer ? c10::MemoryFormat::Contiguous
                 : (memory_format ? *memory_format : c10::MemoryFormat::Contiguous);
  return convert(vTensor{
      api::context(),
      sizes.vec(),
      convert_dtype(dtype ? *dtype : c10::kFloat),
      storage_type,
      use_buffer ? api::GPUMemoryLayout::TENSOR_WIDTH_PACKED
                 : get_gpu_memory_layout(storage_type, effective_memory_format),
  });
}

static Tensor empty_strided(
    const IntArrayRef sizes,
    const IntArrayRef strides,
    const std::optional<ScalarType> dtype,
    const std::optional<c10::Layout> layout,
    const std::optional<Device> device,
    const std::optional<bool> pin_memory) {
  if (native_buffer_requested()) {
    TORCH_CHECK(sizes.size() == strides.size() &&
                    c10::_compute_contiguous<int64_t>(sizes, strides, 1),
                "Vulkan native buffer B1 does not support non-contiguous empty_strided");
  }
  return empty_memory_format(
      sizes, dtype, layout, device, pin_memory, c10::MemoryFormat::Contiguous);
}

#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("aten::empty.memory_format"),
      at::native::vulkan::ops::empty_memory_format);
  m.impl(
      TORCH_SELECTIVE_NAME("aten::_empty_affine_quantized"),
      at::native::vulkan::ops::_empty_affine_quantized);
  m.impl(
      TORCH_SELECTIVE_NAME("aten::empty_strided"),
      TORCH_FN(at::native::vulkan::ops::empty_strided));
}

#endif /* USE_VULKAN_API */

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
