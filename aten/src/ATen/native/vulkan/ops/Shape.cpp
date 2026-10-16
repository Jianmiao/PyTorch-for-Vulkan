#include <ATen/InferSize.h>
#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/Utils.h>
#include <ATen/native/vulkan/ops/NativeBuffer.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

static Tensor view_internal(const Tensor& self_arg, const IntArrayRef shape) {
  api::Context* const context = api::context();

  Tensor self = self_arg.is_vulkan() ? self_arg : self_arg.vulkan();
  vTensor& v_self = convert(self);
  if (is_native_buffer(self)) {
    TORCH_CHECK(!self.requires_grad(), "Vulkan native buffer view supports inference only");
    const auto inferred_size = at::infer_size_dv(shape, self.numel());
    return convert(v_self.buffer_view(IntArrayRef(inferred_size).vec()));
  }
  TORCH_CHECK(v_self.storage_type() != api::StorageType::BUFFER,
              "Vulkan native buffer view/alias is not implemented yet");

  if (std::getenv("VK_TRACE")) {
    std::string s;
    for (auto x : shape) s += std::to_string(x) + ",";
    std::fprintf(
        stderr, "[view] self=%lld,%lld,%lld,%lld -> %s\n",
        (long long)(self_arg.dim() > 0 ? self_arg.size(0) : 1),
        (long long)(self_arg.dim() > 1 ? self_arg.size(1) : 1),
        (long long)(self_arg.dim() > 2 ? self_arg.size(2) : 1),
        (long long)(self_arg.dim() > 3 ? self_arg.size(3) : 1),
        s.c_str());
  }

  at::DimVector inferred_size = at::infer_size_dv(shape, self.numel());
  IntArrayRef output_size(inferred_size);

  vTensor v_output{
      context,
      output_size.vec(),
      v_self.dtype(),
  };
  if (v_self.is_quantized()) {
    v_output.set_is_quantized();
    v_output.set_scale(v_self.get_scale());
    v_output.set_zero_point(v_self.get_zero_point());
  }

  api::StorageBuffer buffer(
      context, v_self.dtype(), v_self.gpu_numel(), true);

  utils::pack_vtensor_to_staging(v_self, buffer.buffer());

  api::PipelineBarrier pipeline_barrier{};
  add_buffer_barrier(
      pipeline_barrier,
      buffer.buffer(),
      // Previous access
      api::PipelineStage::COMPUTE,
      api::MemoryAccessType::WRITE,
      // Next access
      api::PipelineStage::COMPUTE,
      api::MemoryAccessType::READ);

  utils::pack_buffer_to_vtensor(buffer.buffer(), v_output, pipeline_barrier);

  return convert(v_output);
}

inline Tensor view(const Tensor& self_arg, IntArrayRef shape) {
  return view_internal(self_arg, shape);
}

static Tensor squeeze_dim(const Tensor& self, int64_t dim) {
  const int64_t ndim = self.dim();
  dim = c10::maybe_wrap_dim(dim, ndim);
  if (is_native_buffer(self)) {
    TORCH_CHECK(!self.requires_grad(), "Vulkan native squeeze supports inference only");
    auto sizes = self.sizes().vec();
    auto strides = self.strides().vec();
    if (sizes[dim] == 1) {
      sizes.erase(sizes.begin() + dim);
      strides.erase(strides.begin() + dim);
    }
    return convert(convert(self).buffer_strided_view(sizes, strides, self.storage_offset()));
  }
  if (self.size(dim) != 1) {
    return is_native_buffer(self) ? view_internal(self, self.sizes()) : self;
  }
  std::vector<int64_t> output_size(self.sizes().begin(), self.sizes().end());
  output_size.erase(output_size.begin() + dim);
  return view_internal(self, output_size);
}

static Tensor squeeze(const Tensor& self) {
  if (is_native_buffer(self)) {
    TORCH_CHECK(!self.requires_grad(), "Vulkan native squeeze supports inference only");
    std::vector<int64_t> sizes, strides;
    for (int64_t i = 0; i < self.dim(); ++i) {
      if (self.size(i) != 1) {
        sizes.push_back(self.size(i)); strides.push_back(self.stride(i));
      }
    }
    return convert(convert(self).buffer_strided_view(sizes, strides, self.storage_offset()));
  }
  std::vector<int64_t> output_size;
  output_size.reserve(self.dim());
  for (const int64_t size : self.sizes()) {
    if (size != 1) output_size.push_back(size);
  }
  return view_internal(self, output_size);
}

static Tensor _reshape_alias(
    const Tensor& self_arg,
    const IntArrayRef shape,
    const IntArrayRef strides) {
  if (is_native_buffer(self_arg)) {
    TORCH_CHECK(shape.size() == strides.size(), "Vulkan reshape_alias sizes/strides mismatch");
    const auto output = view_internal(self_arg, shape);
    TORCH_CHECK(output.strides() == strides,
                "Vulkan reshape_alias only supports canonical contiguous strides");
    return output;
  }
  return view_internal(self_arg, shape);
}

static Tensor alias(const Tensor& self) {
  TORCH_CHECK(is_native_buffer(self), "Vulkan alias currently requires native buffer mode");
  TORCH_CHECK(!self.requires_grad(), "Vulkan native alias supports inference only");
  return convert(convert(self).buffer_strided_view(self.sizes().vec(), self.strides().vec(), self.storage_offset()));
}

#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::alias"), TORCH_FN(alias));
  m.impl(TORCH_SELECTIVE_NAME("aten::view"), TORCH_FN(view));
  m.impl(TORCH_SELECTIVE_NAME("aten::squeeze"), TORCH_FN(squeeze));
  m.impl(TORCH_SELECTIVE_NAME("aten::squeeze.dim"), TORCH_FN(squeeze_dim));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::_reshape_alias"), TORCH_FN(_reshape_alias));
}

#endif /* USE_VULKAN_API */

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
