#pragma once

#include <ATen/OpaqueTensorImpl.h>
#include <c10/core/Contiguity.h>

#include <algorithm>
#include <utility>

namespace at {
// Buffer strides are logical strides; legacy texture storage remains opaque.
// Shallow copies must keep this derived type and its Vulkan stride metadata.
template <typename OpaqueHandle>
struct VulkanOpaqueTensorImpl : public OpaqueTensorImpl<OpaqueHandle> {
  VulkanOpaqueTensorImpl(
      at::DispatchKeySet key_set,
      const caffe2::TypeMeta data_type,
      c10::Device device,
      OpaqueHandle opaque_handle,
      c10::IntArrayRef sizes,
      c10::IntArrayRef strides,
      int64_t offset = 0)
      : OpaqueTensorImpl<OpaqueHandle>(
            key_set,
            data_type,
            device,
            opaque_handle,
            sizes,
            false),
        strides_(strides.vec()) {
    TensorImpl::set_storage_offset(offset);
  }

  IntArrayRef strides_custom() const override {
    return strides_;
  }

  SymIntArrayRef sym_strides_custom() const override {
    return c10::fromIntArrayRefKnownNonNegative(strides_);
  }

  c10::intrusive_ptr<TensorImpl> shallow_copy_and_detach(
      const c10::VariableVersion& version_counter,
      bool allow_tensor_metadata_change) const override {
    return shallow_copy_impl(version_counter, allow_tensor_metadata_change);
  }

  c10::intrusive_ptr<TensorImpl> shallow_copy_and_detach(
      c10::VariableVersion&& version_counter,
      bool allow_tensor_metadata_change) const override {
    return shallow_copy_impl(std::move(version_counter), allow_tensor_metadata_change);
  }

  void shallow_copy_from(const c10::intrusive_ptr<TensorImpl>& impl) override {
    const auto* source = dynamic_cast<const VulkanOpaqueTensorImpl*>(impl.get());
    TORCH_CHECK(source, "Expected Vulkan TensorImpl for shallow copy");
    OpaqueTensorImpl<OpaqueHandle>::copy_tensor_metadata(
        source, this, this->version_counter(), this->allow_tensor_metadata_change());
    strides_ = source->strides_;
    this->refresh_numel();
  }

  c10::SymBool sym_is_contiguous_custom(
      c10::MemoryFormat memory_format) const override {
    // Buffers expose logical strides; legacy textures have all-zero strides.
    if (std::any_of(strides_.begin(), strides_.end(),
                    [](int64_t stride) { return stride != 0; })) {
      const auto sizes = this->sizes();
      const IntArrayRef strides(strides_);
      switch (memory_format) {
        case c10::MemoryFormat::Contiguous:
          return c10::_compute_contiguous<int64_t>(sizes, strides, this->numel());
        case c10::MemoryFormat::ChannelsLast:
          return c10::_compute_channels_last_contiguous_2d<int64_t>(sizes, strides);
        case c10::MemoryFormat::ChannelsLast3d:
          return c10::_compute_channels_last_contiguous_3d<int64_t>(sizes, strides);
        default:
          return false;
      }
    }
    return true;
  }

 private:
  template <typename Version>
  c10::intrusive_ptr<TensorImpl> shallow_copy_impl(
      Version&& version_counter, bool allow_tensor_metadata_change) const {
    auto result = c10::make_intrusive<VulkanOpaqueTensorImpl>(
        this->key_set(), this->dtype(), this->device(), this->opaque_handle(),
        this->sizes(), strides_custom());
    OpaqueTensorImpl<OpaqueHandle>::copy_tensor_metadata(
        this, result.get(), std::forward<Version>(version_counter),
        allow_tensor_metadata_change);
    result->refresh_numel();
    return result;
  }

  const char* tensorimpl_type_name() const override {
    return "VulkanOpaqueTensorImpl";
  }

  // TODO: storing strides separately is unnecessary, the base TensorImpl
  // has space for them
  SmallVector<int64_t, 5> strides_;
};

} // namespace at
