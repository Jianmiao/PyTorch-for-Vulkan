#pragma once

#ifdef USE_VULKAN_API

#include <ATen/core/Tensor.h>
#include <optional>

namespace at::native::vulkan::ops {

Tensor scaled_dot_product_attention_vulkan(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const std::optional<Tensor>& attn_mask,
    double dropout_p,
    bool is_causal,
    std::optional<double> scale,
    bool enable_gqa);

} // namespace at::native::vulkan::ops

#endif // USE_VULKAN_API
