#include <ATen/ATen.h>
#include <ATen/native/vulkan/ops/SDPA.h>
#include <ATen/ops/scaled_dot_product_attention.h>
#include <cmath>

#ifdef USE_VULKAN_API

namespace at::native::vulkan::ops {

// Compatibility-first SDPA entry point.  The first supported contract is
// deliberately small and explicit: FP32 inference, four-dimensional Q/K/V,
// optional 2D/4D bool or float32 mask, causal, valid GQA and dropout_p == 0.
// The math itself runs through the
// stock CPU implementation and the result is copied back to Vulkan.  This is
// a semantic fallback for applications that require the public SDPA operator;
// it makes no performance or native-backward claim.
Tensor scaled_dot_product_attention_vulkan(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const std::optional<Tensor>& attn_mask,
    double dropout_p,
    bool is_causal,
    std::optional<double> scale,
    bool enable_gqa) {
  TORCH_CHECK(query.defined() && key.defined() && value.defined(),
              "Vulkan SDPA requires defined Q/K/V tensors");
  TORCH_CHECK(query.is_vulkan() && key.is_vulkan() && value.is_vulkan(),
              "Vulkan SDPA expects Q/K/V on the Vulkan device");
  TORCH_CHECK(query.scalar_type() == kFloat && key.scalar_type() == kFloat &&
                  value.scalar_type() == kFloat,
              "Vulkan SDPA compatibility path supports float32 only");
  TORCH_CHECK(query.dim() == 4 && key.dim() == 4 && value.dim() == 4,
              "Vulkan SDPA compatibility path supports rank-4 Q/K/V only");
  if (attn_mask.has_value()) {
    TORCH_CHECK(attn_mask->defined(),
                "Vulkan SDPA does not accept an undefined attention mask");
    TORCH_CHECK(attn_mask->scalar_type() == kBool ||
                    attn_mask->scalar_type() == kFloat,
                "Vulkan SDPA compatibility path supports bool or float32 masks only");
    TORCH_CHECK(!attn_mask->requires_grad(),
                "Vulkan SDPA compatibility path requires an inference mask");
    TORCH_CHECK(attn_mask->dim() == 2 || attn_mask->dim() == 4,
                "Vulkan SDPA compatibility path supports 2D or 4D masks only");
    TORCH_CHECK(attn_mask->device().is_cpu() || attn_mask->is_vulkan(),
                "Vulkan SDPA compatibility path supports CPU or Vulkan masks only");
    TORCH_CHECK(!is_causal,
                "Vulkan SDPA does not accept attn_mask together with is_causal");
    const auto batch = query.size(0);
    const auto heads = query.size(1);
    const auto query_length = query.size(-2);
    const auto key_length = key.size(-2);
    if (attn_mask->dim() == 2) {
      TORCH_CHECK(attn_mask->size(0) == query_length &&
                      attn_mask->size(1) == key_length,
                  "Vulkan SDPA 2D mask must have shape [L, S]");
    } else {
      TORCH_CHECK(
          (attn_mask->size(0) == 1 || attn_mask->size(0) == batch) &&
              (attn_mask->size(1) == 1 || attn_mask->size(1) == heads) &&
              (attn_mask->size(2) == 1 || attn_mask->size(2) == query_length) &&
              (attn_mask->size(3) == 1 || attn_mask->size(3) == key_length),
          "Vulkan SDPA 4D mask is not broadcastable to [B, H, L, S]");
    }
  }
  TORCH_CHECK(std::isfinite(dropout_p) && dropout_p >= 0.0 && dropout_p <= 1.0,
              "Vulkan SDPA dropout_p must be in the range [0, 1]");
  TORCH_CHECK(dropout_p == 0.0,
              "Vulkan SDPA compatibility path requires dropout_p=0");
  TORCH_CHECK(!query.requires_grad() && !key.requires_grad() &&
                  !value.requires_grad(),
              "Vulkan SDPA compatibility path is inference-only");
  TORCH_CHECK(query.numel() > 0 && key.numel() > 0 && value.numel() > 0,
              "Vulkan SDPA compatibility path does not support empty tensors");
  TORCH_CHECK(query.size(0) == key.size(0) && query.size(0) == value.size(0) &&
                  key.size(1) == value.size(1),
              "Vulkan SDPA compatibility path requires matching batch and K/V head counts");
  if (enable_gqa) {
    TORCH_CHECK(query.size(1) >= key.size(1) &&
                    query.size(1) % key.size(1) == 0,
                "Vulkan SDPA GQA requires query heads to be a multiple of K/V heads");
  } else {
    TORCH_CHECK(query.size(1) == key.size(1),
                "Vulkan SDPA without GQA requires matching Q/K/V head counts");
  }
  TORCH_CHECK(query.size(-1) == key.size(-1),
              "Vulkan SDPA compatibility path requires matching Q/K head dimensions");
  TORCH_CHECK(key.size(-2) == value.size(-2),
              "Vulkan SDPA compatibility path requires matching K/V sequence lengths");
  if (scale.has_value()) {
    TORCH_CHECK(std::isfinite(*scale),
                "Vulkan SDPA compatibility path requires a finite scale");
  }

  std::optional<Tensor> attn_mask_cpu;
  if (attn_mask.has_value()) {
    attn_mask_cpu = attn_mask->is_vulkan() ? attn_mask->cpu() : *attn_mask;
  }

  const Tensor query_cpu = query.cpu();
  const Tensor key_cpu = key.cpu();
  const Tensor value_cpu = value.cpu();
  const Tensor output_cpu = at::scaled_dot_product_attention(
      query_cpu, key_cpu, value_cpu, attn_mask_cpu, dropout_p, is_causal,
      scale, enable_gqa);
  return output_cpu.vulkan();
}

} // namespace at::native::vulkan::ops

#endif // USE_VULKAN_API
