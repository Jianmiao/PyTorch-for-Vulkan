// SSBO Vulkan backend bridge for the stock torch Vulkan device kernels.
// Each entry returns a defined tensor when the SSBO path succeeded, or an
// undefined tensor so the caller can fall back to the stock texture path.
#pragma once
#include <ATen/ATen.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace ssbo {

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
Tensor silu(const Tensor& x);
Tensor gelu(const Tensor& x);
Tensor sigmoid(const Tensor& x);
Tensor tanh(const Tensor& x);
Tensor elementwise_binary(const Tensor& a, const Tensor& b, int64_t op);
Tensor elementwise_scalar(const Tensor& a, double scalar, int64_t op);
Tensor upsample(const Tensor& x, IntArrayRef output_size, bool bilinear);
Tensor cat(const TensorList& tensors, int64_t dim);


} // namespace ssbo
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
