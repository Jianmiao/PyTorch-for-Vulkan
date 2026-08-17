// SSBO Vulkan backend: bridges the stock torch Vulkan device kernels to the
// SSBO compute backend (data downloaded to CPU, computed on GPU via SSBO,
// uploaded back). Fallbacks return undefined tensors.
#include "SsboBackend.h"

#include <ATen/native/vulkan/ops/Common.h>
#include <torch/library.h>

#include <cstring>
#include <vector>

extern "C" {
void vkssbo_init(const char* spv_dir);
float* vkssbo_elementwise(int op, int mode, const float* a, const float* b,
                          float scalar, int numel);
float* vkssbo_matmul(const float* a, const float* b, int B, int M, int K, int N, int transB);
float* vkssbo_softmax(const float* a, int rows, int cols);
float* vkssbo_layernorm(const float* a, const float* w, const float* b,
                        int rows, int cols, float eps, int has_w, int has_b);
float* vkssbo_groupnorm(const float* a, const float* w, const float* b,
                        int B, int C, int H, int W, int G, float eps, int has_w, int has_b);
float* vkssbo_conv2d(const float* x, const float* w, const float* bias,
                     int B, int C, int O, int H, int W, int Ho, int Wo,
                     int KH, int KW, int SH, int SW, int PH, int PW, int G);
float* vkssbo_conv3d(const float* x, const float* w, const float* bias,
                     int B, int C, int O, int D, int H, int W, int Do, int Ho, int Wo,
                     int KD, int KH, int KW, int SD, int SH, int SW,
                     int PD, int PH, int PW, int G);
float* vkssbo_conv1d(const float* x, const float* w, const float* bias,
                     int B, int C, int O, int L, int Lo, int KL, int SL, int PL, int G);
float* vkssbo_upsample(const float* a, int B, int C, int H, int W, int Ho, int Wo, int mode);
float* vkssbo_cat4(const float* a, const float* b, const float* c, const float* d,
                   int B, int H, int W, int C0, int C1, int C2, int C3, int O);
void vkssbo_free(float* p);
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

Tensor cpu_float(const Tensor& t) {
  Tensor x = t;
  if (x.is_vulkan()) x = x.to("cpu");
  if (x.dtype() != at::kFloat) x = x.to(at::kFloat);
  return x.contiguous();
}

Tensor to_vulkan(Tensor x) {
  return x.contiguous().to(at::Device(at::kVulkan));
}

Tensor from_cpu(float* p, int64_t numel, IntArrayRef shape) {
  if (!p) return Tensor();
  Tensor t = at::from_blob(p, shape, at::TensorOptions().dtype(at::kFloat));
  // from_blob steals no ownership; copy then free.
  Tensor out = t.clone().contiguous();
  vkssbo_free(p);
  return out;
}

bool is_float_input(const Tensor& x) {
  return x.scalar_type() == at::kFloat || x.scalar_type() == at::kHalf;
}

} // namespace

bool enabled() {
  return g_enabled;
}

void init() {
  static bool done = false;
  if (!done) {
    vkssbo_init(nullptr);
    done = true;
  }
}

Tensor convolution(
    const Tensor& input,
    const Tensor& weight,
    const std::optional<Tensor>& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups) {
  if (!enabled() || !is_float_input(input) || input.dim() != 4) return Tensor();
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
    if (dilation[0] != 1 || dilation[1] != 1) return Tensor();
    int64_t B = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);
    int64_t O = w.size(0), KH = w.size(2), KW = w.size(3);
    int64_t Ho = (H + 2 * padding[0] - KH) / stride[0] + 1;
    int64_t Wo = (W + 2 * padding[1] - KW) / stride[1] + 1;
    float* p = vkssbo_conv2d(
        x.data_ptr<float>(), w.data_ptr<float>(), b.data_ptr<float>(),
        (int)B, (int)C, (int)O, (int)H, (int)W, (int)Ho, (int)Wo,
        (int)KH, (int)KW, (int)stride[0], (int)stride[1],
        (int)padding[0], (int)padding[1], (int)groups);
    return to_vulkan(from_cpu(p, B * O * Ho * Wo, {B, O, Ho, Wo}));
  } catch (const std::exception&) {
    return Tensor();
  }
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
    if (dilation[0] != 1) return Tensor();
    int64_t B = x.size(0), C = x.size(1), L = x.size(2);
    int64_t O = w.size(0), KL = w.size(2);
    int64_t Lo = (L + 2 * padding[0] - KL) / stride[0] + 1;
    float* p = vkssbo_conv1d(
        x.data_ptr<float>(), w.data_ptr<float>(), b.data_ptr<float>(),
        (int)B, (int)C, (int)O, (int)L, (int)Lo, (int)KL,
        (int)stride[0], (int)padding[0], (int)groups);
    return to_vulkan(from_cpu(p, B * O * Lo, {B, O, Lo}));
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
    for (int64_t d : dilation) {
      if (d != 1) return Tensor();
    }
    IntArrayRef s = stride.size() == 1 ? stride : stride;
    IntArrayRef pp = padding.size() == 1 ? padding : padding;
    int64_t sd = s[0], sh = s.size() > 1 ? s[1] : s[0], sw = s.size() > 2 ? s[2] : s[0];
    int64_t pd = pp[0], ph = pp.size() > 1 ? pp[1] : pp[0], pw = pp.size() > 2 ? pp[2] : pp[0];
    int64_t B = x.size(0), C = x.size(1), D = x.size(2), H = x.size(3), W = x.size(4);
    int64_t O = w.size(0), KD = w.size(2), KH = w.size(3), KW = w.size(4);
    int64_t Do = (D + 2 * pd - KD) / sd + 1;
    int64_t Ho = (H + 2 * ph - KH) / sh + 1;
    int64_t Wo = (W + 2 * pw - KW) / sw + 1;
    float* p = vkssbo_conv3d(
        x.data_ptr<float>(), w.data_ptr<float>(), b.data_ptr<float>(),
        (int)B, (int)C, (int)O, (int)D, (int)H, (int)W,
        (int)Do, (int)Ho, (int)Wo,
        (int)KD, (int)KH, (int)KW,
        (int)sd, (int)sh, (int)sw, (int)pd, (int)ph, (int)pw, (int)groups);
    return to_vulkan(from_cpu(p, B * O * Do * Ho * Wo, {B, O, Do, Ho, Wo}));
  } catch (const std::exception&) {
    return Tensor();
  }
}

Tensor linear(const Tensor& input, const Tensor& weight, const std::optional<Tensor>& bias) {
  if (!enabled() || !is_float_input(input)) return Tensor();
  init();
  try {
    Tensor x = cpu_float(input);
    Tensor w = cpu_float(weight).t().contiguous();
    Tensor b = bias && bias->defined() ? cpu_float(*bias) : Tensor();
    int64_t B = x.size(0), M = x.size(1), K = x.size(2);
    int64_t N = w.size(1);
    int64_t total = x.numel();
    float* p = vkssbo_matmul(x.data_ptr<float>(), w.data_ptr<float>(),
                             (int)B, (int)M, (int)K, (int)N, 0);
    Tensor out = from_cpu(p, total, x.sizes());
    if (!out.defined()) return Tensor();
    if (b.defined()) out = out + b.view({1, N}).contiguous();
    return to_vulkan(out);
  } catch (const std::exception&) {
    return Tensor();
  }
}

Tensor mm(const Tensor& a, const Tensor& b) {
  if (!enabled() || !is_float_input(a) || a.dim() != 2) return Tensor();
  init();
  try {
    Tensor x = cpu_float(a);
    Tensor w = cpu_float(b);
    int64_t M = x.size(0), K = x.size(1), N = w.size(1);
    float* p = vkssbo_matmul(x.data_ptr<float>(), w.data_ptr<float>(), 1, (int)M, (int)K, (int)N, 0);
    return to_vulkan(from_cpu(p, M * N, {M, N}));
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
                             (int)B, (int)M, (int)K, (int)N, 0);
    return to_vulkan(from_cpu(p, B * M * N, {B, M, N}));
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
    return to_vulkan(from_cpu(p, rows * cols, x.sizes()));
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
    Tensor t2 = t.reshape({-1, cols});
    int64_t rows = t2.size(0);
    Tensor ww = w && w->defined() ? cpu_float(*w).contiguous() : Tensor();
    Tensor bb = b && b->defined() ? cpu_float(*b).contiguous() : Tensor();
    float* p = vkssbo_layernorm(
        t2.data_ptr<float>(),
        ww.defined() ? ww.data_ptr<float>() : nullptr,
        bb.defined() ? bb.data_ptr<float>() : nullptr,
        (int)rows, (int)cols, (float)eps, ww.defined() ? 1 : 0, bb.defined() ? 1 : 0);
    return to_vulkan(from_cpu(p, rows * cols, x.sizes()));
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
    if (t.dim() != 4) return Tensor();
    int64_t B = t.size(0), C = t.size(1), H = t.size(2), W = t.size(3);
    Tensor ww = w && w->defined() ? cpu_float(*w).contiguous() : Tensor();
    Tensor bb = b && b->defined() ? cpu_float(*b).contiguous() : Tensor();
    float* p = vkssbo_groupnorm(
        t.data_ptr<float>(),
        ww.defined() ? ww.data_ptr<float>() : nullptr,
        bb.defined() ? bb.data_ptr<float>() : nullptr,
        (int)B, (int)C, (int)H, (int)W, (int)num_groups, (float)eps,
        ww.defined() ? 1 : 0, bb.defined() ? 1 : 0);
    return to_vulkan(from_cpu(p, B * C * H * W, {B, C, H, W}));
  } catch (const std::exception&) {
    return Tensor();
  }
}

Tensor silu(const Tensor& x) {
  if (!enabled() || !is_float_input(x)) return Tensor();
  init();
  try {
    Tensor t = cpu_float(x);
    float* p = vkssbo_elementwise(OP_SILU, 0, t.data_ptr<float>(), nullptr, 0.0f, (int)t.numel());
    return to_vulkan(from_cpu(p, t.numel(), x.sizes()));
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
    return to_vulkan(from_cpu(p, t.numel(), x.sizes()));
  } catch (const std::exception&) {
    return Tensor();
  }
}

Tensor sigmoid(const Tensor& x) {
  if (!enabled() || !is_float_input(x)) return Tensor();
  init();
  try {
    Tensor t = cpu_float(x);
    float* p = vkssbo_elementwise(OP_SIGMOID, 0, t.data_ptr<float>(), nullptr, 0.0f, (int)t.numel());
    return to_vulkan(from_cpu(p, t.numel(), x.sizes()));
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
    return to_vulkan(from_cpu(p, t.numel(), x.sizes()));
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
    return to_vulkan(from_cpu(p, x.numel(), a.sizes()));
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
    return to_vulkan(from_cpu(p, x.numel(), a.sizes()));
  } catch (const std::exception&) {
    return Tensor();
  }
}

Tensor upsample(const Tensor& x, IntArrayRef output_size, bool bilinear) {
  if (!enabled() || !is_float_input(x) || x.dim() != 4) return Tensor();
  init();
  try {
    Tensor t = cpu_float(x);
    int64_t B = t.size(0), C = t.size(1), H = t.size(2), W = t.size(3);
    int64_t Ho = output_size[0], Wo = output_size[1];
    float* p = vkssbo_upsample(t.data_ptr<float>(), (int)B, (int)C, (int)H, (int)W,
                               (int)Ho, (int)Wo, bilinear ? 1 : 0);
    return to_vulkan(from_cpu(p, B * C * Ho * Wo, {B, C, Ho, Wo}));
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
    return to_vulkan(from_cpu(p, B * O * H * W, {B, O, H, W}));
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
  m.impl("group_norm", [](const Tensor& x, int64_t num_groups,
                          const std::optional<Tensor>& w, const std::optional<Tensor>& b,
                          double eps, bool cudnn_enabled) -> Tensor {
    (void)cudnn_enabled;
    return ssbo::group_norm(x, num_groups, w, b, eps);
  });
  m.impl("silu", [](const Tensor& x) -> Tensor { return ssbo::silu(x); });
  m.impl("conv3d", [](const Tensor& x, const Tensor& w, const std::optional<Tensor>& b,
                      IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation,
                      int64_t groups) -> Tensor {
    return ssbo::conv3d(x, w, b, stride, padding, dilation, groups);
  });
}

} // namespace ssbo
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
