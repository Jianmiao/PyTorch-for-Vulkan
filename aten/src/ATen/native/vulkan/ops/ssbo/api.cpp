// vulkan_ssbo: C API exposing SSBO-based compute ops (compiled into torch).
#ifndef VKSSBO_EXPORT
#define VKSSBO_EXPORT
#endif
// Standalone DLL owns the VMA implementation; inside torch, Allocator.cpp
// provides it (same header from third_party/VulkanMemoryAllocator).
#ifdef VKSSBO_STANDALONE
#define VMA_IMPLEMENTATION
#endif
#include <include/vk_mem_alloc.h>

#include "core.h"
#include "spv_embedded.h"
#include <cstdlib>
#include <vector>
#include <mutex>
#include <string>

using namespace vkssbo;

static std::vector<uint32_t> load_spv(const std::string& name) {
  auto& m = at::native::vulkan::ssbo::spv_map();
  auto it = m.find(name + ".spv");
  if (it == m.end()) throw Error("cannot find embedded shader: " + name + ".spv");
  const uint8_t* data = it->second.first;
  size_t sz = it->second.second;
  std::vector<uint32_t> spv(sz / 4);
  memcpy(spv.data(), data, sz);
  return spv;
}

struct Dispatch {
  Device& d;
  ComputeManager& cm;
  VkDescriptorSetLayout ds_layout;
  VkDescriptorSet desc_set;

  Dispatch(Device& dev, ComputeManager& mgr)
      : d(dev), cm(mgr), ds_layout(VK_NULL_HANDLE), desc_set(VK_NULL_HANDLE) {}

  void begin(int num_bindings) {
    (void)num_bindings;
    ds_layout = d.global_ds_layout;
    VkDescriptorSetAllocateInfo dsai{};
    dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsai.descriptorPool = d.desc_pool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts = &ds_layout;
    VKSSBO_CHECK(vkAllocateDescriptorSets(d.device, &dsai, &desc_set));
  }

  void bind(int binding, VkBuffer buf, VkDeviceSize size) {
    VkDescriptorBufferInfo dbi{};
    dbi.buffer = buf;
    dbi.offset = 0;
    dbi.range = size;
    VkWriteDescriptorSet wds{};
    wds.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    wds.dstSet = desc_set;
    wds.dstBinding = (uint32_t)binding;
    wds.descriptorCount = 1;
    wds.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    wds.pBufferInfo = &dbi;
    vkUpdateDescriptorSets(d.device, 1, &wds, 0, nullptr);
  }

  void run(const std::string& shader, const void* params, uint32_t params_size,
           uint32_t gx, uint32_t gy, uint32_t gz) {
    // Select fp16-compute variant when the device supports shaderFloat16.
    std::string name = d.has_shader_f16 ? shader + "_fp16" : shader + "_fp32";
    Pipeline& p = cm.get_pipeline_locked(name, load_spv(name));
    VkCommandBuffer cmd = cm.cmd;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p.pipeline);
    vkCmdPushConstants(cmd, p.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, params_size, params);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p.layout, 0, 1, &desc_set, 0, nullptr);
    vkCmdDispatch(cmd, gx, gy, gz);
  }

  void cleanup() {
    // Return the descriptor set to the pool for reuse.
    if (desc_set) {
      vkFreeDescriptorSets(d.device, d.desc_pool, 1, &desc_set);
      desc_set = VK_NULL_HANDLE;
    }
    ds_layout = VK_NULL_HANDLE;
  }
};

// All SSBOs store float16 (bandwidth halved). Data crosses as fp32 from
// Python; convert here.
static void upload_direct(const float* data, size_t numel, Buffer& dev) {
  std::vector<uint16_t> halfs(numel);
  for (size_t i = 0; i < numel; i++) {
    float f = data[i];
    uint32_t bits;
    memcpy(&bits, &f, 4);
    // IEEE half conversion with round-to-nearest-even.
    uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t exp = (int32_t)((bits >> 23) & 0xff) - 127 + 15;
    uint32_t mant = bits & 0x7fffffu;
    uint16_t h;
    if (exp >= 31) {
      h = (uint16_t)(sign | 0x7c00u);
    } else if (exp <= 0) {
      if (exp < -10) {
        h = (uint16_t)sign;
      } else {
        mant |= 0x800000u;
        uint32_t shift = (uint32_t)(14 - exp);
        uint32_t m = (mant >> shift) & 0x3ffu;
        uint32_t rem = mant & ((1u << shift) - 1);
        uint32_t half = 1u << (shift - 1);
        if (rem > half || (rem == half && (m & 1u))) m++;
        h = (uint16_t)(sign | m);
      }
    } else {
      uint32_t m = (mant >> 13) & 0x3ffu;
      uint32_t rem = mant & 0x1fffu;
      if (rem > 0x1000u || (rem == 0x1000u && (m & 1u))) {
        m++;
        if (m == 0x400u) {
          m = 0;
          exp++;
          if (exp >= 31) {
            h = (uint16_t)(sign | 0x7c00u);
            halfs[i] = h;
            continue;
          }
        }
      }
      h = (uint16_t)(sign | ((uint32_t)exp << 10) | m);
    }
    halfs[i] = h;
  }
  void* p = map_buffer(dev);
  memcpy(p, halfs.data(), numel * sizeof(uint16_t));
  unmap_buffer(dev);
}

static float* download_direct(Buffer& dev, size_t numel) {
  float* out = (float*)malloc(numel * sizeof(float));
  void* p = map_buffer(dev);
  const uint16_t* halfs = (const uint16_t*)p;
  for (size_t i = 0; i < numel; i++) {
    uint16_t h = halfs[i];
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x3ffu;
    uint32_t bits;
    if (exp == 0) {
      if (mant == 0) {
        bits = sign;
      } else {
        int32_t e = -14;
        uint32_t m = mant;
        while (!(m & 0x400u)) { m <<= 1; e--; }
        m &= 0x3ffu;
        bits = sign | ((uint32_t)(e + 127) << 23) | (m << 13);
      }
    } else if (exp == 31) {
      bits = sign | 0x7f800000u | (mant << 13);
    } else {
      bits = sign | ((exp + 112u) << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &bits, 4);
    out[i] = f;
  }
  unmap_buffer(dev);
  return out;
}

extern "C" {

VKSSBO_EXPORT void vkssbo_init(const char* spv_dir) {
  (void)spv_dir;  // SPIR-V is embedded; nothing to configure.
  Device::get();
}

VKSSBO_EXPORT void vkssbo_free(float* p) { free(p); }

VKSSBO_EXPORT float* vkssbo_elementwise(int op, int mode, const float* a,
                                        const float* b, float scalar, int numel) {
  try {
  auto& d = Device::get();
  auto& cm = ComputeManager::get();

  cm.begin();
  Dispatch disp(d, cm);
  disp.begin(3);
  Buffer da = create_device_buffer(numel  * 2);
  upload_direct(a, numel, da);
  disp.bind(0, da.buffer, da.size);
  Buffer db = create_device_buffer(numel  * 2);
  if (mode == 1) {
    upload_direct(b, numel, db);
  }
  disp.bind(1, db.buffer, db.size);
  Buffer dout = create_device_buffer(numel  * 2);
  disp.bind(2, dout.buffer, dout.size);

  struct P { int op, mode, numel; float scalar; } p{op, mode, numel, scalar};
  disp.run("elementwise", &p, sizeof(p), (numel + 255) / 256, 1, 1);
  cm.submit_and_wait();
  float* out = download_direct(dout, numel);
  disp.cleanup();
  return out;
  } catch (const std::exception& e) {
    fprintf(stderr, "[vkssbo] elementwise error: %s\n", e.what());
    return nullptr;
  }
}

VKSSBO_EXPORT float* vkssbo_matmul(const float* a, const float* b, int B, int M,
                                   int K, int N, int transB) {
  try {
  auto& d = Device::get();
  auto& cm = ComputeManager::get();

  cm.begin();
  Dispatch disp(d, cm);
  disp.begin(3);
  Buffer da = create_device_buffer((size_t)B * M * K  * 2);
  upload_direct(a, (size_t)B * M * K, da);
  disp.bind(0, da.buffer, da.size);
  Buffer db = create_device_buffer((size_t)B * K * N  * 2);
  upload_direct(b, (size_t)B * K * N, db);
  disp.bind(1, db.buffer, db.size);
  Buffer dout = create_device_buffer((size_t)B * M * N  * 2);
  disp.bind(2, dout.buffer, dout.size);

  struct P { int B, M, K, N, transB; } p{B, M, K, N, transB};
  disp.run("matmul", &p, sizeof(p), (uint32_t)(N + 7) / 8, (uint32_t)(B * M + 7) / 8, 1);
  cm.submit_and_wait();
  float* out = download_direct(dout, (size_t)B * M * N);
  disp.cleanup();
  return out;
  } catch (const std::exception& e) {
    fprintf(stderr, "[vkssbo] matmul error: %s\n", e.what());
    return nullptr;
  }
}

VKSSBO_EXPORT float* vkssbo_softmax(const float* a, int rows, int cols) {
  try {
  auto& d = Device::get();
  auto& cm = ComputeManager::get();

  cm.begin();
  Dispatch disp(d, cm);
  disp.begin(2);
  Buffer da = create_device_buffer((size_t)rows * cols  * 2);
  upload_direct(a, (size_t)rows * cols, da);
  disp.bind(0, da.buffer, da.size);
  Buffer dout = create_device_buffer((size_t)rows * cols  * 2);
  disp.bind(1, dout.buffer, dout.size);

  struct P { int rows, cols; } p{rows, cols};
  disp.run("softmax", &p, sizeof(p), rows, 1, 1);
  cm.submit_and_wait();
  float* out = download_direct(dout, (size_t)rows * cols);
  disp.cleanup();
  return out;
  } catch (const std::exception& e) {
    fprintf(stderr, "[vkssbo] softmax error: %s\n", e.what());
    return nullptr;
  }
}

VKSSBO_EXPORT float* vkssbo_layernorm(const float* a, const float* w,
                                      const float* b, int rows, int cols,
                                      float eps, int has_w, int has_b) {
  try {
  auto& d = Device::get();
  auto& cm = ComputeManager::get();

  cm.begin();
  Dispatch disp(d, cm);
  disp.begin(4);
  Buffer da = create_device_buffer((size_t)rows * cols  * 2);
  upload_direct(a, (size_t)rows * cols, da);
  disp.bind(0, da.buffer, da.size);
  Buffer dout = create_device_buffer((size_t)rows * cols  * 2);
  disp.bind(1, dout.buffer, dout.size);
  Buffer dw = create_device_buffer(has_w ? ((size_t)cols * 2) : 2);
  if (has_w) upload_direct(w, (size_t)cols, dw);
  disp.bind(2, dw.buffer, dw.size);
  Buffer dbb = create_device_buffer(has_b ? ((size_t)cols * 2) : 2);
  if (has_b) upload_direct(b, (size_t)cols, dbb);
  disp.bind(3, dbb.buffer, dbb.size);

  struct P { int rows, cols; float eps; int has_w, has_b; } p{rows, cols, eps, has_w, has_b};
  disp.run("layernorm", &p, sizeof(p), rows, 1, 1);
  cm.submit_and_wait();
  float* out = download_direct(dout, (size_t)rows * cols);
  disp.cleanup();
  return out;
  } catch (const std::exception& e) {
    fprintf(stderr, "[vkssbo] layernorm error: %s\n", e.what());
    return nullptr;
  }
}

VKSSBO_EXPORT float* vkssbo_conv2d(const float* x, const float* w,
                                   const float* bias, int B, int C, int O,
                                   int H, int W, int Ho, int Wo, int KH,
                                   int KW, int SH, int SW, int PH, int PW,
                                   int G) {
  try {
  auto& d = Device::get();
  auto& cm = ComputeManager::get();

  cm.begin();
  Dispatch disp(d, cm);
  disp.begin(4);
  Buffer dx = create_device_buffer((size_t)B * C * H * W  * 2);
  upload_direct(x, (size_t)B * C * H * W, dx);
  if (std::getenv("VK_TRACE")) {
    float* check = download_direct(dx, (size_t)B * C * H * W);
    fprintf(stderr, "[conv-upload] x[0]=%.4f x[last]=%.4f numel=%zu\n",
            check[0], check[(size_t)B * C * H * W - 1], (size_t)B * C * H * W);
    free(check);
  }
  disp.bind(0, dx.buffer, dx.size);
  Buffer dw = create_device_buffer((size_t)O * (C / G) * KH * KW  * 2);
  upload_direct(w, (size_t)O * (C / G) * KH * KW, dw);
  disp.bind(1, dw.buffer, dw.size);
  Buffer db = create_device_buffer((size_t)O  * 2);
  upload_direct(bias, (size_t)O, db);
  disp.bind(2, db.buffer, db.size);
  Buffer dout = create_device_buffer((size_t)B * O * Ho * Wo  * 2);
  disp.bind(3, dout.buffer, dout.size);

  struct P {
    int B, C, O, H, W, Ho, Wo, KH, KW, SH, SW, PH, PW, G;
  } p{B, C, O, H, W, Ho, Wo, KH, KW, SH, SW, PH, PW, G};
  fprintf(stderr, "[conv] B=%d C=%d O=%d H=%d W=%d Ho=%d Wo=%d KH=%d KW=%d\n",
          B, C, O, H, W, Ho, Wo, KH, KW);
  disp.run("conv2d", &p, sizeof(p), (Wo + 7) / 8, (Ho + 7) / 8, B * O);
  cm.submit_and_wait();
  float* out = download_direct(dout, (size_t)B * O * Ho * Wo);
  disp.cleanup();
  return out;
  } catch (const std::exception& e) {
    fprintf(stderr, "[vkssbo] conv2d error: %s\n", e.what());
    return nullptr;
  }
}

VKSSBO_EXPORT float* vkssbo_conv3d(const float* x, const float* w,
                                   const float* bias, int B, int C, int O,
                                   int D, int H, int W, int Do, int Ho, int Wo,
                                   int KD, int KH, int KW,
                                   int SD, int SH, int SW,
                                   int PD, int PH, int PW, int G) {
  try {
  auto& d = Device::get();
  auto& cm = ComputeManager::get();

  cm.begin();
  Dispatch disp(d, cm);
  disp.begin(4);
  Buffer dx = create_device_buffer((size_t)B * C * D * H * W * 2);
  upload_direct(x, (size_t)B * C * D * H * W, dx);
  disp.bind(0, dx.buffer, dx.size);
  Buffer dw = create_device_buffer((size_t)O * (C / G) * KD * KH * KW * 2);
  upload_direct(w, (size_t)O * (C / G) * KD * KH * KW, dw);
  disp.bind(1, dw.buffer, dw.size);
  Buffer db = create_device_buffer((size_t)O * 2);
  upload_direct(bias, (size_t)O, db);
  disp.bind(2, db.buffer, db.size);
  Buffer dout = create_device_buffer((size_t)B * O * Do * Ho * Wo * 2);
  disp.bind(3, dout.buffer, dout.size);

  struct P {
    int B, C, O, D, H, W, Do, Ho, Wo, KD, KH, KW, SD, SH, SW, PD, PH, PW, G;
  } p{B, C, O, D, H, W, Do, Ho, Wo, KD, KH, KW, SD, SH, SW, PD, PH, PW, G};
  fprintf(stderr, "[conv3d] B=%d C=%d O=%d D=%d H=%d W=%d Do=%d Ho=%d Wo=%d K=%dx%dx%d\n",
          B, C, O, D, H, W, Do, Ho, Wo, KD, KH, KW);
  disp.run("conv3d", &p, sizeof(p), (Wo + 7) / 8, (Ho * Do + 7) / 8, B * O);
  cm.submit_and_wait();
  float* out = download_direct(dout, (size_t)B * O * Do * Ho * Wo);
  disp.cleanup();
  return out;
  } catch (const std::exception& e) {
    fprintf(stderr, "[vkssbo] conv3d error: %s\n", e.what());
    return nullptr;
  }
}

VKSSBO_EXPORT float* vkssbo_conv1d(const float* x, const float* w,
                                   const float* bias, int B, int C, int O,
                                   int L, int Lo, int KL, int SL, int PL,
                                   int G) {
  // (B,C,L) and (O,C,KL) are layout-identical to (B,C,1,L) / (O,C,1,KL).
  // Fake H dim: size 1, no padding.
  return vkssbo_conv2d(x, w, bias, B, C, O, 1, L, 1, Lo, 1, KL, 1, SL, 0, PL, G);
}

VKSSBO_EXPORT float* vkssbo_groupnorm(const float* a, const float* w,                                      const float* b, int B, int C, int H,
                                      int W, int G, float eps, int has_w,
                                      int has_b) {
  try {
  auto& d = Device::get();
  auto& cm = ComputeManager::get();

  cm.begin();
  Dispatch disp(d, cm);
  disp.begin(4);
  Buffer da = create_device_buffer((size_t)B * C * H * W  * 2);
  upload_direct(a, (size_t)B * C * H * W, da);
  disp.bind(0, da.buffer, da.size);
  Buffer dout = create_device_buffer((size_t)B * C * H * W  * 2);
  disp.bind(1, dout.buffer, dout.size);
  Buffer dw = create_device_buffer(has_w ? ((size_t)C * 2) : 2);
  if (has_w) upload_direct(w, (size_t)C, dw);
  disp.bind(2, dw.buffer, dw.size);
  Buffer dbb = create_device_buffer(has_b ? ((size_t)C * 2) : 2);
  if (has_b) upload_direct(b, (size_t)C, dbb);
  disp.bind(3, dbb.buffer, dbb.size);

  struct P { int B, C, H, W, G; float eps; int has_w, has_b; } p{B, C, H, W, G, eps, has_w, has_b};
  disp.run("groupnorm", &p, sizeof(p), (uint32_t)(B * C + 255) / 256, 1, 1);
  cm.submit_and_wait();
  float* out = download_direct(dout, (size_t)B * C * H * W);
  disp.cleanup();
  return out;
  } catch (const std::exception& e) {
    fprintf(stderr, "[vkssbo] groupnorm error: %s\n", e.what());
    return nullptr;
  }
}

VKSSBO_EXPORT float* vkssbo_upsample(const float* a, int B, int C, int H, int W,
                                     int Ho, int Wo, int mode) {
  try {
  auto& d = Device::get();
  auto& cm = ComputeManager::get();

  cm.begin();
  Dispatch disp(d, cm);
  disp.begin(2);
  Buffer da = create_device_buffer((size_t)B * C * H * W  * 2);
  upload_direct(a, (size_t)B * C * H * W, da);
  disp.bind(0, da.buffer, da.size);
  Buffer dout = create_device_buffer((size_t)B * C * Ho * Wo  * 2);
  disp.bind(1, dout.buffer, dout.size);

  struct P { int B, C, H, W, Ho, Wo, mode; } p{B, C, H, W, Ho, Wo, mode};
  disp.run("upsample", &p, sizeof(p), (uint32_t)(Wo + 7) / 8, (uint32_t)(Ho + 7) / 8, (uint32_t)(B * C));
  cm.submit_and_wait();
  float* out = download_direct(dout, (size_t)B * C * Ho * Wo);
  disp.cleanup();
  return out;
  } catch (const std::exception& e) {
    fprintf(stderr, "[vkssbo] upsample error: %s\n", e.what());
    return nullptr;
  }
}

VKSSBO_EXPORT float* vkssbo_cat4(const float* a0, const float* a1,
                                 const float* a2, const float* a3, int B,
                                 int H, int W, int C0, int C1, int C2, int C3,
                                 int O) {
  try {
  auto& d = Device::get();
  auto& cm = ComputeManager::get();

  cm.begin();
  Dispatch disp(d, cm);
  disp.begin(5);
  size_t base = (size_t)B * H * W;
  Buffer d0 = create_device_buffer(C0 > 0 ? base * C0 * sizeof(float) : 2);
  if (C0 > 0) upload_direct(a0, base * C0, d0);
  disp.bind(0, d0.buffer, d0.size);
  Buffer d1 = create_device_buffer(C1 > 0 ? base * C1 * sizeof(float) : 2);
  if (C1 > 0) upload_direct(a1, base * C1, d1);
  disp.bind(1, d1.buffer, d1.size);
  Buffer d2 = create_device_buffer(C2 > 0 ? base * C2 * sizeof(float) : 2);
  if (C2 > 0) upload_direct(a2, base * C2, d2);
  disp.bind(2, d2.buffer, d2.size);
  Buffer d3 = create_device_buffer(C3 > 0 ? base * C3 * sizeof(float) : 2);
  if (C3 > 0) upload_direct(a3, base * C3, d3);
  disp.bind(3, d3.buffer, d3.size);
  Buffer dout = create_device_buffer(base * O  * 2);
  disp.bind(4, dout.buffer, dout.size);

  struct P { int B, H, W, C0, C1, C2, C3, O; } p{B, H, W, C0, C1, C2, C3, O};
  disp.run("cat4", &p, sizeof(p), (uint32_t)(W + 7) / 8, (uint32_t)(H + 7) / 8, (uint32_t)(B * O));
  cm.submit_and_wait();
  float* out = download_direct(dout, base * O);
  disp.cleanup();
  return out;
  } catch (const std::exception& e) {
    fprintf(stderr, "[vkssbo] cat4 error: %s\n", e.what());
    return nullptr;
  }
}

} // extern "C"