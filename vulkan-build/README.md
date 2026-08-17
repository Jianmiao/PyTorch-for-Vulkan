# PyTorch for Vulkan (custom build archive)

从源码构建的 PyTorch 2.13.0a0+gitcf30153,启用官方 Vulkan 后端,归档于此以便
日后修复/重编�?避免从零重新下载源码构建)�?
## 内容

- `pytorch-src/` �?完整 PyTorch 源码�?含修改后�?Vulkan 后端 + `build/` 增量产物,
  ninja 可直接增量重编译)
- `wheels/` �?已构建的 wheel(cp310/cp311/cp312/cp313,win_amd64,FP16 推理�?

说明:wheels �?*官方 PyTorch 源码构建**(启用官方 texture �?Vulkan 后端并打�?少量补丁),不是重写。真正的重写产物�?SSBO(Storage Buffer)�?Vulkan 后端
`vulkan_ssbo`(位于 ComfyUI 整合包的 tools/vulkan_ssbo,�?GLSL 源码 + C API +
编译好的 DLL),它替代官�?texture 后端成为主计算路径�?
**`torch-...-cp313-...-SSBO.whl`**:在官方构建基础�?*内嵌 SSBO 后端**
(2026-08-14):vulkan 设备上的 conv2d/conv1d/conv3d/linear/matmul/bmm/softmax/
layer_norm/group_norm/silu/sigmoid/tanh/upsample/cat 等算子直接由 SSBO 计算
(官方 texture 后端仅作 fallback),并带权重 GPU 缓存(VRAM 驻留,LRU,默认上限
12GB,可用环境变量 `VKSSBO_WEIGHT_CACHE_MB` 调整)。ComfyUI 便携版使�?cp313,装这�?wheel 即可让原�?`device="vulkan"` 程序�?SSBO�?
## 构建配置(当时的构建命�?

```
set USE_VULKAN=1
set USE_VULKAN_SHADERC_RUNTIME=1
set USE_VULKAN_WRAPPER=0
set USE_VULKAN_FP16_INFERENCE=1
set USE_CUDA=0
set USE_DISTRIBUTED=0
```

环境:
- VS 2022 BuildTools(VsDevCmd.bat -arch=x64 -host_arch=x64)
- Python 3.10/3.11/3.12/3.13(各建一�?wheel)
- ninja、cmake、Vulkan SDK(构建时用,运行不需�?
- 构建命令:`python setup.py bdist_wheel`(�?ninja -C build torch_cpu.dll)

## 对官�?Vulkan 后端的修�?源码内已包含)

- aten/src/ATen/native/vulkan/ops/Clamp.cpp �?加入 clamp 支持补丁
- aten/src/ATen/native/vulkan/ops/Convolution.cpp �?卷积修复
- aten/src/ATen/native/vulkan/ops/Mm.cpp �?matmul/linear 支持
- aten/src/ATen/native/vulkan/ops/Shape.cpp �?view dtype 修复
- aten/src/ATen/native/vulkan/Context.cpp �?长链�?flush 机制(防栈溢出)
- aten/src/ATen/native/vulkan/Packing.cpp �?kHalf 支持
- aten/src/ATen/native/vulkan/Resource.cpp �?内存预算追踪

注意:官方 texture 后端仍受 maxImageDimension3D 限制,主运行后端是
SSBO �?`vulkan_ssbo`(�?ComfyUI 整合�?tools/vulkan_ssbo)�?
## 已知问题(本机 torch 2.13 build)

- CPU �?oneDNN:conv2d 大通道(C>=128)/groups 卷积结果错误 �?ComfyUI 中已禁用
  mkldnn(torch.backends.mkldnn.enabled = False,vkssbo-backend 加载时设�?
- 2D/4D 大张量的 torch.sum/softmax 数值错�?double 4D softmax/conv1d double 崩溃
  �?测试与主链路均已绕开(参�?test_torch.py 的写�?
- 这些不影�?ComfyUI 主链�?所有算子由 SSBO 后端接管)

## 安装 wheel

将对�?cp 版本�?wheel 安装到目�?python:
`pip install wheels/torch-2.13.0a0+gitcf30153-cpXXX-cpXXX-win_amd64.whl`

## Vulkan SDK

重新编译 torch / vulkan_ssbo 需�?Vulkan SDK 1.4.357.0(仅编译时,运行不需�?�?本机网络无法直连下载,�?https://vulkan.lunarg.com/sdk/home 下载
`VulkanSDK-1.4.357.0-Installer.exe` 后安�?或将 SDK 目录放置�?`D:\PyTorch for Vulkan\VulkanSDK`(构建时设�?VULKAN_SDK 环境变量指向�?�?构建 vulkan_ssbo 仅需其中 `Include/vulkan/` 头文件与 `Lib/vulkan-1.lib`�?