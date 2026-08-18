# PyTorch for Vulkan

这是面向 Windows 的 PyTorch Vulkan 定制版本。官方 PyTorch 2.13.0 仅作为可编译的基础源码，我们自己的工作集中在 Vulkan SSBO 后端及其算子接入。

## 自研 Vulkan SSBO 后端

主计算路径使用 Vulkan Storage Buffer（SSBO），不依赖官方纹理后端的大型图像布局限制。核心实现位于：

```text
aten/src/ATen/native/vulkan/ops/ssbo/
```

主要特点：

- Vulkan SSBO 计算内核和显式同步。
- GPU 驻留权重缓存，带 LRU 限制；默认上限 12 GB，可通过 `VKSSBO_WEIGHT_CACHE_MB` 调整。
- 支持 FP16 权重和激活推理。
- 内嵌 SPIR-V shader，便于打包后的 wheel 使用。
- 未覆盖的算子可以回退到官方 Vulkan 实现。

## 当前算子覆盖

当前 SSBO 路径覆盖 ComfyUI 主链路中的核心算子：

- `linear`、`matmul`、批量矩阵乘法
- 2D 卷积
- `softmax`
- `layer_norm`、`group_norm`
- 一元和二元逐元素算子
- `copy`、`concat`、shape 操作和 `upsample`

## 源码结构

| 路径 | 内容 |
| --- | --- |
| `aten/src/ATen/native/vulkan/ops/ssbo/` | SSBO 运行时、Vulkan API、内存和 dispatch |
| `aten/src/ATen/native/vulkan/ops/` | 算子接入和回退选择 |
| `tools/gen_vulkan_spv.py` | shader 编译和嵌入工具 |
| `vulkan-build/` | CPython 3.10-3.13 的 Windows 构建脚本 |

## 预编译 wheel

CPython 3.13、Windows x64 版本发布在 GitHub Releases，无需克隆仓库：

[下载 torch-2.13.0a0+gitcf30153-cp313-cp313-win_amd64-SSBO.whl](https://github.com/Jianmiao/PyTorch-for-Vulkan/releases/download/v2.13.0-vulkan-ssbo-cp313/torch-2.13.0a0%2Bgitcf30153-cp313-cp313-win_amd64-SSBO.whl)

直接安装：

```powershell
python -m pip install "https://github.com/Jianmiao/PyTorch-for-Vulkan/releases/download/v2.13.0-vulkan-ssbo-cp313/torch-2.13.0a0%2Bgitcf30153-cp313-cp313-win_amd64-SSBO.whl"
```

SHA-256：

```text
3428B6703D1BDD20F3A363198FE473E2958467AFCCC607365DDB76E95A315988
```

运行需要支持 Vulkan 的 Windows 驱动；重新编译时需要 Vulkan SDK，正常安装 wheel 运行不需要 SDK。

## 源码构建

`vulkan-build/` 中的脚本启用以下配置：

```text
USE_VULKAN=1
USE_VULKAN_SHADERC_RUNTIME=1
USE_VULKAN_WRAPPER=0
USE_VULKAN_FP16_INFERENCE=1
USE_CUDA=0
USE_DISTRIBUTED=0
```

构建环境需要 Visual Studio 2022 Build Tools、Ninja、CMake、Vulkan SDK 以及对应版本的 Python。脚本中的本机路径需要按实际环境调整。通用构建命令：

```powershell
python setup.py bdist_wheel
```
