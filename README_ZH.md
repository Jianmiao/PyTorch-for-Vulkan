# PyTorch for Vulkan

这是基于上游 PyTorch 2.13.0、正在建设 Vulkan Storage Buffer（SSBO）后端的项目。
我们的目标是：让用户选择 GPU 时，不必受单一厂商的软件生态限制，让现有 PyTorch 应用能够在更多厂商的硬件上实际运行。

当前开发和本地验证主要集中在 Windows 与 Intel 核显/UMA 环境；更广的硬件与操作系统覆盖属于建设目标，不代表已经支持。

[English](README.md) | [项目路线图](docs/vulkan-roadmap.md)

## 为什么选择 Vulkan

Vulkan 提供了跨 GPU 厂商的计算接口。我们希望降低现有 PyTorch 应用使用不同硬件的适配成本，服务对象包括核显、新独显和老显卡。延长老硬件的使用寿命是项目的一项价值。

支持 Vulkan 本身不保证某张显卡、模型或算子可以使用。实际支持还取决于设备特性、驱动、张量语义和执行路径，必须逐项验证。原生执行、混合执行、CPU 回退、不支持与未验证需要明确区分。

## 项目目标与优先级

- 首先完成实用的单设备推理兼容：张量与算子行为正确，资源生命周期可靠，回退清楚可见。
- 让普通 PyTorch 程序及 ComfyUI、GPT-SoVITS 等应用适配复用同一组后端能力。
- 逐步扩大经过验证的 GPU 厂商和操作系统范围，公开每种配置的支持边界与证据。
- 模型驻留、卸载、预取和高级注意力策略由应用或可选插件选择。已经承诺支持的工作负载应有不依赖这些插件的可用默认路径。

完整原生训练需要单独完成算子、反向传播和数值验收。不同品牌 GPU 共同执行同一任务属于独立的未来研究方向；跨厂商可移植不代表显存自动合并，也不承诺所有 GPU 性能相同。应用可以继续使用已经成熟可用的 CUDA、XPU 或 ROCm 路径。

## 当前开发状态

2026 年 10 月源码快照仍属于实验性兼容性建设，不是新的 wheel 发行，也不代表完整 PyTorch Vulkan 支持。以下验证主要覆盖本地 Windows/Intel UMA 配置，不能外推到所有 Vulkan 显卡或其他操作系统。

- 小型 FP32 原生 buffer 计算链及受限的 view/copy/stride 语义已通过本地验证；完整 Storage、任意张量形状和原生 backward 仍未完成。
- 公共 Vulkan SDPA 当前使用 CPU 数学计算并返回 Vulkan 张量。受限 FP32 推理接口已有 mask、causal、GQA 验证，但不属于原生注意力加速。
- MSVC OpenMP 运行库选择和 Vulkan 退出清理修复已在源码中；已有便携环境与旧 wheel 不会自动包含这些修复。
- ComfyUI/GPT-SoVITS 配套适配使用独立的 standalone SSBO 桥。小型 provider 测试和 GSV 短语音验证不等于完整原生注意力、训练支持或端到端加速验收。

选择运行时或引用历史任务书前，请阅读[当前范围与验证记录](docs/vulkan-source-status-20261016.md)。

## 自研 Vulkan SSBO 后端

主计算路径使用 Vulkan Storage Buffer（SSBO），不依赖官方纹理后端的大型图像布局限制。核心实现位于：

```text
aten/src/ATen/native/vulkan/ops/ssbo/
```

主要特点：

- Vulkan SSBO 计算内核和显式同步。
- 使用 GPU Storage Buffer 承载张量，适合大型推理 workload，避免纹理尺寸限制。
- 在支持的 SSBO 推理路径中使用 FP16 权重和激活；不代表通用原生张量已完整支持 FP16。
- 内嵌 SPIR-V shader，便于打包后的 wheel 使用。
- 存在兼容的已注册路径时，可回退到原有 Vulkan 实现；不支持的情况仍需明确处理。

摆脱纹理布局限制不等于取消设备内存限制，也不等于取消当前原生张量实现的范围限制。

## 编程语言与工具

以下说明定制后端及配套工具的分工，不代表整个上游派生仓库的语言占比，也不代表每个已发行 wheel 都包含这些能力。

| 语言 / 工具 | 在 Vulkan SSBO 工作中的职责 |
| --- | --- |
| C++ | Vulkan 运行时、buffer 与资源生命周期、驻留缓存、同步、分派、融合、调优及 PyTorch/ATen 接入 |
| Rust | 基于记账快照的内存预算、安全准入、缓存限额和驱逐建议 |
| GLSL | Vulkan 计算 shader 源码，编译为 SPIR-V 并嵌入部署 |
| Python | 后端配置、shader 构建工具、回归测试、基准和配套应用适配 |
| Windows Batch | Windows 构建、环境检查与测试入口 |
| CMake | 原生构建配置及通过 Cargo 可选接入 Rust 静态库 |

Rust 实现位于 `aten/src/ATen/native/vulkan/ops/ssbo/memory_policy/`。构建环境有 Cargo 时可接入 Rust，否则保留 C++ 策略回退。Rust 接收记账数据，不拥有、分配或销毁 Vulkan 句柄；C++ 负责 Vulkan 资源所有权和生命周期。Vulkan 是计算 API，SPIR-V 是编译后的 shader 格式。

## 当前算子覆盖

SSBO 实现包含 ComfyUI 工作负载使用的以下算子类别路径：

- `linear`、`matmul`、批量矩阵乘法
- 2D 卷积
- `softmax`
- `layer_norm`、`group_norm`
- 一元和二元逐元素算子
- `copy`、`concat`、shape 操作和 `upsample`

这些是已有实现的类别，不代表每种 shape、dtype、layout 或梯度组合都受支持，也不代表任意 PyTorch 程序可以完全在 Vulkan 上执行。实际边界以当前验证记录为准。

## 源码结构

| 路径 | 内容 |
| --- | --- |
| `aten/src/ATen/native/vulkan/ops/ssbo/` | SSBO 运行时、Vulkan API、内存和 dispatch |
| `aten/src/ATen/native/vulkan/ops/ssbo/memory_policy/` | Rust 预算策略、C 接口和 Cargo 配置 |
| `aten/src/ATen/native/vulkan/ops/ssbo/shaders/` | GLSL 计算 shader 源码 |
| `aten/src/ATen/native/vulkan/ops/` | 算子接入和回退选择 |
| `tools/gen_vulkan_spv.py` | shader 编译和嵌入工具 |
| `torch/vulkan_ssbo_configuration.py` | Python 后端配置 |
| `test/*vulkan_ssbo*.py` | 集成运行时回归与基准 |
| `vulkan-build/` | CPython 3.10-3.15 的 Windows 构建脚本 |

配套 ComfyUI 适配和 standalone `tools/vulkan_ssbo/` 测试、打包工具维护在单独的 ComfyUI 工作区，不属于本 PyTorch 源码仓库。

## 预编译 wheel

以下是较早的 Windows 构建产物，不包含 2026 年 10 月源码快照中的全部变更和验证，不能据此推断当前开发源码的支持状态。

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

## 上游与项目贡献

本项目基于上游 [PyTorch](https://github.com/pytorch/pytorch) 2.13.0 源码。Python API、ATen、dispatcher、autograd 和原有后端等框架能力来自上游；本项目的工作集中在 Vulkan SSBO 后端、预算策略、计算内核、算子接入及配套工具。第三方依赖保留各自的作者与许可证。

GitHub 贡献者列表反映保留的仓库提交历史，包括上游贡献者，并非 Vulkan SSBO 团队名单。Languages 面板统计 GitHub Linguist 识别的仓库文件，不代表定制后端的实现占比；上面的语言分工表用于说明技术选择。

## 开源协议

本项目遵循 PyTorch 原版 BSD-style License。仓库保留了未经修改的原始
`LICENSE` 和 `NOTICE` 文件，以及其中的版权和来源声明。
