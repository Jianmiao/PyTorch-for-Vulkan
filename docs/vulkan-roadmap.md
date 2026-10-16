# PyTorch for Vulkan roadmap / 项目路线图

Updated / 更新：2026-10-16

## Mission / 项目目标

Give users more choice of GPU without tying existing PyTorch applications to
one vendor's software ecosystem. Build a reusable Vulkan backend with correct
tensor/operator semantics, reliable resource lifetimes and explicit capability
reporting. Integrated GPUs, new discrete GPUs and older GPUs are all potential
beneficiaries, subject to device features and validation.

让用户选择 GPU 时，不必受单一厂商的软件生态限制。建设可复用的 Vulkan
后端，保证张量与算子语义正确、资源生命周期可靠、能力声明真实。核显、新独显
和老显卡均属于潜在服务对象，实际支持取决于设备特性与验证结果。

Cross-vendor portability is the main goal. Other operating systems require
separate qualification. Heterogeneous multi-GPU execution, pooled VRAM and
complete native training are not implied by single-device compatibility.

跨厂商可移植是主目标，其他操作系统需要分别验证。单设备兼容不代表不同品牌
GPU 已能共同执行同一任务、显存可以合并，或已经完成原生训练支持。

## Starting point / 已有基础

The [source status record](vulkan-source-status-20261016.md) separates verified
behavior from limitations. Local Windows/Intel UMA checks already cover bounded
FP32 native-buffer chains (B1), selected views/copies (B2), and capability/copy
boundaries (B3). These remain regression baselines; they are not full Storage,
arbitrary layouts or native backward. Public SDPA currently uses CPU math with
Vulkan output. Companion application checks are separate from native-backend
qualification, and older installed runtimes do not inherit source fixes.

[源码状态记录](vulkan-source-status-20261016.md)区分了已验证行为与限制。
B1 小型 FP32 原生链、B2 部分视图/拷贝和 B3 能力/拷贝边界已有本地
Windows/Intel UMA 证据，后续作为回归基线，不从零重做。完整 Storage、任意布局
和原生 backward 尚未完成。公共 SDPA 仍为 CPU 计算、Vulkan 输出；配套应用
验证不能替代原生后端验收，旧运行环境也不会自动获得源码修复。

## Near-term milestones / 近期里程碑

| Step / 阶段 | Work / 工作 | Completion evidence / 验收依据 |
| --- | --- | --- |
| 1. Standard tensor inference / 标准张量推理 | Extend the B1/B2/B3 baseline for declared device, dtype, copy, view/alias and mutation semantics. / 沿已有基线补齐声明支持的设备、类型、拷贝、视图/别名和修改语义。 | A small ordinary PyTorch program works without application patches; CPU comparisons pass; unsupported combinations are rejected or use verified, visible fallback; resource/version behavior is correct. / 独立普通 PyTorch 小程序无需应用补丁即可运行，CPU 对照通过，不支持组合明确处理，资源与版本行为正确。 |
| 2. Default inference operators / 默认推理算子 | Close reusable Linear/Conv/Norm and standard SDPA gaps for selected workloads. / 针对选定负载补齐可复用算子和标准 SDPA 语义。 | Supported workloads run without advanced plugins; no ignored parameters; native and fallback paths and transfer costs are identified. / 默认路径可用、不忽略参数，明确原生/回退路径与传输成本。 |
| 3. Application and runtime acceptance / 应用与运行时验收 | Verify the same capabilities through an independent program, ComfyUI and GPT-SoVITS with matched isolated runtimes. / 在匹配的隔离运行时中，用独立程序和两个应用验证同一组能力。 | Loading, execution, cancellation/errors and unloading behave correctly; preserve existing GSV regressions and complete the remaining ComfyUI flow. / 加载、执行、取消/异常、卸载正确，保留已有 GSV 回归并补完 ComfyUI 流程。 |
| 4. Reproducible delivery and wider coverage / 可复现交付与扩大覆盖 | Document dependencies, runtime identity, known limits and repeatable checks; qualify additional vendors and operating systems separately. / 明确依赖、实际运行时、限制和复现步骤，逐项验证更多厂商与系统。 | Publish support only for measured configurations; distinguish native, hybrid, CPU fallback, unsupported and unverified cases. / 只对实际验收配置声明支持，区分原生、混合、CPU 回退、不支持与未验证。 |

### Next scoped target: B4 device/dtype copies / 下一步：B4 设备与类型拷贝

Source inspection shows that native-buffer `copy_` currently requires FP32 on
both sides. The next proposed increment keeps Vulkan storage FP32 and adds
CPU-side FP16/FP64 conversion at upload/download boundaries, through ordinary
`.to(device, dtype)` and `copy_` calls. This is planned work, not a new supported
capability or native GPU FP16/FP64 storage.

源码核对发现，原生 buffer 的 `copy_` 目前要求两端均为 FP32。下一步建议保持
Vulkan 存储为 FP32，在上传/下载边界补齐 CPU FP16/FP64 转换，让普通
`.to(device, dtype)` 和 `copy_` 调用满足预期。这是待实施工作，不是新增已支持
能力，也不是 GPU 原生 FP16/FP64 存储支持。

Start with CPU-reference cases for CPU FP64 to Vulkan FP32, Vulkan FP32 to CPU
FP64, and a noncontiguous CPU FP16 destination. Preserve destination identity,
view/alias relationships and version behavior; verify temporary lifetimes and
safe rejection without mutating the destination. Reuse B1/B2/B3 regressions
and validate only in a matched isolated runtime with small UMA inputs.

先固定三类 CPU 对照：CPU FP64 转 Vulkan FP32、Vulkan FP32 转 CPU FP64、
以及写入非连续 CPU FP16 目标。验收须保留目标对象、视图/别名关系与正确版本
行为，保证临时资源存活安全，不支持的操作应在修改目标之前拒绝。复用 B1/B2/B3
回归，在匹配的隔离运行时中用小型 UMA 输入验证。

A previously passing tiny chain is not evidence that the broader tensor
milestone is complete. Full native training and heterogeneous multi-GPU
research have separate acceptance criteria and do not block a bounded
inference release.

不能仅凭已经通过的小计算链宣称标准张量阶段完成。完整原生训练与异构多 GPU
研究分别规划和验收，不作为受限推理版本交付的前置条件。

## Backend and optional policies / 本体与可选策略

The backend owns tensor semantics, execution dependencies, basic allocation,
safe reclamation, errors and honest resource accounting. C++ owns Vulkan
resources; Rust can provide accounting-based budgets and admission advice.
Model placement, unloading, prefetch and advanced attention strategies belong
to applications or optional plugins. Plugins must respect live references and
in-flight work. Supported workloads need a usable default path without them.

本体负责张量语义、执行依赖、基础分配、安全回收、错误传播与真实资源统计。
C++ 管理 Vulkan 资源，Rust 可提供基于记账的预算与安全准入建议。模型驻留、
卸载、预取和高级注意力策略由应用或可选插件选择，并遵守有效引用与执行中任务
的约束。已经承诺支持的工作负载必须有不依赖高级插件的可用默认路径。

## Validation boundaries / 验证边界

GPU experiments currently stay within small, interruptible UMA workloads.
Wheel production, production-runtime migration, real discrete-GPU performance
and 1080P long-workload acceptance remain separate deferred work. Simulation
is not real hardware qualification. This roadmap does not change those bounds
or promise identical performance on different GPUs.

当前 GPU 实验保持小规模、可中止的 UMA 范围。wheel 构建、生产运行时迁移、
真实独显性能和 1080P 长负载验收仍为单独待推进事项；模拟不能替代真机验证。
本路线图不改变这些边界，也不承诺不同 GPU 性能相同。
