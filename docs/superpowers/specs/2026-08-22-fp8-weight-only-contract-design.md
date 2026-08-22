# FP8 仅权重量化格式契约设计

日期：2026-08-22

## 目标

增加一个有版本号的、基于 CPU 参考实现的 FP8 仅权重量化数据契约，
同时覆盖 E4M3FN 和 E5M2 两种变体。契约负责编码、验证、解码和安全
回退，让现有 FP32/FP16 SSBO 算子可以使用解码后的 FP32 权重进行
兼容性验证。

本阶段只处理格式和兼容性，不启用 FP8 Vulkan feature、storage、
shader、pipeline、resident storage 或性能路径。

## 文件与运行边界

新增：

- tools/vulkan_ssbo/fp8_weight_only.py
- tools/vulkan_ssbo/test_fp8_weight_only_contract_runtime.py

FP8 模块只依赖 PyTorch CPU tensor，不加载 Vulkan，不修改全局环境，
不改变现有 vkssbo.py 的 dtype 选择。

两个运行入口共用同一回归：

- UMA 入口使用 FP8 weight-only contract 标签；
- 离散策略模拟入口使用 FP8 weight-only simulation 标签。

离散模拟只验证强制 staging、预算、timeline 和回退语义，不能用于
得出 FP8 独显性能结论。

## v1 payload 格式

payload 使用冻结字段的数据类表示：

text:
qweight            uint8 tensor [out_features, in_features]
scale              float32 tensor [out_features, ceil(in_features / 32)]
variant            "e4m3fn" 或 "e5m2"
in_features        正整数
block_size         32
format_version     1
scale_granularity  "per_output_channel_block"
scale_mode         "max_abs_to_finite"
special_values     "finite_only"

每个权重占用一个字节，不做 nibble 打包。每个输出通道的每 32 个输入
元素共享一个 FP32 scale，因此奇数 in_features 不需要额外的 nibble
尾部处理。

## Scale 规则

对每个输出通道的 block：

text:
scale = max_abs(block) / max_finite(variant)

全零 block 固定使用 scale=1。v1 使用连续 tensor；打包器不接受
调用方传入的 scale，避免同一 payload 出现无法比较的量化结果。

变体常量固定为：

text:
E4M3FN max_finite = 448
E5M2   max_finite = 57344

E4M3FN 为默认变体，适合精度优先的权重。E5M2 必须显式指定，适合
动态范围优先的场景。

## 编码和解码语义

编码流程：

1. 要求源权重为二维 floating-point tensor，并转换为连续 FP32。
2. 拒绝包含 NaN 或 Inf 的源权重，`pack_weight` 抛出稳定的
   `ValueError("nonfinite_weight")`；调用方保留原始 FP32 权重。
3. 按 block 计算 scale。
4. 用 value / scale 进行 FP8 编码。
5. 使用最近可表示值和 ties-to-even 舍入。
6. 输出 uint8 code 和 FP32 scale。

解码公式：

text:
decoded = fp8_decode(qweight) * scale

解码器使用固定的 CPU 参考规则，包含正常数、次正规数、符号位和零。
E4M3FN 的 NaN code、E5M2 的 Inf/NaN code 都不属于 v1 有效权重数据；
payload 出现这些特殊 code 时必须回退，而不是把非有限值送入 SSBO。

## 接口和结果

模块提供以下接口：

text:
@dataclass(frozen=True)
class Fp8WeightOnly:
    qweight: torch.Tensor
    scale: torch.Tensor
    variant: str
    in_features: int
    block_size: int = 32
    format_version: int = 1
    scale_granularity: str = "per_output_channel_block"
    scale_mode: str = "max_abs_to_finite"
    special_values: str = "finite_only"

def pack_weight(
    weight: torch.Tensor,
    variant: str = "e4m3fn",
) -> Fp8WeightOnly: ...

def resolve_weight(
    original_weight: torch.Tensor,
    payload: Fp8WeightOnly,
) -> tuple[torch.Tensor, str, str]: ...

resolve_weight 的合法结果为：

text:
(连续 FP32 解码权重,
 "fp8_weight_only",
 "supported_e4m3fn" 或 "supported_e5m2")

任何验证失败都返回：

text:
(原始权重的连续 FP32 副本,
 "fp32_fallback",
 稳定的失败原因)

`resolve_weight` 的 payload 回退不得修改原始 tensor，也不得创建
resident entry。源权重在 `pack_weight` 阶段被拒绝时，由调用方继续使用
原始 FP32 权重。

## 验证门槛

回归使用固定随机种子和三个代表性 Linear shape：

text:
128 x 256
1024 x 1280
4096 x 4096

两种变体都必须执行 CPU Linear A/B：

- E4M3FN 相对 L2 < 0.08；
- E5M2 相对 L2 < 0.15。

另外覆盖：

- 奇数 in_features；
- 全零 block；
- 大动态范围权重；
- NaN/Inf 源权重；
- 非法 variant；
- 错误版本、block size、scale mode 和 special-values 标记；
- qweight/scale 的 dtype、shape、连续性错误；
- scale 非有限、非正；
- 原始权重 shape 不匹配；
- payload 含特殊 Inf/NaN code。

合法 payload 的 SSBO bridge 使用解码后的 FP32 权重，要求现有
SSBO 误差门槛 max_abs < 3e-2。非法 payload 的 FP32 fallback
也必须满足同一门槛。测试结束后 resident_entries 必须为 0。

最终 JSON 至少包含：

text:
schema_version
format_version
variants
error_rows
supported_max_abs
fallback_reasons
resident_entries

父进程只解析子进程最后一行 JSON，并使用现有受控 Windows 终止方式
避免 native Vulkan teardown 退出码污染。

## 能力与策略边界

- 不新增或启用 fp8_supported、fp8_weight_only_enabled 等运行时
  能力字段。
- 不修改 FP16 native storage、BF16 fallback、INT8/INT4 fallback。
- 不生成 FP8 shader，不做 shader 编译探针。
- 不改变 async transfer、resident、autotune 或设备选择策略。
- 不把 CPU 编码/解码耗时或离散模拟耗时写成 FP8 性能结果。
- 不构建 Python wheel，不制作独显一次性性能整合包。

## 任务书记录

完成回归后，在 VULKAN_SSBO_DEVELOPMENT_TASKBOOK_ZH.md 记录：

- 两种 FP8 变体和 v1 scale 语义；
- E4M3FN/E5M2 的误差结果；
- UMA 和离散模拟入口的退出码与 RESULT: PASS；
- 明确 FP8 Vulkan storage、shader、pipeline、resident 和性能路径仍关闭。
