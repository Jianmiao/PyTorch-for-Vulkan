# INT4 仅权重量化格式契约设计

日期：2026-08-22

## 目标

增加一个有版本号的、基于 CPU 参考实现的 INT4 仅权重量化数据契约，
让 Vulkan SSBO 后端能够验证数据、执行解码，并在任何不支持或不安全的
情况下回退到现有 FP32 权重路径。本阶段只处理兼容性和格式，不启用
INT4 Vulkan feature、storage、shader、pipeline，也不产生 INT4 性能结论。

## 范围

实现文件放在现有 INT8 契约旁边：
`tools/vulkan_ssbo/int4_weight_only.py`；独立回归为
`test_int4_weight_only_contract_runtime.py`。UMA 和离散策略软件模拟都必须
运行同一套测试。UMA 入口和离散模拟入口会在 INT8 契约测试之后调用该回归。

该契约只有两种明确结果：

1. `int4_weight_only`：数据有效，解码为连续的 FP32 权重，交给现有算子路径。
2. `fp32_fallback`：数据损坏、不支持或形状不匹配，返回原始权重的连续 FP32 副本。

任何非法 payload 都不能隐式进入未来的 INT4 dispatch。

## v1 格式

payload 使用冻结字段的数据类表示：

```text
qweight          uint8 tensor [out_features, ceil(in_features / 2)]
scale            float32 tensor [out_features, ceil(in_features / 32)]
zero_point       int16 tensor [out_features, ceil(in_features / 32)]
in_features      positive integer
block_size       integer 32
format_version   integer 1
scheme           "affine"
scale_granularity "per_output_channel_block"
nibble_order     "low_even_high_odd"
```

每个字节保存两个无符号 4-bit code。输入列 `2*j` 使用低 nibble，输入列
`2*j+1` 使用高 nibble。当 `in_features` 为奇数时，最后一个高 nibble
被忽略。code 范围为 `[0, 15]`。对于 block `b`，解码公式为：

```text
decoded = (code - zero_point[row, b]) * scale[row, b]
```

打包器按输出通道的 block 计算 affine min/max 参数。全零 block 使用
`scale=1`、`zero_point=0`，确保零值可以精确表示。v1 打包器始终生成
连续 tensor，不接受调用方传入的 zero-point。

## 验证与回退

以下情况必须被拒绝，并返回稳定的原因字符串：

- 版本、scheme、granularity、nibble order 或 block size 错误；
- 字段不是 tensor，或 rank、dtype、shape 错误，或 storage 非连续；
- `in_features` 与原始权重或打包字节宽度不一致；
- scale 不是有限值或小于等于零；
- zero-point 超出 `[0, 15]`，或 shape 与 scale 不一致；
- qweight 不符合 uint8/nibble 表示范围。

`resolve_weight(original_weight, payload)` 返回 `(weight, mode, reason)`。
有效 payload 返回连续 FP32 解码结果，并设置
`mode="int4_weight_only"`。任何验证失败或 shape 不匹配，都返回原始权重
转换后的连续 FP32，并设置 `mode="fp32_fallback"`。不得修改原始 tensor。

## 数值门槛

回归使用固定随机种子和代表性 Linear shape：
`128x256`、`1024x1280`、`4096x4096`。记录
`F.linear(input, dequantized_weight)` 相对于原始 FP32 参考结果的相对 L2
误差。v1 候选门槛为每个代表性 shape 的相对 L2 均 `< 0.10`。

测试还会用解码后的权重执行一次小型 SSBO Linear，并要求满足现有
FP16 storage 容差（`max_abs < 3e-2`）。这只是 bridge/fallback 检查，
不是 INT4 dispatch 性能基准。

非法 payload 必须解析为 FP32 fallback，并在相同的 SSBO 容差内复现原始
权重参考结果。测试必须确认清理后没有残留 resident entry。

## 能力与策略边界

- 本阶段不新增启用态的 `int4_weight_only_enabled` 运行时能力字段；格式模块
  只运行在 CPU 上。
- 现有 `shaderInt8` 和 8-bit storage 字段仍然只用于诊断。
- 在单独的 kernel 设计获批前，不要求 INT4 shader 编译探针。
- `VKSSBO_SIMULATE_DISCRETE=1` 只改变策略和拓扑行为，不能因此把 INT4
  变成 GPU dispatch，也不能把模拟耗时当作性能结果。
- INT8、BF16、FP16、native SSBO、torch 14/14 和现有融合回归仍是强制门槛。

## 失败处理与报告

测试在新进程中运行 Vulkan 子进程，输出一条最终 JSON 记录，并使用现有的
受控 Windows 终止方式，避免 native Vulkan teardown 崩溃。JSON 记录格式
元数据、每个 shape 的相对 L2 误差、SSBO bridge 误差、fallback 原因和
resident entry 数量。只有所有断言和子进程退出码都成功时，父进程才打印
`INT4 weight-only format/fallback regression: PASS`。

## 交付物

- `int4_weight_only.py`：格式、打包、验证和 FP32 fallback 模块；
- `test_int4_weight_only_contract_runtime.py`：契约和 bridge 回归；
- UMA 和离散模拟入口集成；
- 任务书结果记录，明确写出 INT4 Vulkan storage、kernel 和性能路径仍关闭。

## 非目标

- 不实现 INT4 Vulkan shader、feature 启用或 pipeline；
- 不改变 INT4 resident storage 或异步传输策略；
- 不构建 wheel，不制作外部整合包；
- 不根据 CPU 解码或离散模拟得出性能结论。
