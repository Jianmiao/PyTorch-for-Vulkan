# FP4 仅权重量化格式契约设计

日期：2026-08-22

## 目标

增加一个有版本号的、基于 CPU 参考实现的 FP4 仅权重量化数据契约，
采用有限值 E2M1 codebook。契约负责编码、验证、解码和安全回退，让
现有 FP32/FP16 SSBO 算子可以使用解码后的 FP32 权重进行兼容性验证。

本阶段只处理格式和兼容性，不启用 FP4 Vulkan feature、storage、shader、
pipeline、resident storage 或性能路径。

## 文件与运行边界

新增：

- `tools/vulkan_ssbo/fp4_weight_only.py`
- `tools/vulkan_ssbo/test_fp4_weight_only_contract_runtime.py`

修改：

- `run_uma_test.bat`
- `run_discrete_simulation_test.bat`
- `VULKAN_SSBO_DEVELOPMENT_TASKBOOK_ZH.md`

FP4 模块只依赖 PyTorch CPU tensor，不加载 Vulkan，不修改全局环境，
不改变现有 `vkssbo.py` 的 dtype 选择。离散模拟只验证强制 staging、
预算、timeline 和 fallback 语义，不能用于得出 FP4 独显性能结论。

## v1 payload 格式

payload 使用冻结字段的数据类表示：

```text
qweight            uint8 tensor [out_features, ceil(in_features / 2)]
scale              float32 tensor [out_features, ceil(in_features / 32)]
in_features        正整数
format_version     1
scheme              "e2m1"
scale_granularity  "per_output_channel_block"
scale_mode         "max_abs_to_finite"
nibble_order       "low_even_high_odd"
block_size         32
```

每个 byte 保存两个 FP4 code：偶数列使用低 nibble，奇数列使用高
nibble。`in_features` 为奇数时，最后一个 byte 的高 nibble 必须为零；
非零 padding nibble 属于非法 payload 并触发 FP32 fallback。

## E2M1 codebook 和 scale

E2M1 使用 1 个 sign bit、2 个 exponent bit、1 个 mantissa bit、bias=1。
v1 不包含 NaN 或 Inf，16 个 code 全部表示有限值。正数 code 的固定
解码表为：

```text
0x0 ->  0.0    0x1 ->  0.5    0x2 ->  1.0    0x3 ->  1.5
0x4 ->  2.0    0x5 ->  3.0    0x6 ->  4.0    0x7 ->  6.0
```

`0x8` 到 `0xf` 为对应的负值；`0x8` 的负零不会由打包器生成，但
仍属于有限且可解码的 payload。`max_finite` 固定为 `6.0`。

对每个输出通道的 32 元素 block：

```text
scale = max_abs(block) / 6.0
```

全零 block 固定使用 `scale=1.0`。打包器不接受调用方传入的 scale，
避免同一 payload 出现无法比较的量化结果。

## 编码、解码和 fallback

编码流程：

1. 要求源权重为二维 floating-point tensor，并转换为连续 FP32。
2. 包含 NaN 或 Inf 时抛出 `ValueError("nonfinite_weight")`，调用方
   继续使用原始 FP32 权重。
3. 按 block 计算 scale。
4. 对 `weight / scale` 按绝对值选择最近的 E2M1 有限值；距离相等时
   优先 mantissa bit 为 0 的候选，再选择较小的 code，保证结果确定。
5. 将相邻两个 code 打包到一个 byte，输出连续 uint8 qweight 和
   连续 float32 scale。

解码公式为：

```text
decoded = fp4_decode(qweight) * scale
```

`resolve_weight` 必须按固定顺序验证 payload 类型、版本、scheme、
metadata、字段 dtype/rank/连续性、原始权重 shape、qweight/scale
shape、scale finite/positive、奇数列 padding nibble。任何失败都返回：

```text
(原始权重的连续 FP32 副本, "fp32_fallback", 稳定的失败原因)
```

合法结果为：

```text
(连续 FP32 解码权重, "fp4_weight_only", "supported_e2m1")
```

fallback 不修改原始 tensor，也不得创建 resident entry。

## 验证门槛

回归使用固定随机种子和三个代表性 Linear shape：

```text
128 x 256
1024 x 1280
4096 x 4096
```

固定 E2M1 scale 规则的 CPU Linear A/B 最大相对 L2 必须 `< 0.12`。
额外覆盖：

- 奇数 `in_features` 和零 padding nibble；
- 全零 block；
- 大动态范围权重；
- 非有限源权重；
- 错误版本、scheme、scale mode、nibble order 和 block size；
- qweight/scale 的 dtype、shape、连续性错误；
- scale 非有限、非正；
- 原始权重 shape 不匹配；
- 合法 payload 的 SSBO bridge，`max_abs < 3e-2`；
- 非法 payload 的 FP32 fallback bridge，同一误差门槛；
- 测试结束 `resident_entries=0`。

父进程只解析子进程最后一行 JSON，并使用现有受控 Windows 终止方式
避免 native Vulkan teardown 退出码污染。JSON 至少包含
`schema_version`、`format_version`、`scheme`、`error_rows`、
`supported_max_abs`、`fallback_reasons` 和 `resident_entries`。

## 能力与策略边界

- 不新增或启用 `fp4_supported`、`fp4_weight_only_enabled` 等能力字段。
- 不修改 FP16 native storage、BF16 fallback、INT8/INT4/FP8 fallback。
- 不生成 FP4 shader，不做 shader 编译探针。
- 不改变 async transfer、resident、autotune 或设备选择策略。
- 不把 CPU 编码/解码耗时或离散模拟耗时写成 FP4 性能结果。
- 不构建 Python wheel，不制作独显一次性性能整合包。

## 任务书记录

完成回归后，在 `VULKAN_SSBO_DEVELOPMENT_TASKBOOK_ZH.md` 记录：

- E2M1 codebook、4 bit nibble 打包和每 32 元素 scale 语义；
- 固定 shape 的最大相对 L2 和 SSBO bridge 误差；
- UMA 和离散模拟入口的退出码与 `RESULT: PASS`；
- 明确 FP4 Vulkan storage、shader、pipeline、resident 和性能路径仍关闭。
