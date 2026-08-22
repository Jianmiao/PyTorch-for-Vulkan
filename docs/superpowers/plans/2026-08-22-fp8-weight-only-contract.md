# FP8 仅权重量化契约实现计划

> **给 agent worker 的说明：** 必须按任务逐项执行；每个步骤使用复选框跟踪。实现时优先使用 TDD，先让契约测试失败，再写最小实现。

**目标：** 增加 E4M3FN/E5M2 两种 FP8 仅权重量化的 CPU 格式契约、解码和 FP32 fallback，并把兼容性回归接入 UMA 与离散策略模拟入口。

**架构：** 新增独立的 fp8_weight_only.py，使用固定 256 项软件 codebook 处理 FP8 编码和解码，不依赖 Vulkan FP8 feature 或现有 runtime dtype 选择。新增隔离的 runtime contract test 验证格式、误差、SSBO bridge 和清理；两个 BAT 入口只调用该测试。

**技术栈：** Python 3.13、PyTorch CPU tensor、现有 vkssbo.py ctypes bridge、Vulkan SSBO DLL、Windows BAT、固定种子 JSON 回归。

---

## 文件边界

- 新建 D:/ComfyUI Vulkan beat/tools/vulkan_ssbo/fp8_weight_only.py：FP8 payload 数据类、E4M3FN/E5M2 variant 规格、256 项 codebook、pack/validate/decode/fallback。
- 新建 D:/ComfyUI Vulkan beat/tools/vulkan_ssbo/test_fp8_weight_only_contract_runtime.py：父进程隔离、子进程 PyTorch/SSBO 验证、JSON schema 和受控退出。
- 修改 D:/PyTorch for Vulkan/pytorch-src/run_uma_test.bat：在 INT4 contract 后调用 FP8 contract。
- 修改 D:/PyTorch for Vulkan/pytorch-src/run_discrete_simulation_test.bat：在 INT4 simulation 后调用 FP8 simulation。
- 修改 D:/PyTorch for Vulkan/pytorch-src/VULKAN_SSBO_DEVELOPMENT_TASKBOOK_ZH.md：记录两种变体和回归证据。
- 不修改 Vulkan C++、Rust、device_info、shader、async、resident 或 autotune 实现；不构建 wheel。

### 任务 1：先写失败的 FP8 契约测试

**文件：**
- 新建 D:/ComfyUI Vulkan beat/tools/vulkan_ssbo/test_fp8_weight_only_contract_runtime.py

- [ ] 步骤 1：写隔离入口和基本 payload 断言。

测试支持默认父进程和 --child 子进程。子进程固定 torch.manual_seed(43)，导入 fp8_weight_only，构造 [7,35] 权重，分别调用 pack_weight(weight, "e4m3fn") 和 pack_weight(weight, "e5m2")。断言 format_version=1、block_size=32、scale_granularity="per_output_channel_block"、scale_mode="max_abs_to_finite"、special_values="finite_only"；qweight 为 torch.uint8、shape=(7,35)、连续；scale 为 torch.float32、shape=(7,2)、连续；variant 与调用一致。

- [ ] 步骤 2：写已知 code、变体和奇数列测试。

使用手工 payload 验证固定 codebook：E4M3FN 的 0x38/0xb8/0x30/0x40 分别解码为 1/-1/0.5/2；E5M2 的 0x3c/0xbc/0x38/0x40 分别解码为 1/-1/0.5/2。额外验证 E4M3FN 的 0x7e 为 448、E5M2 的 0x7b 为 57344。使用 in_features=3 的 payload 确认一个字节一个值，不存在 nibble 尾部问题。

- [ ] 步骤 3：写 CPU Linear 误差门槛。

对 shape 128x256、1024x1280、4096x4096，分别生成固定输入和权重，两个 variant 都执行 resolve_weight 后的 F.linear A/B。每个 E4M3FN 相对 L2 必须小于 0.08；每个 E5M2 相对 L2 必须小于 0.15；JSON 保存每个 variant/shape 的误差。

- [ ] 步骤 4：写非法输入和 fallback 断言。

覆盖非法 variant、format_version、block_size、scale_granularity、scale_mode、special_values、qweight dtype/shape/连续性、scale dtype/shape/连续性、非有限或非正 scale、原始权重 shape、payload 含特殊 code。每个非法 payload 必须返回连续 FP32、mode="fp32_fallback" 和稳定 reason；pack_weight 对 NaN/Inf 源权重必须抛出 ValueError("nonfinite_weight")。

- [ ] 步骤 5：运行失败测试。

运行：

    & 'D:/ComfyUI Vulkan beat/tools/Python313/python.exe' 'D:/ComfyUI Vulkan beat/tools/vulkan_ssbo/test_fp8_weight_only_contract_runtime.py'

预期：失败原因是 fp8_weight_only 模块或导入符号不存在，而不是 Vulkan 初始化或 Python 环境错误。

### 任务 2：实现固定 FP8 软件 codebook

**文件：**
- 新建 D:/ComfyUI Vulkan beat/tools/vulkan_ssbo/fp8_weight_only.py

- [ ] 步骤 1：定义 payload 和 variant 常量。

定义 Fp8WeightOnly(qweight, scale, variant, in_features, block_size=32, format_version=1, scale_granularity="per_output_channel_block", scale_mode="max_abs_to_finite", special_values="finite_only")。variant 只接受 e4m3fn/e5m2。常量为 E4M3FN_MAX_FINITE=448.0、E5M2_MAX_FINITE=57344.0。

- [ ] 步骤 2：实现 256 项 decode table。

对每个 code 拆出 sign、exponent、mantissa。E4M3FN 使用 exponent_bits=4、mantissa_bits=3、bias=7；exponent=15 且 mantissa=7 的正负 code 标记为 NaN，其余 finite code 按正常数和次正规数规则解码。E5M2 使用 exponent_bits=5、mantissa_bits=2、bias=15；exponent=31 且 mantissa=0 标记 Inf，mantissa>0 标记 NaN，其余 finite code 解码。保存 finite_mask、decode_values 和每个 code 的 mantissa parity，供验证和 ties-to-even 编码使用。

- [ ] 步骤 3：实现最近值编码。

对有限 FP32 normalized value，在对应 variant 的 finite decode table 中按绝对距离选择最近 code；负值使用符号位后的正值候选。距离相等时选择 mantissa 最低位为 0 的候选；若仍相等选择较小 code，保证结果确定。该编码只用于 CPU 参考，不追求运行时速度。

- [ ] 步骤 4：实现 pack_weight。

只接受二维 floating-point tensor；转换为连续 FP32，若不是有限值则抛出 ValueError("nonfinite_weight")。按每个输出通道每 32 列计算 max_abs；非零 block 使用 max_abs/max_finite，零 block 使用 1.0。逐元素编码为 uint8，输出 qweight [rows, cols] 和 scale [rows, ceil(cols/32)]，所有输出连续。

- [ ] 步骤 5：实现严格验证、特殊 code 拒绝和 resolve_weight。

验证顺序固定为 payload 类型、版本、variant、metadata、字段类型、rank/dtype、连续性、原始权重 shape、qweight/scale shape、scale finite/positive、qweight special code。特殊 code reason 必须区分 e4m3fn_nan_code、e5m2_inf_code、e5m2_nan_code。失败返回 original_weight.float().contiguous(), "fp32_fallback", reason；成功按 codebook 解码并乘 block scale，返回连续 FP32、"fp8_weight_only" 和 supported_e4m3fn 或 supported_e5m2。

- [ ] 步骤 6：运行契约测试和编译检查。

运行：

    & 'D:/ComfyUI Vulkan beat/tools/Python313/python.exe' -m py_compile 'D:/ComfyUI Vulkan beat/tools/vulkan_ssbo/fp8_weight_only.py' 'D:/ComfyUI Vulkan beat/tools/vulkan_ssbo/test_fp8_weight_only_contract_runtime.py'
    & 'D:/ComfyUI Vulkan beat/tools/Python313/python.exe' 'D:/ComfyUI Vulkan beat/tools/vulkan_ssbo/test_fp8_weight_only_contract_runtime.py'

预期：编译成功，两个 variant 的误差门槛通过，父进程输出 FP8 weight-only format/fallback regression: PASS。

### 任务 3：增加 SSBO bridge 和资源清理证据

**文件：**
- 修改 D:/ComfyUI Vulkan beat/tools/vulkan_ssbo/test_fp8_weight_only_contract_runtime.py

- [ ] 步骤 1：增加合法 payload bridge。

子进程用固定 [3,11] 输入和 [7,11] 权重分别测试 E4M3FN/E5M2，调用 vk.linear(input, decoded_weight)，与 torch.nn.functional.linear(input, decoded_weight) 比较，两个 variant 都要求 max_abs < 3e-2。

- [ ] 步骤 2：增加 fallback bridge 和清理断言。

至少使用错误 variant、错误 scale 和特殊 code 三个 payload 调用 resolve_weight，再把返回的 FP32 权重送入 vk.linear，与原始权重 CPU reference 比较并要求 max_abs < 3e-2。测试末尾读取 vk.memory_stats()，要求 resident_entries=0。

- [ ] 步骤 3：输出稳定 JSON。

JSON 至少包含 schema_version=1、format_version、variants、error_rows、supported_max_abs、fallback_reasons、resident_entries。父进程只解析最后一行 JSON，child rc 必须为 0。

### 任务 4：接入 UMA 和离散策略模拟入口

**文件：**
- 修改 D:/PyTorch for Vulkan/pytorch-src/run_uma_test.bat
- 修改 D:/PyTorch for Vulkan/pytorch-src/run_discrete_simulation_test.bat

- [ ] 步骤 1：在 UMA 入口接入 FP8 contract。

在现有 INT4 weight-only contract 调用后加入：

    if exist "%VKSSBO_TEST_ROOT%\test_fp8_weight_only_contract_runtime.py" call :run_required "FP8 weight-only contract" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_fp8_weight_only_contract_runtime.py"

保持 UMA 设备选择、VKSSBO_SIMULATE_DISCRETE=0、日志和退出码逻辑不变。

- [ ] 步骤 2：在离散模拟入口接入 FP8 simulation。

在现有 INT4 weight-only simulation 调用后加入同一形式，标签使用 FP8 weight-only simulation。保留现有模拟拓扑、强制 staging、虚拟预算和 async timeline 环境变量；不得把模拟耗时写入 FP8 性能结论。

- [ ] 步骤 3：运行两个入口。

运行：

    & 'D:/PyTorch for Vulkan/pytorch-src/run_uma_test.bat'
    & 'D:/PyTorch for Vulkan/pytorch-src/run_discrete_simulation_test.bat'

预期：两份日志最后均为 RESULT: PASS，并包含对应 FP8 PASS；不得运行真实 dGPU 性能入口，不得构建 wheel。

### 任务 5：更新任务书并完成全量验收

**文件：**
- 修改 D:/PyTorch for Vulkan/pytorch-src/VULKAN_SSBO_DEVELOPMENT_TASKBOOK_ZH.md

- [ ] 步骤 1：记录 FP8 证据。

记录 E4M3FN/E5M2、32 元素 block、max_abs scale、误差门槛和实际结果，说明 NaN/Inf 与特殊 code 的 fallback，以及 UMA/离散模拟入口均 PASS。

- [ ] 步骤 2：重跑已有强制回归。

确认 native SSBO、torch 14/14、FP16 resident/boundary、BF16 fallback、INT8 capability/storage/probe/contract、INT4 contract、融合和离散策略模拟仍通过。复用入口日志，不改变默认 async 和设备选择。

- [ ] 步骤 3：最终检查并提交。

运行：

    git -C 'D:/PyTorch for Vulkan/pytorch-src' diff --check
    git -C 'D:/PyTorch for Vulkan/pytorch-src' status --short

确认没有意外修改、没有 wheel 产物、备份目录存在，并提交格式实现、入口接入和任务书记录。最终报告必须区分 UMA 真实兼容性、离散策略模拟兼容性和 FP8 性能尚未测试。

### 提交节奏

- [ ] 契约失败测试提交。
- [ ] 软件 codebook/格式实现和绿色契约测试提交。
- [ ] BAT 入口与任务书证据提交。
