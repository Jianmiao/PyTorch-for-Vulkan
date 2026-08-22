# INT4 仅权重量化契约实现计划

> **给 agent worker 的说明：** 必须按任务逐项执行；每个步骤使用复选框跟踪。实现时优先使用 TDD，先让契约测试失败，再写最小实现。

**目标：** 增加 INT4 仅权重量化的 CPU 格式契约、FP32 解码与安全回退，并把兼容性回归接入 UMA 和离散策略模拟入口，同时保持 INT4 Vulkan 路径关闭。

**架构：** 在 `tools/vulkan_ssbo` 中新增独立的 `int4_weight_only.py`，它只负责 v1 payload 的数据类、打包、验证和 FP32 解码。新增 runtime contract test 在隔离子进程中检查格式、数值门槛、SSBO bridge 和清理；两个 BAT 入口只负责调用该测试，不改变设备选择或默认 async 策略。

**技术栈：** Python 3.13、PyTorch、现有 `vkssbo.py` ctypes bridge、Vulkan SSBO DLL、Windows BAT、固定种子 JSON 回归。

---

## 文件边界

- 新建 `D:/ComfyUI Vulkan beat/tools/vulkan_ssbo/int4_weight_only.py`：只包含 `Int4WeightOnly`、`pack_weight`、`resolve_weight` 和内部验证/解码辅助函数；不得加载 Vulkan 或修改全局环境。
- 新建 `D:/ComfyUI Vulkan beat/tools/vulkan_ssbo/test_int4_weight_only_contract_runtime.py`：父进程负责隔离执行和 JSON 断言，子进程负责 PyTorch/SSBO 操作；复用现有受控 Windows 终止模式。
- 修改 `D:/PyTorch for Vulkan/pytorch-src/run_uma_test.bat`：在 INT8 weight-only contract 后调用 INT4 contract。
- 修改 `D:/PyTorch for Vulkan/pytorch-src/run_discrete_simulation_test.bat`：在 INT8 weight-only simulation 后调用 INT4 simulation；不改变模拟环境变量。
- 修改 `D:/PyTorch for Vulkan/pytorch-src/VULKAN_SSBO_DEVELOPMENT_TASKBOOK_ZH.md`：记录通过证据，并明确 INT4 Vulkan storage/kernel/performance 仍关闭。
- 不修改 Vulkan C++/Rust 实现，不增加 `device_info()` 能力字段，不生成 shader，不构建 wheel。

### 任务 1：先写失败的 INT4 契约测试

**文件：**
- 新建：`D:/ComfyUI Vulkan beat/tools/vulkan_ssbo/test_int4_weight_only_contract_runtime.py`

- [ ] **步骤 1：写测试入口和固定格式断言。**

测试支持默认父进程和 `--child` 子进程。子进程固定 `torch.manual_seed(37)`，构造 weight `[7, 35]`，验证 `pack_weight(weight)` 产生：`format_version=1`、`scheme="affine"`、`scale_granularity="per_output_channel_block"`、`nibble_order="low_even_high_odd"`、`block_size=32`、`in_features=35`、`qweight.dtype=torch.uint8`、`qweight.shape=(7,18)`、`scale/zero_point.shape=(7,2)`，并且所有输出连续。

- [ ] **步骤 2：写数值和 nibble 顺序断言。**

调用 `resolve_weight(weight, packed)`，要求 mode 为 `int4_weight_only`、reason 为稳定的有效原因、输出为连续 FP32；用三个固定 shape `128x256`、`1024x1280`、`4096x4096` 检查 `F.linear` 相对 L2 均小于 `0.10`。测试显式覆盖奇数 `in_features`，确认最后一个高 nibble 不影响输出。

- [ ] **步骤 3：写非法 payload fallback 断言。**

逐个检查错误版本、错误 block size、非连续 qweight、错误 qweight dtype、非连续 scale、非有限 scale、非正 scale、zero-point 越界、zero-point shape 错误、`in_features` 不匹配、原始权重 shape 不匹配。每个 case 都必须返回原始权重的连续 FP32、`mode="fp32_fallback"`、预期 reason，并且不得修改原始 weight。

- [ ] **步骤 4：运行测试，确认它因模块不存在而失败。**

运行：

```powershell
& 'D:\ComfyUI Vulkan beat\tools\Python313\python.exe' 'D:\ComfyUI Vulkan beat\tools\vulkan_ssbo\test_int4_weight_only_contract_runtime.py'
```

预期：失败，错误应指向 `int4_weight_only` 模块或其导入符号不存在，而不是环境初始化失败。

### 任务 2：实现 v1 打包、验证和 FP32 解码

**文件：**
- 新建：`D:/ComfyUI Vulkan beat/tools/vulkan_ssbo/int4_weight_only.py`

- [ ] **步骤 1：定义冻结 payload 和常量。**

实现以下接口和字段：

```python
@dataclass(frozen=True)
class Int4WeightOnly:
    qweight: torch.Tensor
    scale: torch.Tensor
    zero_point: torch.Tensor
    in_features: int
    block_size: int = 32
    format_version: int = 1
    scheme: str = "affine"
    scale_granularity: str = "per_output_channel_block"
    nibble_order: str = "low_even_high_odd"

def pack_weight(weight: torch.Tensor) -> Int4WeightOnly: ...
def resolve_weight(original_weight: torch.Tensor,
                   payload: Int4WeightOnly) -> tuple[torch.Tensor, str, str]: ...
```

`pack_weight` 只接受 2D floating-point weight，先转为连续 FP32；按每行每 32 列计算 min/max；正常 block 使用 `scale=(max-min)/15`、`zero_point=round(-min/scale).clamp(0,15)`，全零 block 使用 `scale=1`、`zero_point=0`。将 code 转为 uint8，偶数列放低 nibble、奇数列放高 nibble，奇数列尾部的高 nibble 置零。

- [ ] **步骤 2：实现不依赖 Vulkan 的严格验证。**

内部验证函数按稳定优先级返回字符串或 `None`。验证顺序固定为元数据、字段类型、rank/dtype、连续性、`in_features` 与 shape、packed byte 宽度、scale/zero-point shape、finite/范围。qweight 是 uint8 时其每个 byte 天然覆盖两个 nibble，因此只需验证 dtype、shape 和连续性。

- [ ] **步骤 3：实现解码和安全回退。**

解码时从低/高 nibble 恢复 `[out_features, in_features]` code，按 block 索引选择 scale/zero-point，再计算 `(code - zero_point) * scale`，结果必须是连续 FP32。验证失败时返回 `original_weight.float().contiguous()`、`"fp32_fallback"` 和对应 reason；不能原地修改输入。

- [ ] **步骤 4：运行契约测试使其通过。**

运行任务 1 的命令。预期输出包含 `INT4 weight-only format/fallback regression: PASS`，并输出 JSON 中的 per-shape relative L2、fallback reasons 和 `resident_entries=0`。

### 任务 3：补充 SSBO bridge 与资源清理检查

**文件：**
- 修改：`D:/ComfyUI Vulkan beat/tools/vulkan_ssbo/test_int4_weight_only_contract_runtime.py`

- [ ] **步骤 1：增加小型 SSBO Linear bridge。**

子进程构造固定 `[3, 11]` 输入和 `[7, 11]` 权重，使用合法 payload 解码后的 FP32 权重调用现有 `vk.linear(x, dequantized)`，与 `torch.nn.functional.linear(x, dequantized)` 比较，要求 `max_abs < 3e-2`。对至少两个非法 payload 使用 fallback 权重调用 `vk.linear`，与原始权重 CPU reference 比较并要求同一门槛。

- [ ] **步骤 2：增加清理和 JSON schema 断言。**

子进程末尾读取 `vk.memory_stats()`，要求 `resident_entries == 0`；输出包含 `schema_version=1`、格式字段、`error_rows`、`supported_max_abs`、`fallback_reasons` 和 `resident_entries`。父进程必须检查子进程退出码为 0，并只解析最后一行 JSON，兼容 backend 诊断输出。

- [ ] **步骤 3：运行隔离测试和 Python 编译检查。**

运行：

```powershell
& 'D:\ComfyUI Vulkan beat\tools\Python313\python.exe' -m py_compile 'D:\ComfyUI Vulkan beat\tools\vulkan_ssbo\int4_weight_only.py' 'D:\ComfyUI Vulkan beat\tools\vulkan_ssbo\test_int4_weight_only_contract_runtime.py'
& 'D:\ComfyUI Vulkan beat\tools\Python313\python.exe' 'D:\ComfyUI Vulkan beat\tools\vulkan_ssbo\test_int4_weight_only_contract_runtime.py'
```

预期：编译成功，回归输出 PASS，child rc=0。

### 任务 4：接入 UMA 和离散策略模拟入口

**文件：**
- 修改：`D:/PyTorch for Vulkan/pytorch-src/run_uma_test.bat`
- 修改：`D:/PyTorch for Vulkan/pytorch-src/run_discrete_simulation_test.bat`

- [ ] **步骤 1：在 UMA 入口加入 INT4 contract。**

在现有 `test_int8_weight_only_contract_runtime.py` 调用之后加入：

```bat
if exist "%VKSSBO_TEST_ROOT%\test_int4_weight_only_contract_runtime.py" call :run_required "INT4 weight-only contract" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_int4_weight_only_contract_runtime.py"
```

保持设备选择、`VKSSBO_SIMULATE_DISCRETE=0`、日志路径和失败传播逻辑不变。

- [ ] **步骤 2：在离散模拟入口加入 INT4 contract。**

在现有 INT8 simulation 调用之后加入同一 `:run_required` 形式，标签使用 `INT4 weight-only simulation`。保留现有模拟环境变量；测试结果只能说明策略兼容性。

- [ ] **步骤 3：执行两个入口并检查退出码。**

运行：

```powershell
& 'D:\PyTorch for Vulkan\pytorch-src\run_uma_test.bat'
& 'D:\PyTorch for Vulkan\pytorch-src\run_discrete_simulation_test.bat'
```

预期：两份日志最后均为 `RESULT: PASS`，各自包含 INT4 PASS；不得构建 wheel，不得运行真实 dGPU 性能入口。

### 任务 5：更新任务书并完成全量回归

**文件：**
- 修改：`D:/PyTorch for Vulkan/pytorch-src/VULKAN_SSBO_DEVELOPMENT_TASKBOOK_ZH.md`

- [ ] **步骤 1：在阶段 F 当前状态中记录 INT4 证据。**

记录 v1 格式、32 元素 block、CPU FP32 解码、非法 payload FP32 fallback、UMA 与离散模拟入口 PASS，并明确 `INT4 Vulkan storage/kernel/performance` 仍未启用。

- [ ] **步骤 2：运行现有强制回归。**

确认 native SSBO、torch 14/14、FP16 resident、BF16 fallback、INT8 capability/storage/probe/contract、融合回归和离散策略模拟均未被破坏。复用入口日志，不修改默认 async 或设备选择。

- [ ] **步骤 3：做最终差异和退出码检查。**

运行：

```powershell
git -C 'D:\PyTorch for Vulkan\pytorch-src' diff --check
git -C 'D:\PyTorch for Vulkan\pytorch-src' status --short
```

确认没有意外修改、没有 wheel 产物、备份目录存在，并在最终报告中区分 UMA 真实兼容性、离散策略模拟兼容性、INT4 性能尚未测试。

### 提交节奏

- [ ] 契约测试提交：只包含测试文件的失败测试。
- [ ] 格式实现提交：包含 `int4_weight_only.py` 和绿色契约测试。
- [ ] 入口与任务书提交：包含两个 BAT 接入和任务书证据。

每次提交前都要在 `D:/PyTorch for Vulkan/backups/` 创建当前相关文件的时间戳备份；不得回退其他 dirty changes。
