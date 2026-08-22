# PyTorch Vulkan SSBO 后端开发任务书

版本：2026-08-21  
适用范围：`D:\PyTorch for Vulkan\pytorch-src` 中的 PyTorch Vulkan SSBO 后端及其测试、基准和构建辅助文件。  
当前要求：持续开发和验证；**所有阶段完成并通过验收前，不构建 Python wheel**。

## 0. 我们现在要做什么

目标不是把所有算子简单塞进一个着色器，而是把 PyTorch 的 Vulkan 执行路径升级为一套可维护、可回退、按硬件适配的 SSBO 运行时：

1. 根据 Vulkan memory heap 判断 UMA、离散或混合内存拓扑。
2. UMA 使用持久映射和共享缓冲，离散 GPU 使用 device-local VRAM、staging ring 和异步传输。
3. 让 SSBO Tensor 在连续受支持算子之间保持 GPU resident，避免 `GPU -> CPU -> GPU` 往返。
4. 以计算图为单位规划临时激活内存，复用 buffer，限制权重和激活缓存。
5. 对常见模式做融合，对 kernel、tile、workgroup、精度和内存布局做设备特定自动调优。
6. 逐步加入 FP16、BF16、INT8、weight-only INT4、FP8/FP4 等低精度路径，并为每条路径保留误差门槛和回退实现。
7. 覆盖推理与训练需要的 Conv1D、Conv2D、Conv3D、LoRA/参数更新基础能力，同时保持标准 PyTorch fallback、autograd 和 CPU 语义。

最终服务对象是所有支持 Vulkan 的 GPU，包括不支持 CUDA、ROCm 或 XPU 的旧设备。性能目标是：在相同模型、步数和精度下，尽可能接近或超过 XPU；任何优化都必须以数值正确性和稳定性为前提。

## 1. 当前状态

已完成并保留：

- SSBO 基础设备、权重 LRU、瞬时 buffer pool 和 Rust 内存策略层。
- 5D+ buffer 布局及 CPU/GPU roundtrip。
- Conv1D grouped/stride/dilation、Conv2D grouped/stride/dilation、Conv3D。
- nearest/bilinear upsample、5D GroupNorm、native group norm 统计输出。
- Conv2D x32v4 默认 kernel；x32 scalar、x32 C32 保留为 A/B 变体。
- `test_torch.py` 14/14 和 native Vulkan SSBO regression 的既有通过状态。
- 增量构建和运行时 DLL 同步流程。

阶段 A 已完成：

- 已记录 Vulkan memory heap/type、budget/usage、拓扑和 device 能力。
- 已将拓扑字段接入 Rust policy snapshot，并增加 UMA/discrete/mixed 单元测试。
- 本机 Arc 驱动当前报告为单一共享 heap，分类为 `UMA`；若后续驱动暴露独立系统 heap，将分类为 `DISCRETE` 或 `MIXED`。

阶段 B 已完成并验收：

- UMA 路径采用 device-local + host-visible 持久映射，上传/下载执行显式 flush/invalidate。
- 离散/混合路径的 device-local buffer 通过 host-visible staging buffer 和 `vkCmdCopyBuffer` 上传、下载；transfer-to-compute barrier 已接入。当前仍是同步提交，异步队列与 timeline semaphore 留待阶段 D 在真实独显上验证。
- `memory_stats()` 已报告 `staging_bytes`、`staging_peak_bytes`、`staging_uploads`、`staging_downloads`，并继续报告 topology、heap、pool、weight 和 resident 指标。
- 验收结果（2026-08-20，Intel Arc UMA）：Rust memory policy 11/11、native Vulkan SSBO regression PASS、`test_torch.py` 14/14。

已完成（本轮）：

- UMA buffer 持久映射和显式 flush/invalidate；非 UMA 保留临时映射回退。
- 原生 SSBO bridge 的有界 resident mirror cache：输入命中时复用 SSBO，输出登记 LRU；`resident_bytes/entries/hits/misses` 纳入内存统计；`empty_cache()` 和 `vkssbo_free()` 联动清理。
- host lease 默认启用并带 TensorImpl 弱引用、版本检查；可用 `VKSSBO_HOST_LEASE=0` 关闭。
- Conv2D 新增 vec4 累加 kernel（32 个 fp32 累加槽不变），默认替代旧 x32v4；`VKSSBO_CONV_V4ACC=0` 可回退。当前 Arc 基准：SDXL 形状约再快 10%，VAE 大卷积约再快 5%。
- 原生 MatMul 改为 64x64 tiled shader（16x16 workgroup、每线程 4x4、FP32 累加、FP16 存储），覆盖 `mm`、`bmm` 和 `linear`。
- 已把 128x64 MatMul 作为独立候选 shader 嵌入；通用默认仍使用 64x64，`VKSSBO_MATMUL_128=1` 仅用于显式 A/B。不得把单一 GPU 的 shape 阈值写死进 PyTorch 默认路径；后续改为设备标识和能力键控的 autotune。
- `linear + bias` 已在同一个 MatMul dispatch 中融合；`VKSSBO_MATMUL_BIAS_FUSION=0` 可回退到旧 CPU bias 路径。
- `linear` 直接使用 PyTorch `[N,K]` 权重和 shader `transB=2` 语义，不再每次在 CPU 执行 `t().contiguous()`。
- 阶段 C 三条链路已完成 A/B 覆盖：`conv -> silu -> conv`、`linear -> bias -> silu`、`matmul -> softmax`；关闭 resident 时命中 0，开启后命中 4，三条输出 A/B 最大误差均为 0。
- 原生矩阵回归（Arc UMA）：`mm` max_abs=0.01529、`bmm`=0.01550、`linear`=0.01526、融合 bias `linear`=0.01539。
- 大 linear `2x256x1280 -> 10240`：融合加直接权重版在 ComfyUI bridge 基准中为 46.25 ms；128x64 候选只在部分大 shape/设备上获益，因此当前不作为通用默认。
- 原生 MatMul 自动调优首个稳定版本已经验收：候选为 64x64 与 128x64；键包含 vendor/device/driver、内存拓扑、FP16、共享内存、工作组能力、budget bucket、B/M/K/N、transB 和 bias。
- 大维度不再使用下一次幂粗分桶：`<=128` 按 8、`<=4096` 按 64、`>4096` 按 256；大 N 同时保留精确值，`N=4608` 与 `N=5120` 已验证会产生两个独立缓存条目。
- 每个新键使用 3 轮交错 A/B（64→128、128→64、64→128）和 GPU timestamp median；128x64 只有稳定快至少 3% 才会替代保守 64x64。
- timestamp delta 已按 `timestampValidBits` 处理回绕；输入上传 barrier 位于首个计时间隔之前；诊断接口新增 `benchmark_dispatches`。
- 调优前按实时 Vulkan heap budget/usage 为两份候选输出保留 10% 安全空间；内存紧张时跳过调优并执行 64x64。
- 持久化缓存使用跨进程文件锁、同目录临时文件和原子替换；两个并发进程分别调优 N=4608/5120 后，缓存保留两条记录。
- 2026-08-21 验收：focused cache `entries 0→1→2`、`tune_runs 0→1→2`、`benchmark_dispatches 0→6→12`；跨进程缓存 PASS；native Vulkan SSBO regression PASS；`test_torch.py` 14/14。
- 当前 Arc UMA 的交错外部 A/B（512x1280→5120，12 样本）：64x64 median 27.760 ms、p95 31.701 ms；128x64 median 28.641 ms、p95 32.010 ms。自动调优正确保留 64x64，避免把当前设备的负收益候选设为默认。
- Conv2D 自动调优已完成并通过验收：在已验证的 `conv2d_x32v4acc` 与 `conv2d_x32c32` 之间按完整设备/驱动/拓扑/能力/卷积 shape 建立独立有界缓存；C32 仅在 shared memory >= 40 KiB、C 可被 32 整除时参与比较，其他设备和 shape 保留原有 v4-acc/fallback 路径。
- Conv2D 调优采用 3 轮交错 `C32→V4Acc`、`V4Acc→C32`、`C32→V4Acc`，GPU timestamp median，C32 需稳定快至少 3% 才晋升；探针输出限制为 16x16，两个独立输出和 10% budget reserve 门控，六个 descriptor set 在提交完成后才释放。
- 新增 `torch.ops.vulkan_ssbo.conv2d_autotune_stats()` 与 `VKSSBO_CONV_AUTOTUNE_CACHE`；显式设置 `VKSSBO_CONV_X32/C32/V4/V4ACC` 或关闭 `VKSSBO_AUTOTUNE`、`VKSSBO_CONV_AUTOTUNE` 时完全跳过调优。新增 focused/concurrent/benchmark 回归，测试夹具固定输入以避免随机数导致的假失败。
- Conv2D focused：`entries 0→1→2`、`tune_runs 0→1→2`、`benchmark_dispatches 0→6→12`、重复 shape 命中且不新增 probe；并发两个 shape 的缓存均保留。调优失败 key 已加入有界进程内抑制，动态 budget 使用 256 MiB bucket，测试 reference 会把输入、权重和 bias 全部显式搬回 CPU。
- 2026-08-21 自动调优阶段最终复验：Conv2D/MatMul focused 与 concurrent cache、native regression、`test_torch.py` 14/14 全部 PASS；producer metadata/fusion 增量构建后的最新 SHA-256 见下一条。
- 本机 Arc UMA 外部基准（1x32x64x64→64，缓存命中，5 次）：自动路径 median 2.730 ms / p95 3.368 ms；手动 v4-acc median 2.954 ms / p95 5.598 ms，C32 median 3.249 ms / p95 7.293 ms；自动选择保守 v4-acc，最大 A/B 误差 0。p95 易受共享内存带宽和系统负载影响，不据单机数据写死通用默认。
- producer metadata 已接入 `SsboBackend`：记录 `conv1d/conv2d/conv3d/matmul/norm/elementwise/upsample/cat` 等生产者、元素数和版本，弱引用过期或普通 Tensor `_version()` 改变后查询返回 `valid=0`；入口为 `torch.ops.vulkan_ssbo.producer_metadata(Tensor)`。
- 新增显式 `torch.ops.vulkan_ssbo.conv2d_silu` building block：在满足已验证的 3x3/s1/d1/g1、FP16、16x16 workgroup、x32v4acc 能力边界时使用单 shader，否则安全回退为普通 Conv2D 后接 SiLU。Arc UMA `1x32x64x64 -> 32` A/B：unfused median 3.179 ms / p95 10.020 ms，fused median 2.754 ms / p95 3.370 ms，约 1.154x，max_abs 0.007812。此接口是显式融合 building block，不宣称已有自动图融合。
- 本轮构建后两套运行时 DLL SHA-256 均为 `4759DFA580F48EFE726587985C11C6DC0412FD1C38663D2EDD975E4AC75C86B0`；producer metadata、Conv2D autotune focused/concurrent、native regression 和 `test_torch.py` 14/14 全部 PASS。
- 新增显式 `torch.ops.vulkan_ssbo.linear_silu` building block：复用 64x64/128x64 MatMul FP32 累加与 FP16 存储，在同一 shader 最终写回前完成可选 bias 和 SiLU；`VKSSBO_MATMUL_SILU_FUSION=0` 安全回退为现有 fused-bias Linear 后接独立 SiLU。MatMul autotune key 已加入 activation 类型并升级 revision，避免融合与非融合错误共用选择。Arc UMA `2x256x1280 -> 10240` 交错 A/B（8 次）：unfused median 112.236 ms / p95 145.050 ms，fused median 86.491 ms / p95 107.703 ms，约 1.298x，max_abs 0.062500。
- Linear+SiLU 增量构建后的两套运行时 DLL SHA-256 均为 `264EB3E5DE700489161E346DA9A2C0C9EC56A4C463E179BF1FC7FFA01D2E13F1`；Linear+SiLU red/green regression、producer metadata、MatMul/Conv2D autotune focused/concurrent、native regression 和 `test_torch.py` 14/14 全部 PASS。
- 新增显式 `torch.ops.vulkan_ssbo.linear_gelu` building block：与 SiLU 共用 MatMul 写回阶段的 activation 枚举和独立 autotune key，GELU 使用与既有 elementwise 路径一致的 tanh 近似；`VKSSBO_MATMUL_GELU_FUSION=0` 回退为 Linear 后接独立 GELU。Arc UMA `2x256x1280 -> 5120` 交错 A/B（8 次）：unfused median 61.838 ms / p95 71.189 ms，fused median 43.222 ms / p95 47.662 ms，约 1.431x，max_abs 0.062363。
- Linear+GELU 增量构建后的两套运行时 DLL SHA-256 均为 `F3F038EB9B20B73406A5AED82D624C762F970C44965B8FDB77974D564C51920E`；Linear+GELU、Linear+SiLU、producer metadata、MatMul/Conv2D autotune focused/concurrent、native regression 和 `test_torch.py` 14/14 全部 PASS。
- 新增显式 `torch.ops.vulkan_ssbo.matmul_softmax` building block：输入为 `[B,M,K]` query 和 `[B,N,K]` key，MatMul 以 `transB=1` 直接读取 key、将 scale 写入 push constant，并在同一 command buffer 中用显式 compute barrier 接 Softmax；`VKSSBO_MATMUL_SOFTMAX_FUSION=0` 或融合 C API 不可用时回退到 MatMul→Scale→Softmax。Arc UMA 回归 shape `[3,17,13] x [3,19,13]` 最大绝对误差 `0.000849`，融合/回退 producer metadata 均正确。
- MatMul+Scale+Softmax Arc UMA A/B（`query[20,1024,64]`、`key[20,1024,64]`、FP32 输入，FP16 SSBO 中间结果，8 次交错）：unfused median `251.355 ms` / p95 `287.167 ms`，fused median `127.743 ms` / p95 `155.505 ms`，约 `1.968x`；融合与回退结果最大差异 `0.000000`。该收益主要来自避免 score 矩阵的 CPU 往返，不代表所有设备/shape 的固定收益。
- MatMul+Scale+Softmax 增量构建后两套运行时 DLL SHA-256 均为 `77C1DEB8035B4BB4E32018C0B17E4317DE1C3DE0EBF7BE6846BC8D14428B24E1`；MatMul+Softmax red/green、Linear SiLU/GELU、producer metadata、MatMul/Conv2D autotune focused/concurrent、native regression 和 `test_torch.py` 14/14 全部 PASS。
- 本次使用的历史 `build_cp313.bat` 会在 DLL 安装后继续执行 wheel 打包；该副产物没有发布，已改名为 `.local-temp` 以避免被识别为可发布 wheel。后续增量验证应直接调用 DLL 构建目标，最终发布阶段再按 3.10～3.15 统一构建。

当前处理：

- MatMul 与 Conv2D 自动调优均已验收，producer metadata、显式 Conv2D+SiLU、Linear+Bias+SiLU、Linear+Bias+GELU 和 MatMul+Scale+Softmax building block 已完成。
- 2026-08-22 已在真实 NVIDIA GeForce RTX 4060 Laptop GPU（driver 610.88、8188 MiB VRAM）完成阶段 D 的功能与同步语义验证：拓扑为 `mixed`，安全 device-local budget 为 6,673,976,524 bytes，独立 transfer queue 和 timeline semaphore 均可用。`VKSSBO_ASYNC_TRANSFER=1` 时原生路径记录 `asyncRequested=1、asyncEligible=1、asyncEnabled=1、resourcesReady=1`，并实际提交 transfer timeline signal 后由 compute submission 等待该 signal；native regression、14-test、离散策略模拟以及三条融合回归全部 PASS。
- 上述验证证明真实离散 GPU 的选择、staging、timeline 提交与回退语义正确；它不包含 upload/kernel/download 分段耗时或队列并行度采样，不能等价为 async overlap 已获得性能收益。下一步是为 PyTorch-integrated SSBO 与 ComfyUI standalone SSBO 分别建立可重复的分段基准，在同步和异步两种模式下记录同一 shape 的 median/p95 和时间线，再决定默认启用范围与后续内核优化优先级。
- 2026-08-22 profiling bridge 已完成：新增 `torch.ops.vulkan_ssbo.profile_reset()` / `profile_stats()`，记录上传 host 转换、submit+wait、下载映射/转换、实际 staging copy wait、字节数和提交次数。最终 DLL 上 native regression、`test_torch.py` 14/14、async capability regression 均通过。
- 2026-08-22 新增 `test/bench_vulkan_ssbo_transfer.py` 与契约测试 `test/test_vulkan_ssbo_transfer_benchmark.py`。每次运行在独立进程中固定 seed、shape、warmup、repeat，输出结构化 JSON；`run_discrete_gpu_test.bat` 已接入 sync/async 两次采样并写入日志。UMA 自检中请求 async 正确回退：`eligible=0、enabled=0`；软件离散模拟中 `eligible=1、enabled=1` 且 staging download wait 可观测。模拟 `[64,128]x[128,64]`、5 次、transfer latency=100us：sync median 32.592 ms，async median 48.029 ms；该结果仅证明测量和门控路径，不代表真实独显 overlap 收益，原因是当前 microbenchmark 每次调用都强制完成上传/计算/下载，尚未形成跨算子流水线。
- 2026-08-22 真实 NVIDIA GeForce RTX 4060 Laptop GPU（driver 610.88，8,188 MiB，Vulkan 1.4.341）完成第一组分段基准。`mixed` 拓扑、device-local safety budget `6,673,976,524` bytes；独立 transfer queue、timeline semaphore 和异步资源均可用。固定 `[64,128] x [128,64]`、FP32、warmup=3、repeats=9：sync median/p95 `0.7289/0.8491 ms`，async median/p95 `0.7336/0.9740 ms`，async 相对 sync median `+0.645%`、p95 `+14.710%`；submit+wait 累计 `0.8898 -> 1.2010 ms`（`+34.97%`），download transfer wait `0.6101 -> 0.5846 ms`。两种模式均上传 `294,912` bytes、下载 `73,728` bytes、9 次 compute submission。native regression、14/14、离散模拟和融合回归全部 PASS，日志最终 `RESULT: PASS`。这说明当前单算子边界下异步没有实测收益，暂不把 `VKSSBO_ASYNC_TRANSFER=1` 设为默认；下一步应测试 resident 多算子流水线和更大传输/计算比，只有重复测量稳定获益才扩大默认启用范围。
- 2026-08-22 新增 standalone resident JSON 基准 `tools/vulkan_ssbo/bench_resident_json.py` 及契约测试。Arc UMA 本地 smoke（长度 `1,048,576`、深度 `8`、warmup=2、repeats=7）结果：eager median/p95 `13.507/22.249 ms`，逐次 resident `9.812/11.861 ms`，单提交 resident graph `3.188/6.838 ms`；最大误差分别 `0.00851/0.01252/0.00335`，三条路径清理后 `resident_entries=0`。相对 eager，逐次 resident 约 `1.38x`，单提交 graph 约 `4.24x`；该数据证明减少 host 往返和合并 dispatch 的收益，但不是独显结果。BAT 已改为在 sync/async 两个独立进程分别记录 resident 基准，后续在 RTX 4060 上复测。
- 2026-08-22 测试顺序调整：当前先完成 UMA 的兼容性与性能测试，离散 GPU 只做软件策略模拟兼容性；真实离散 GPU 性能不纳入当前入口，后续再集中一次性测试。新增 `run_uma_test.bat`，固定选择 `Intel(R) Arc(TM) Graphics`、关闭离散模拟并运行 UMA 回归与性能基准；新增 `run_discrete_simulation_test.bat`，以 UMA 设备为执行宿主，强制离散拓扑、staging、预算和 timeline 分支，只运行兼容性回归。旧的 `run_discrete_gpu_test.bat` 已加显式 `VKSSBO_ENABLE_REAL_DGPU_TEST=1` 门槛，避免当前阶段误跑真实独显性能。
- 2026-08-22 阶段 D 收尾验收：`uma_test.log` 和 `discrete_simulation_test.log` 均为 `RESULT: PASS`。UMA 入口完成设备选择、async 能力回退、fence、native SSBO、14/14 torch、内存预算、transient pool、resident、MatMul/Conv2D autotune、producer metadata、融合回归及 resident graph/算子基准；离散模拟入口完成离散策略、staging、预算、timeline、resident、14/14 torch 和融合回归。真实独显性能入口默认跳过，未构建 wheel。
- 2026-08-22 阶段 F 第一项 FP16 验收：新增 `tools/vulkan_ssbo/test_fp16_resident_runtime.py`，覆盖 eager FP16 输出、FP16 resident activation/weight、resident MatMul、四层 resident graph 和清理。Intel Arc UMA 与离散策略模拟均通过：eager/resident/matmul/graph 最大误差分别 `0.001720/0.002247/0.004684/0.000867`，清理后 `resident_entries=0`；`VKSSBO_ACT_FP16=1` 的 `test_torch.py` 为 `14 OK, 0 FAIL`。UMA cached-operator FP16 代表性 median：`linear(2,256,1280,10240)=16.22 ms`、`geglu(...,5120)=15.75 ms`、`linear(...,1280)=3.30 ms`、`linear(2,1024,640,640)=2.79 ms`；对应同轮 FP32 对照为 `18.65/18.25/3.44/3.11 ms`。这些是当前 Arc 的诊断数据，不据此固定通用 kernel 选择；BF16 native storage 保持关闭。
- 2026-08-22 FP16 边界与 BF16 探测：新增 `test_fp16_boundaries_runtime.py`，非连续 view、源 Tensor 原地修改后的 resident 生命周期、FP64 输入 fallback 均在 UMA 与离散模拟通过，最大误差 `0.000969/0.000000/0.002882`，resident 清理为 0。新增 `test_bf16_fallback_runtime.py`；BF16 elementwise/matmul 均明确回退到 FP32，UMA 与离散模拟最大误差 `2.38419e-07/0`。当前设备/runtime 只启用 FP16 shader storage，BF16 native storage 保持关闭。
- 2026-08-22 INT8 weight-only 首版格式契约已锁定但未启用 kernel：新增 `tools/vulkan_ssbo/int8_weight_only.py` 与 `test_int8_weight_only_contract_runtime.py`，格式为连续 `int8[N,K]` 权重、按输出通道的正 `float32[N]` 对称 scale、zero-point 省略、FP32 反量化/累加；非连续权重、非法 scale、zero-point 或版本字段均回退到现有 FP32 权重路径。UMA 与离散策略模拟入口均接入该回归；当前不声明 Vulkan INT8 storage、INT8 shader 或性能收益，后续再根据设备能力和误差门槛决定是否实现。固定种子多 shape 量化参考（`128x256`、`1024x1280`、`4096x4096`）最大输出相对 L2 `0.008655`，候选误差门槛暂定 `< 0.02`；该数据仅用于未来 kernel gate，不代表 INT8 dispatch 性能。
- 2026-08-22 INT8 能力探测完成：`device_info()` 新增 `shader_int8_supported` 与 `int8_weight_only_enabled` 字段，物理 feature 查询兼容 Vulkan 1.2 核心和 `VK_KHR_shader_float16_int8` 扩展；新增 `test_int8_capability_runtime.py` 并接入 UMA、离散模拟和受保护的真实 dGPU 入口。当前 Intel Arc UMA 实测 `shaderInt8=1`、`int8_weight_only_enabled=0`；能力字段只用于诊断，未启用 INT8 feature、storage 或 kernel，仍强制 FP32 fallback。增量 `torch_cpu.dll` SHA-256 为 `7CC50EBA031847C8C82C04B9C247CBEF0F06003E9C906E0AA34F0D48402CDD58`；UMA 与离散模拟完整入口、native SSBO、`test_torch.py` 14/14、INT8 capability/format regressions 均 PASS。
- 2026-08-22 INT8 storage 边界确认：`device_info()` 新增 `storage_buffer_8bit_supported`、`uniform_and_storage_buffer_8bit_supported`、`storage_push_constant_8bit_supported` 三项只读字段，查询兼容 Vulkan 1.2 核心和 `VK_KHR_8bit_storage`，不把 feature 加入逻辑设备启用链。Intel Arc UMA 实测三项均为 `1`；系统 `vulkaninfo` 中 RTX 4060 的三项也均为 `true`。新增 `test_int8_storage_probe_runtime.py` 与 `shaders/int8_storage_probe.comp`，用仓库内 `glslangValidator` 针对 Vulkan 1.2 编译最小 `int8_t` SSBO，探针 PASS；它只验证编译能力，不创建 pipeline、不声明 INT8 性能收益。UMA 与离散策略模拟完整入口均 PASS，INT8 weight-only 仍固定 FP32 fallback。
- 2026-08-22 INT4 仅权重量化格式契约完成：新增 `tools/vulkan_ssbo/int4_weight_only.py` 与 `test_int4_weight_only_contract_runtime.py`。v1 使用 32 元素 block、每字节低/高 nibble 打包、FP32 scale、INT16 zero-point；合法 payload 只在 CPU 解码为连续 FP32 权重，非法元数据、非连续 tensor、scale/zero-point 越界和 shape 不匹配均返回显式 FP32 fallback。固定代表性 Linear shape 的最大相对 L2 为 `0.083642`（门槛 `< 0.10`），SSBO bridge 最大误差 `0.002267`，清理后 `resident_entries=0`。UMA 真实兼容性入口与离散策略模拟入口均通过，后者验证 `policy=discrete`、强制 staging 和 timeline 模拟不改变契约结果；INT4 Vulkan storage、shader、pipeline、resident 和性能路径仍保持关闭。
- 2026-08-22 FP8 仅权重量化格式契约完成：新增 `tools/vulkan_ssbo/fp8_weight_only.py` 与 `test_fp8_weight_only_contract_runtime.py`，同时支持 `E4M3FN` 和 `E5M2`。v1 使用每 32 个权重共享一个 FP32 `max_abs/max_finite` scale，`qweight` 为每权重一个 `uint8`；仅接受 finite 权重和 finite payload，E4M3FN `0x7f/0xff`、E5M2 Inf/NaN code 及非法元数据均显式回退到连续 FP32。固定代表性 Linear shape 的最大相对 L2 为 E4M3FN `0.024333`（门槛 `< 0.08`）、E5M2 `0.048400`（门槛 `< 0.15`），SSBO bridge 最大绝对误差分别为 `0.001107/0.001162`，清理后 `resident_entries=0`。UMA 真实兼容性入口与离散策略模拟入口均接入并通过；FP8 Vulkan feature、storage、shader、pipeline、resident 和性能路径仍保持关闭。
- 2026-08-22 FP4 仅权重量化格式契约完成：新增 `tools/vulkan_ssbo/fp4_weight_only.py` 与 `test_fp4_weight_only_contract_runtime.py`，采用有限值 E2M1 codebook（正值 `0、0.5、1、1.5、2、3、4、6`），每 32 个权重共享 FP32 `max_abs/6` scale，两个 4 bit code 按低 nibble 偶数列/高 nibble 奇数列打包；奇数列未使用的高 nibble 非零时显式回退。固定代表性 Linear shape 最大相对 L2 为 `0.113173`（门槛 `< 0.12`），SSBO bridge 最大绝对误差为 `0.002713`，非法 metadata、scale、padding 和非有限源权重均回退 FP32，清理后 `resident_entries=0`。UMA 真实兼容性入口与离散策略模拟入口均接入；FP4 Vulkan feature、storage、shader、pipeline、resident 和性能路径仍保持关闭。
- 2026-08-22 UMA benchmark 退出路径修复：`bench_cached.py` 的算子/FP16 benchmark 与 `bench_fixed.py` 原本能完整输出数据，但在 Python/Vulkan native 清理阶段以 `0xC0000409` 退出；参照同目录 runtime benchmark 的既有模式，新增输出 flush 后的受控进程结束。新增 `test_bench_cached_exit_runtime.py`、`test_bench_fixed_exit_runtime.py` 先复现失败再验证修复，当前两类 benchmark 完整输出且 `rc=0`；不改变测量逻辑或性能结论。
- resident mirror cache 和代表性算子链继续作为回归基础；每个算子仍会为 stock Vulkan Tensor 生成 CPU mirror，后续继续评估 producer metadata、融合算子和独立 TensorImpl storage/`copy_` 协议。stock Vulkan runtime 与 SSBO runtime 使用不同 Vulkan device，不能把自研 `VkBuffer` 伪装成 stock Vulkan storage。

已知风险：

- 当前 stock Vulkan runtime 与 SSBO runtime 使用独立 Vulkan device，不能假设两个 runtime 可以直接共享 `VkBuffer`。
- host lease 是带弱引用和版本检查的过渡型 mirror 方案，代表性连续算子链已通过 A/B；它仍不等价于真正的 resident Tensor，view、in-place 和 autograd 扩展前必须继续保留 materialize fallback。
- producer metadata 同样是过渡层：普通 Tensor 使用 `_version()` 检查 in-place 失效；inference Tensor 没有版本计数，只能记录为 advisory metadata，任何融合实现必须在无法证明无别名/无 in-place 时回退。
- Intel 驱动需要显式 dispatch barrier；不能用无同步的多 dispatch 取代现有同步。
- fp16 存储会产生可见舍入误差，必须持续报告 `max_abs`、均值误差和图像 NCC。

## 2. 不变的工程约束

- 每次改动前在 `D:\PyTorch for Vulkan\backups\` 创建带时间戳的局部备份。
- 不删除或回退用户已有 dirty changes；不修改模型文件；不向仓库外写入大文件。
- GPU buffer、权重缓存和临时激活峰值不得超过 Vulkan budget 的安全上限；默认保留至少 20% 安全余量。
- Rust 只负责策略计算和预算，不拥有或销毁 Vulkan handle；Vulkan handle 生命周期由 C++ runtime 管理。
- 所有新路径必须有环境开关或能力检测，并保留可靠 fallback。
- MSVC DLL 构建完成后，必须同步到源码运行时实际加载的 `torch/lib/torch_cpu.dll`，再运行测试。
- 不以单次偶然 benchmark 宣称收益；每项优化至少提供 warmup、median/p95、输入 shape、dtype、kernel variant 和对照组。
- 本任务书完成前不构建 wheel；wheel 构建是最后独立阶段。
- dGPU 性能结论必须来自真实离散设备上的重复运行；软件离散模拟只能验证 staging、预算、timeline 和回退语义，不能用于宣称超过 XPU 或启用默认 async。

## 3. 分阶段执行顺序

### 阶段 A：基线、备份和硬件拓扑

目标：建立可重复的硬件和性能基线。

工作项：

- 记录 `VkPhysicalDeviceMemoryProperties`、`VK_EXT_memory_budget`、vendor/device/driver、queue family 和支持的扩展。
- 实现 `UMA / DISCRETE / MIXED / UNKNOWN` 分类，但分类必须允许基准校正。
- 分别记录 device-local、host-visible、host-cached heap 的 budget/usage。
- 为上传、下载、映射和小/中/大 buffer 建立基准。
- 固化 baseline：`test_torch.py`、native regression、Conv2D/MatMul/VAE benchmark 和 SDXL 端到端。

验收：分类信息可从 `torch.ops.vulkan_ssbo.device_info()` 或等价诊断接口读取；基线结果可复现；不改变默认算子结果。

### 阶段 B：架构感知内存策略（已完成）

目标：让分配策略真正区分 UMA 和离散 GPU。

UMA：

- 优先使用 `HOST_VISIBLE` 且适合 device access 的持久映射 buffer。
- 减少 map/unmap 和 CPU staging copy。
- 激活池、权重池和 host mirror 共享同一预算模型，但保留带宽余量。

离散/混合：

- 权重和热激活优先 device-local。
- 使用持久 staging ring，支持批量上传。
- 只在有独立队列、timeline semaphore 和实际重叠收益时启用异步 transfer。
- VRAM 不足时按图级生命周期和 LRU 驱逐到系统内存；不得每个算子来回交换。

验收：两类路径均能运行 14/14；内存统计包含 topology、heap budget、resident、transient、staging 和 weight cache；OOM 时有可预测回退。

### 阶段 C：SSBO resident Tensor（mirror 验收完成，原生 Tensor storage 仍进行中）

目标：连续受支持算子不再经过 CPU。

resident 对象至少保存：

```text
VkBuffer / allocation
offset / bytes
shape / stride / dtype / layout
last_access / producer fence
```

实现要求：

- SSBO -> SSBO 算子直接绑定输入 buffer。
- 只有 `.cpu()`、stock Vulkan fallback、未支持算子或显式导出时才 materialize。
- Tensor 生命周期结束时安全归还 transient pool 或 resident cache。
- 版本计数、in-place 修改、view、contiguous 和 autograd 语义必须明确。
- stock Vulkan 与 SSBO device 不可共享 handle 时，使用明确的 bridge/materialize 边界，不伪装成 zero-copy。

验收：连续 `conv -> silu -> conv`、`linear -> bias -> silu` 和 `matmul -> softmax` 链有 A/B 结果一致性与往返次数统计；错误路径自动回退。当前 Arc UMA 结果：resident 关闭命中 0、开启命中 4，三条链 A/B 最大误差 0；原生回归与 14/14 均通过。由于 stock Vulkan 与 SSBO 使用不同 VkDevice，独立 TensorImpl storage/`copy_` materialize 协议仍作为后续工作，不宣称已实现 zero-copy。

### 阶段 D：异步执行和命令批处理

目标：在可重叠的上传、计算和下载之间建立流水线。

- 使用 command buffer 录制、timeline semaphore 和显式 buffer barrier。
- 移除不必要的每算子 `vkQueueWaitIdle`，但保留 Intel 驱动需要的 RAW/WAR/WAW 依赖。
- 先在离散 GPU 验证 transfer/compute overlap，再决定 UMA 是否启用同一机制。
- 失败或不支持 timeline semaphore 时回退同步路径。

当前进展（2026-08-21）：已完成第一步“能力探测与安全门控”，但尚未宣称异步执行已实现。

- `device_info()` 现在公开 compute queue、transfer queue、queue flags、是否独立 transfer queue、timeline semaphore 支持，以及 `async_transfer_requested/eligible/enabled` 三个状态字段。
- `VKSSBO_ASYNC_TRANSFER=1` 只记录显式请求；只有 DISCRETE/MIXED 拓扑、独立 transfer queue 和 timeline semaphore 同时满足时才报告 `eligible=1`。
- `async_enabled` 保持为 0，直到真正的 transfer/compute timeline 提交路径完成；因此当前默认同步 `vkQueueSubmit + vkQueueWaitIdle` 行为没有变化。
- 当前 Arc UMA 实测：compute queue family=0、transfer-only family=2、timeline semaphore=1，但拓扑为 UMA，故 `eligible=0、enabled=0`；不能用该设备推断离散 GPU 的 overlap 收益。
- 新增回归 `tools/vulkan_ssbo/test_async_transfer_capabilities_runtime.py`，已验证请求开关、字段完整性、UMA 禁止异步和未实现路径不误报。
- 提交等待已改为每次 `vkQueueSubmit` 使用独立 `VkFence` 并调用 `vkWaitForFences`；不再调用全局 `vkQueueWaitIdle`。这不改变当前同步 API 语义，但避免无关队列工作被全局 idle 阻塞，并为后续 timeline 提交保留边界。
- 新增回归 `tools/vulkan_ssbo/test_fence_submit_runtime.py`；在 Arc UMA 上确认 `mode=fence、wait_idle=0`，并通过 native regression 与 `test_torch.py` 14/14。
- 已接入受门控的 timeline 资源骨架：满足条件且显式请求时，逻辑设备会同时创建独立 transfer queue、transfer command pool、timeline semaphore，并使用 concurrent buffer sharing；上传提交 signal timeline，compute submission wait timeline。若设备创建/资源创建失败，自动回退同步 fence 路径。
- 当前 Arc UMA 实测 `resourcesReady=0、asyncEnabled=0`，因此没有改变现有执行路径；本机无法验证离散 GPU 的真实 overlap 收益，不能将这一步计为性能提升。

验收：给出 overlap 前后传输/计算时间线和端到端数据；不得出现 DEVICE_LOST、未初始化读或跨线程竞态。

### 阶段 E：融合、自动调优和图级内存规划

目标：减少中间写回和 kernel 选择成本。

优先融合：

- Linear + Bias + SiLU/GELU。
- Conv + Bias + SiLU。
- Conv + GroupNorm。
- MatMul + Scale + Softmax。

自动调优键必须包括：

```text
vendor_id, device_id, driver_version, memory_topology,
dtype, shape, groups, layout, kernel_variant
```

调优策略：常见 shape 首次有限 A/B，结果持久化；异常值、超时或错误自动降级；不能每次启动重复长时间 autotune。

验收：融合与非融合结果误差在算子门槛内；常见 SDXL shape 的 median/p95 有量化收益；缓存命中后无明显调优开销。

### 阶段 F：低精度和新算子能力

顺序固定为：

1. FP16 resident activation/weight。
2. BF16（硬件或 shader 支持时）。
3. INT8 weight-only 或混合精度路径。
4. weight-only INT4，按 block 保存 scale/zero point。
5. FP8，明确 E4M3/E5M2 和 scale 规则。
6. FP4，仅在有可靠软件解码和误差预算时启用。

同步完善：

- Conv1D/Conv3D 的 stride、dilation、groups、padding 和训练反向所需接口。
- LoRA 低秩增量权重和参数更新的 resident/上传策略。
- 音频模型常见线性、归一化、1D 卷积组合。
- 视频模型常见 3D 卷积和大体积激活的分块策略。

验收：每种 dtype 有独立误差门槛、fallback 和 benchmark；训练路径不静默丢失梯度；14/14 和扩展回归持续通过。

### 阶段 G：全链路验证和发布准备

- 重跑 SDXL 512x512、8 步 Euler、seed=42；报告每步、总耗时、上传/下载/算子占比。
- 图像与 `output/vulkan_test_00005_.png` 比较 mean/std/NCC，目标 NCC >= 0.85。
- 在 UMA 和离散 GPU 各至少完成一次 smoke test；没有对应硬件时使用能力模拟和明确标注。
- 更新源码说明、中文任务记录和变更日志。
- 只有所有阶段验收通过后，才开始 Python 3.10～3.15 wheel 构建、smoke test 和 Releases 上传。

## 4. 性能目标和报告格式

每项优化必须记录：

```text
commit/working-tree snapshot
device + driver + memory topology
shape / dtype / layout / groups
baseline median/p95
optimized median/p95
upload / kernel / download / Python breakdown
GFLOPS or effective bandwidth
max_abs / mean_abs / NCC
fallback condition
```

阶段目标：

- 先消除不必要的 CPU↔GPU 往返，再优化 kernel。
- SDXL 端到端以当前约 120 秒估计值为基线，最终目标 <= 60 秒；如果硬件带宽限制无法达到，必须给出瓶颈证据和下一步方案。
- VAE 512x512 大卷积、SDXL Conv2D、MatMul、GroupNorm、Silu 是优先测量对象。

## 5. 当前执行指令

当前执行 **阶段 F（低精度和新算子能力）**，从 FP16 resident activation/weight 开始：

1. 保留现有改动并创建局部备份。
2. 保持 resident mirror cache 的 LRU、版本检查和 fallback；不把 SSBO `VkBuffer` 伪装成 stock Vulkan storage。
3. 已完成三条代表性链路的 resident 命中和 A/B 正确性覆盖，持续作为回归测试运行。
4. MatMul 的设备/驱动/拓扑/shape keyed autotune 已完成；保持 3 轮交错 median、3% 晋升门槛、内存预算门控和原子持久化缓存作为后续算子的公共设计基线。
5. MatMul 与 Conv2D 的设备感知自动调优已完成；后续自动调优必须复用独立状态、文件锁、原子持久化、3% 晋升门槛和 budget 门控，新增候选前先做多设备能力边界与数值 A/B。
6. 每次增量构建 `torch_cpu.dll` 后同步两个实际测试运行时；持续运行 native regression、`test_torch.py` 14/14、resident 链和各 autotune focused/concurrent 回归。
7. 已完成 MatMul+Scale+Softmax 显式融合及 A/B 验收；阶段 D 的能力探测、安全门控、fence、timeline 骨架以及 UMA/离散模拟入口验收完成。阶段 F 当前推进 FP16 resident activation/weight、BF16 FP32 fallback 和 INT8 格式契约；保持显式误差门槛、fallback 和独立基准，不实现 SSD backing、不实现投机采样、不构建 wheel。
8. 真实独显性能暂缓。后续一次性实测时再显式设置 `VKSSBO_ENABLE_REAL_DGPU_TEST=1` 运行 `run_discrete_gpu_test.bat`；软件模拟结果只用于兼容性，不作为 dGPU 性能结论。
9. FP16 的 view、in-place、fallback 和 resident 生命周期边界已通过；BF16 能力探测已完成并确认走 FP32 fallback，native BF16 storage 保持关闭。INT8 weight-only 首版数据格式、scale/zero-point 语义、安全 fallback 契约、`shaderInt8` 和 8-bit storage 能力探测已通过回归测试锁定；最小 `int8_t` SSBO shader 只完成编译探针，INT8 feature、storage、pipeline 和 kernel 仍未启用，默认仍走 FP32 fallback。INT4 v1 格式契约和 FP32 fallback 已通过 UMA/离散模拟回归，但 INT4 Vulkan feature、storage、shader、pipeline、resident 和性能路径仍未启用。FP8 E4M3FN/E5M2 v1 格式契约和 FP32 fallback 已通过 UMA/离散模拟回归。FP4 E2M1 v1 格式契约和 FP32 fallback 已通过 UMA/离散模拟回归；FP8/FP4 的 Vulkan feature、storage、shader、pipeline、resident 和性能路径仍未启用。

完成当前阶段验收项后，更新本文件的“当前状态”和数据，再开始下一阶段。

## 6. 明确不直接采用的方案

- 不把 UMA 等同于普通 CPU 指针可被 shader 直接访问。
- 不假设 `DEVICE_LOCAL` 一定 `HOST_VISIBLE`。
- 不假设独显只有两个简单内存池。
- 不因为支持双缓冲就宣称有异步收益。
- 不把 SSD 当作默认显存扩展。
- 不把 FP4、2-bit 或 cooperative matrix 当作所有 Vulkan GPU 都具备的硬件能力。
- 不把投机采样作为 SDXL 优化；它只在适合的自回归音频/文本模型阶段单独评估。
