# PyTorch Vulkan SSBO 后端开发任务书

版本：2026-10-06
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

最新实验（2026-10-05）：按用户授权完成最终GN→SiLU候选的小基准与固定512×512猫图
对照。候选SHA-256仍为 `3197F3B9E1DE77AA552A9AC9E540FED4B4172226AC46E3FDAD77E7FC9D42E990`。
先按off/on/on/off四个独立服务运行，再补一个匹配工作量off：off1=114.074秒，但第二采样步
shader调用翻倍、低空闲内存，符合CFG拆批，故从速度汇总剔除。匹配off2/off3=
`120.930/124.463秒`，on1/on2=`103.030/106.210秒`，中位 `122.6965→104.620秒`
（本轮下降14.73%，两样本/模式，不作通用或统计显著性承诺）。五图像素均与稳定猫图
完全相同。matched OFF→ON恰好少272次提交/dispatch/fence、0.957GiB上传与1.914GiB下载；
其他shader频次相同。原生日志确认每图272次GN融合；不能把off1额外采样调度算作融合收益。
最终候选四shape交错15次小基准、120组边界、缺shader恢复、14项算子和30项路由检查通过。
报告 `D:\ComfyUI Vulkan beat\docs\UMA_GROUPNORM_SILU_EXPERIMENT_20261005_ZH.md`，
完整证据 `D:\ComfyUI Vulkan beat\tools\vulkan_ssbo\uma_gn_20261005\comparison.json`。
本轮为实验，未替换正式两处 `473C5BF5...` DLL，GN融合默认仍关闭；所有自建测试服务
已停止，未停止用户程序、未构建wheel、未跑RTX/1080P＋4×、未测完整峰值内存。
全量UMA门禁已在下段补验；下一发布步骤为默认启用策略复核。下方9月25日为实现阶段历史记录。

2026-10-06 候选门禁补充：使用环境变量 `VKSSBO_DLL_NAME=vulkan_ssbo_gn_candidate.dll`、
`VKSSBO_GN_SILU=1`、`OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1` 运行 `run_uma_test.bat`，
得到85个 `[run]`、87条 `[exit]`、非零0、显式跳过0、唯一 `RESULT: PASS`；日志复制到
`D:\ComfyUI Vulkan beat\tools\vulkan_ssbo\uma_gn_gate_20261006\uma_test.log`。
这是候选standalone DLL与未替换的integrated Torch混合回归：`test_torch.py`等standalone
脚本使用候选，device/async/native与部分scheduler/bridge调用集成 `torch.ops`。
`bench_resident_json.py` 的 build 元数据固定读默认 `vulkan_ssbo.dll`，不能用其build字段
证明候选身份；候选SHA-256为 `3197F3B9E1DE77AA552A9AC9E540FED4B4172226AC46E3FDAD77E7FC9D42E990`，正式DLL和
集成torch_cpu.dll未替换。GN专项证据仍见 `D:\ComfyUI Vulkan beat\tools\vulkan_ssbo\uma_gn_20261005`。

最新进展（2026-09-25）：完成稳定FP32主机接口的 GroupNorm→SiLU 单内核候选，
保留原CPU中间上传的NaN/Inf/饱和/RNE half舍入点，最终FP32输出，一次dispatch和提交。
三/双dispatch方案因大形状退化被弃用。四组单内核交错小基准逐位一致，代表形状
`[2,320,64,64]` median `9.62→3.81ms`、`[1,128,256,512]` `51.48→32.10ms`，
但此后修复了旧GroupNorm无缓存参数提前释放问题，最终DLL须重新性能复验；这些受
系统并发/内存波动影响的局部数据不能外推整图收益。最终候选默认/无缓存各120组、
非法参数、缺shader恢复、30项路由契约、14项算子及相关resident/FP16/hostread回归通过。
最终候选SHA-256 `3197F3B9E1DE77AA552A9AC9E540FED4B4172226AC46E3FDAD77E7FC9D42E990`，
文件 `tools/vulkan_ssbo/vulkan_ssbo_gn_candidate.dll`。正式两处DLL仍为 `473C5BF5...`，
ComfyUI新路径 `VKSSBO_GN_SILU=1` 才请求启用，默认关闭；尚未做整图画质/速度验收。
本机另一个项目正在XPU实际推理且可用RAM曾降至约3GB，未停止用户任务，未再并发启动
完整猫图。后续先在资源空闲时用最终候选重跑固定猫图与小基准，再决定是否正式部署。
详细记录 `D:\ComfyUI Vulkan beat\docs\UMA_GROUPNORM_SILU_20260925_ZH.md`，
证据 `D:\ComfyUI Vulkan beat\tools\vulkan_ssbo\uma_gn_20260925`。未构建wheel、
未跑RTX、未更换正式DLL/集成torch_cpu、未刷新独显包，未启动Vulkan测试服务。

当前工作重点（2026-09-19）：按用户要求优先继续 Intel Arc UMA 性能优化，RTX 验收暂缓。
基础门禁通过不等于完整性能优化完成；1080P＋4×工作流与 XPU 对齐仍待验收。
最新增量：完成稳定FP32主机接口的 GEGLU→Linear 单提交路径，保留GPU上的中间清洗与
FP16舍入点，最终输出FP32。奇数维/无bias/FP32与FP16权重/更新/缓存关闭/缺shader恢复等
边界均通过；18项路由契约、14项算子及相关resident/fusion/cache回归通过。固定512猫图
每模式两次，关闭 `116.537/139.709秒`、开启 `117.997/123.358秒`，中位下降约5.81%，但
首轮略慢、样本少，因此ComfyUI默认关闭，仅启动前 `VKSSBO_FF_GRAPH=1` 选择启用。
四张图像素与上一轮完全一致；每图提交 `6393→5833`，逻辑上传少3.125GiB、下载少6.25GiB，
GPU转换dispatch增加560次。新路径预算不足/旧DLL/动态权重/梯度/hooks等保留回退。
最新standalone DLL两副本SHA-256为 `473C5BF56E4935F2C9B887CE056345A2E3AF8CFF1A6E036C2534767DBEA6A813`。
记录 `D:\ComfyUI Vulkan beat\docs\UMA_FEED_FORWARD_20260919_ZH.md`；证据
`D:\ComfyUI Vulkan beat\tools\vulkan_ssbo\uma_ff_20260919`。未构建wheel、未测RTX、未刷新
独显包、未重跑全量UMA门禁；测试服务已停止。以下为本日较早验证记录。

同日后续整图验证已闭环：CPU 多线程原地加/除错误造成纯黑，三个常用 Vulkan 启动入口
增加 OMP/MKL 单线程默认值（仅规避，未修复自建 Torch 混合 OpenMP 运行时）；CLIP 长链
偶发大值导致色块，通过在 standalone submit 尾部补齐 shader-write→host-read 屏障解决。
同一候选 DLL 的屏障关闭/开启红绿回归分别失败/通过，开启后三次 CLIP 对照稳定。
屏障不增加 CPU 等待，双图提交前 host wait 仍为0；14项算子、100轮读取及FP16边界通过。
修复后的同一 DLL、CPU一线程，仅切换激活SIMD开关：固定512×512/8步Euler/seed42猫图
标量两次 `232.645/258.212秒`，SIMD两次 `110.314/121.599秒`，中位 `245.429→115.957秒`
（本轮约2.12倍、耗时下降52.75%；每模式仅两样本，不是通用稳定承诺）。四张图像素哈希
一致，历史正常猫图NCC `0.977245`，人工确认窗台猫正常。每轮含checkpoint加载，不称
模型常驻热启动。逻辑上传/下载仍为约15.4/28.6GiB，提交6393次，后续继续减少这些边界。
最终 standalone DLL 已同步 tools/ComfyUI 副本，SHA-256
`4F0D46414AC2B467D910FB4113D90BED72B933779B7B4626FEB0DADCD305EEFA`。
完整记录 `D:\ComfyUI Vulkan beat\docs\UMA_CAT_VALIDATION_20260919_ZH.md`，
证据目录 `D:\ComfyUI Vulkan beat\tools\vulkan_ssbo\uma_cat_20260919`。
所有本轮测试服务已关闭；未构建wheel、未测RTX/1080P＋4×、未测完整内存峰值、未重跑
全量UMA门禁；以下首轮转换优化记录中的DLL哈希和未出图状态仅为当时快照。

本轮完成 standalone FP32 激活上传的 SIMD 清洗/转换优化，正常 8 元素块保留相同
FP16 round-to-nearest-even 位结果，异常块/尾部保留原标量保护；启动前设置
`VKSSBO_ACTIVATION_SIMD=0` 可回退，不改变 FP16 直接上传或 shader 精度。
4 个独立进程按标量/SIMD/SIMD/标量执行，每模式 30 个样本：LayerNorm median
`7.029→1.191 ms`，GroupNorm `14.154→1.821 ms`，大尺寸 GroupNorm `91.224→9.287 ms`；
上传阶段分别 `5.870→0.202`、`11.901→0.429`、`81.097→2.741 ms`。输入/输出哈希及
提交/传输计数完全一致。Conv2D median `32.608→14.817 ms`，但 GPU timestamp 也有变化，
不能把全部差值归因于 CPU 转换，更不能外推整图倍数。
每模式 2,106,092 个 CPU 位比较及清洗诊断通过；14/14、FP32/FP16 融合、FP16 边界和
100 轮 resident host-read 均通过。新版 standalone DLL 已同步 tools 和 ComfyUI 副本，
SHA-256 `BA8C317F0F9D764A83C0CBE8CC9BB166F753604E54039DC04DF5CE15BCBEF910`；
没有修改集成 `torch_cpu.dll`、构建 wheel、更新独显包或执行 RTX 测试。本轮未重跑完整
UMA 门禁或完整出图，不把历史门禁 PASS 当作新版整流程验收。
数据和具体边界见 `D:\ComfyUI Vulkan beat\docs\UMA_ACTIVATION_UPLOAD_20260919_ZH.md`
及 `tools/vulkan_ssbo/uma_activation_20260919/ab.json`。

已完成并保留：

- SSBO 基础设备、权重 LRU、瞬时 buffer pool 和 Rust 内存策略层。
- 5D+ buffer 布局及 CPU/GPU roundtrip。
- Conv1D grouped/stride/dilation、Conv2D grouped/stride/dilation、Conv3D。
- Conv1D resident graph（连续 FP16 推理态，dense/grouped/stride，ComfyUI 路由）。
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
- 2026-08-23 阶段 D 的低风险资源优化：离散/混合路径新增独立 host-visible staging buffer pool，默认上限 `256 MiB`、最小复用尺寸 `64 KiB`，与 device-local transient pool 分离；`VKSSBO_STAGING_POOL=0` 可关闭，`VKSSBO_STAGING_POOL_MB`/`VKSSBO_STAGING_POOL_MIN_KB` 可调。池只在对应提交完成后归还，`empty_cache()` 会同时清理，`memory_stats()` 暴露 `staging_pool_bytes/entries/hits/misses`。新增 `test_staging_pool_runtime.py`，离散策略模拟连续大矩阵上传/下载命中复用（本机 `hits=4`），并与 threshold、fence、native SSBO、`test_torch.py` 14/14 一起通过。该优化只减少 allocation/free 开销，不宣称实现跨算子 overlap。
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
- 2026-08-22 独显一次性性能整合包已完成本机离散策略模拟验收：`E:\PyTorch for Vulkan SSBO 离散型GPU一次性性能测试整合包\local_discrete_simulation_check.log` 最终 `RESULT: PASS`。包内已包含同步/异步 transfer、native SSBO、14/14 Torch、resident/fusion benchmark，以及 INT8/INT4/FP8/FP4 权重量化契约；模拟只验证包布局、中文路径加载、staging、预算和 timeline/回退语义，不作为真实独显性能结论。修复了中文目录传递给 DLL 窄字符 shader API 的路径编码问题，并让模拟 runner 的初始 async 状态默认为关闭。
- 2026-08-23 完整独显测试环境包已制作并在包内 runtime 验收：`runtime\` 携带 Windows x64 Python 3.13、PyTorch Vulkan `2.13.0a0+gitcf30153`、NumPy 及依赖，未压缩约 `719.54 MiB`；`preflight_environment.bat` 与使用包内 `runtime\python.exe` 的完整离散策略模拟均通过，最终 `RESULT: PASS`。目标机仍需自行安装 Vulkan runtime/NVIDIA 驱动；真实 dGPU 性能尚未在本机宣称。
- 2026-08-23 standalone SSBO 路径同步新增独立 host-visible staging buffer pool：默认上限 `256 MiB`、最小复用尺寸 `64 KiB`，由 `VKSSBO_STAGING_POOL`、`VKSSBO_STAGING_POOL_MB`、`VKSSBO_STAGING_POOL_MIN_KB` 控制；`memory_stats()` 新增 `staging_pool_bytes/entries/hits/misses`，`vkssbo_cache_clear()` 同时清理该 pool。新增 `test_staging_pool_standalone_runtime.py`，通过真实 host-visible allocation 生命周期和提交后归还验证 `hits=1、entries=1、bytes=1048576`，清理后归零；现有 transient pool、resident smoke、resident benchmark contract 和 native DLL 构建均通过。整合包本机离散策略模拟最终 `RESULT: PASS`，新增 standalone 回归 `rc=0`；更新后的静态清单含 `26995` 个文件，二次核对 `missing=0、mismatch=0、unlisted=0`，包内环境预检 `PASS`。该优化只减少 standalone staging allocation/free 开销，不宣称跨算子 overlap。
- 2026-08-25 SDXL 猫咪端到端正确性补强：发现跨 KSampler 任务复用 GPU 权重缓存和 Python 逻辑权重身份表会导致后续任务的 UNet latent 出现全 NaN，最终 PNG 变成全黑或色块。新增 `vkssbo.clear_weight_cache()`，每次 KSampler 开始前清空原生权重缓存、Python `_stable` 身份表和旧缓存表；`test_weight_cache_identity_runtime.py` 增加清理后归零并重新上传校验，回归通过。固定 `42 -> 44 -> 42` 连续 8 步猫咪序列全部有限值、无黑图，最终 PNG 可辨认猫咪；1 步诊断图仅作为中间诊断，不作为画质验收。正式入口关闭临时边界诊断日志，仅保留缓存清理修复。
- 2026-08-25 阶段 G 全量复验：使用包含最新 `torch_cpu.dll` 的 `tools\Python313` 运行时，UMA 全量兼容性/性能入口 `run_uma_test.bat` 最终 `RESULT: PASS`；离散策略模拟入口 `run_discrete_simulation_test.bat` 最终 `RESULT: PASS`。两项均覆盖 native SSBO、14/14 torch、FP16/BF16、INT8/INT4/FP8/FP4 契约、resident、预算、自动调优、融合和相应 UMA/staging/timeline 回退路径。真实独显性能仍暂缓，未据模拟结果宣称性能收益，未构建 wheel。
- 2026-08-27 阶段 D 调度生命周期补强：新增 `torch.ops.vulkan_ssbo.scheduler_stats()`，公开最后提交值、完成值、in-flight/峰值、提交、poll 和 wait 计数；`submit_and_wait()` 改为提交后等待对应完成值，并增加完成批次的 fence 回收与 command buffer 复用。异常路径新增 `ComputeManager::abort()`，清理未提交录制、transfer 和 staging 状态；Conv2D 自动调优异常回退已接入该清理路径。新增 `test_scheduler_lifecycle_torch_runtime.py` 与 standalone `test_vulkan_ssbo_scheduler_runtime.py`，UMA、离散策略模拟和 standalone 均通过，最终 `in_flight=0`，最大误差 `0.000597`。这一步完善了执行器生命周期和可观测性，仍不等价于连续设备 Tensor 或无 host wait 的图执行。
- 2026-08-25 运行时一致性注意：`ComfyUI_windows_portable\python_embeded` 曾加载旧版 `torch_cpu.dll`，导致 INT8 能力字段缺失并使 UMA 入口误报失败；`tools\Python313` 和 `D:\PyTorch for Vulkan\pytorch-src\build\bin` 已验证为新版本。便携 DLL 当时被其他独立 XPU 进程占用，未强制终止该进程；发布前必须在所有使用者退出后同步并重新执行 INT8 capability regression 与两套入口。
- 2026-08-25 独显一次性性能整合包按最新缓存修复更新：同步 `vkssbo.py`、加入 `test_weight_cache_identity_runtime.py` 并接入 runner，更新 `MANIFEST_SHA256.txt`。包内 `preflight_environment.bat` 与 `run_local_discrete_simulation_check.bat` 均通过，日志最终 `RESULT: PASS`；包内模拟仍只验证兼容性，不代表真实独显性能。
- 2026-08-22 UMA benchmark 退出路径修复：`bench_cached.py` 的算子/FP16 benchmark 与 `bench_fixed.py` 原本能完整输出数据，但在 Python/Vulkan native 清理阶段以 `0xC0000409` 退出；参照同目录 runtime benchmark 的既有模式，新增输出 flush 后的受控进程结束。新增 `test_bench_cached_exit_runtime.py`、`test_bench_fixed_exit_runtime.py` 先复现失败再验证修复，当前两类 benchmark 完整输出且 `rc=0`；不改变测量逻辑或性能结论。
- 2026-08-27 阶段 2 修复后复验：`test_producer_metadata_runtime.py` 与 `test_fence_submit_runtime.py` 均通过；完整 `run_uma_test.bat`（33 个主测试）和 `run_discrete_simulation_test.bat`（20 个主测试）均无非零退出，最终均为 `RESULT: PASS`。修复内容为 Conv2D+SiLU 不满足融合能力时在返回空指针前调用 `ComputeManager::abort()`，避免 unfused fallback 继承仍在 recording 的命令缓冲；fence 回归同时兼容旧版 `fence_waits` 和新版 scheduler `value/in_flight` 日志格式。
- 2026-08-27 入口覆盖补全：两个正式 BAT 均加入 standalone `test_scheduler_lifecycle_runtime.py`；复跑后 UMA 共 `34` 项、离散策略模拟共 `21` 项，均为 `RESULT: PASS`，standalone scheduler 项退出码均为 `0`。Vulkan loader 对缺失的可选 SDK layer JSON 打印 `loader_get_json` 提示，但不影响设备初始化、测试结果或退出码。
- 2026-08-27 按算子 profiling 补全：standalone bridge 新增 `bench_operator_profile.py` 与 `test_operator_profile_contract.py`，覆盖 elementwise、linear/matmul、Conv1D/2D/3D、Softmax、LayerNorm/GroupNorm、attention、GEGLU、upsample、cat 的 quick/representative shape；支持环境选择的 FP32/FP16 activation，固定 seed、warmup、repeat，末行输出结构化 JSON，记录 median/p95、估算 FLOPs/GFLOPS、数值误差、submit/wait、dispatch、上传/下载字节和清理状态。standalone C API 新增与集成路径一致的 `vkssbo_profile_reset/profile_stats` 计数器；在支持 timestamp query 的队列上记录非零 GPU command time，不支持时显式累计 timestamp failure，绝不将墙钟时间冒充 kernel 时间。FP32/FP16 quick、representative UMA smoke 与离散策略模拟契约均通过；UMA 入口现将 FP32 与 FP16 representative profile 作为非门控诊断项写入 `uma_test.log`，离散模拟只运行契约不采集性能结论。
- 2026-08-27 ComfyUI profile 路由修复与契约保护：发现 standalone `vkssbo.py` 才是 ComfyUI patch 的实际执行后端，而 `/vkssbo/profile` 原先只读取 `torch.ops.vulkan_ssbo`，会把真实 standalone 统计误报为全零。现已优先读取 standalone 的 profile、memory、scheduler 统计，并在异常时回退到 torch.ops；reset 同时清理两套计数器。新增 `test_profile_route_contract.py`，在不启动 ComfyUI 的情况下验证 standalone 主路径、torch.ops fallback、两套 reset 和端点注册。UMA 与离散策略模拟入口均接入该契约测试。UMA 512x512/8 步猫咪 smoke 已生成可辨认 PNG；一次 profile 记录 compute submit `8424`、dispatch `21226`、上传 `16561170432` bytes、下载 `25986670592` bytes、`queue_wait_idle_count=0`、scheduler `in_flight=0`。
- 2026-08-27 standalone GPU timestamp 补全：standalone `vulkan_ssbo.dll` 新增可选 Vulkan timestamp query pair，按提交记录 GPU command time；若队列不提供有效 timestamp bits，则按 submission 记录 failure 并继续走原有 fence 同步路径。`bench_operator_profile.py` 的 `gpu_timestamp_collection` 改为根据真实采样动态填充，profile 契约增加 submissions/failures 覆盖。Intel Arc UMA quick profile 已实测每次提交 `gpu_timestamp_submissions=1`、GPU 时间非零；native SSBO、14/14 torch、resident benchmark、staging pool 和 profile route 契约均通过。该 GPU 时间只覆盖 Vulkan command 区间，不能替代端到端墙钟时间。
- 2026-08-27 profile 路由与 timestamp 增量后的入口复验：`run_uma_test.bat` 共 35 项、`run_discrete_simulation_test.bat` 共 22 项，均无非零退出并以 `RESULT: PASS` 结束；新增 profile route contract 在 UMA 日志第 133-136 行、离散模拟日志第 125-128 行均为 `rc=0`。standalone DLL 当前 SHA-256 为 `D894A74998CDED7EA648ACF6FB7BB56B688FCC36EC1CD51CCB21190228A285C7`，未构建 wheel、未更新独显发布包。
- 2026-08-27 standalone 设备诊断一致性补全：新增 `vkssbo_device_info_json()` C API 和 Python `vkssbo.device_info()`，返回 standalone 实际使用的设备、物理设备选择、compute queue、内存拓扑/预算、FP16 与 timestamp 能力；ComfyUI `/vkssbo/profile` 增加独立 `device_source`，优先使用 standalone 设备块，设备接口失败时才单独回退 torch.ops。新增 `test_standalone_device_info_runtime.py` 并接入两条正式入口；本次 UMA 入口记录 39 个 `[run]` 项、离散策略模拟记录 24 个 `[run]` 项，均 `RESULT: PASS`。standalone DLL 当前 SHA-256 为 `9DF783F2387AE058E912DFAC37BC8BBF2E32D21C8CEB6A25E4EA34866E4A0B4B`；未构建 wheel、未更新独显一次性性能包。
- 2026-08-27 profile 设备摘要同步：`bench_operator_profile.py` 的 JSON `device` 块改为优先读取 standalone `device_info()`，同时保留旧 DLL 的显式 `standalone_ssbo_legacy` 回退；新增字段包含 backend、topology、vendor/device/driver。quick profile 与 `test_operator_profile_contract.py` 均通过，避免环境变量设备名与实际 Vulkan 设备不一致。
- 2026-08-27 standalone 设备选择对齐：`core.h` 初始化现在复用 PyTorch 的 `VKSSBO_DEVICE_NAME`（大小写不敏感子串匹配）和 `VKSSBO_DEVICE_INDEX` 规则，记录 `physical_device_count/selected_device_index/device_selection`，非法名称或索引明确回退第一个设备。设备诊断契约覆盖 name、index、name/index fallback；重新构建 DLL 后 UMA 与离散策略模拟完整入口均 `RESULT: PASS`。standalone DLL 当前 SHA-256 为 `02B34D1B27653B4CEAC58E3898CCD8B1CDE34FA4C6EDBC494225C925AD609EE9`。
- 2026-08-27 standalone 能力诊断细化：`device_info()` 改为枚举真实 transfer queue、独立 transfer queue 标志和 timeline semaphore 物理支持；`async_transfer_enabled` 仍固定为 0，明确区分硬件能力与 standalone 尚未启用的异步执行。新增字段一致性断言（queue family/support/dedicated/timeline），native SSBO regression 与三项 profile/device 契约均通过。standalone DLL 当前 SHA-256 为 `6D23519E12D3C3AFCEBE9DC3E4506634C20C5A80FCECD691CA364EE0D552F10E`。
- 2026-08-27 resident sidecar 稳定性补强：resident projected-attention 已在固定输入上连续复验 `max_abs=0.000059`、输入阶段 `upload_bytes=0`；新增 resident-input linear 兼容性回归 `test_resident_linear_runtime.py`，覆盖 2D/3D 输入、FP16/FP32 激活、FP16/FP32 权重、有无 bias 共 16 种组合，并对同一权重的第二次调用确认缓存复用且 `upload_bytes=0`，连续运行最大误差 `0.001702`，清理后 `resident_entries=0`。ComfyUI resident dispatch 回归连续 10 次稳定为 `max_abs=0.000248`、`linear_hits=1`、resident 输入上传 `0`；UMA 与离散策略模拟正式入口均已接入该测试，当前分别覆盖 40/25 项并以 `RESULT: PASS` 结束。该增量只完成 resident linear 的数值、dtype、shape、缓存和生命周期契约，仍不宣称已经实现真正无 host wait 的统一设备 Tensor 或端到端性能收益。新增脚本路径：`D:\ComfyUI Vulkan beat\tools\vulkan_ssbo\test_resident_linear_runtime.py`。
- resident mirror cache 和代表性算子链继续作为回归基础；每个算子仍会为 stock Vulkan Tensor 生成 CPU mirror，后续继续评估 producer metadata、融合算子和独立 TensorImpl storage/`copy_` 协议。stock Vulkan runtime 与 SSBO runtime 使用不同 Vulkan device，不能把自研 `VkBuffer` 伪装成 stock Vulkan storage。
- 2026-08-27 resident graph 异步提交基础完成：新增 `resident_graph.submit_async()` / `wait()`、完成值、提交期间延迟释放 resident buffer，并新增 `test_resident_async_runtime.py` 覆盖输入提前离开 Python 作用域、输出下载边界自动等待、图资源退休和最终 `in_flight/resident_entries` 清零。当前 Arc UMA 与离散策略模拟均通过，异步提交时 `waits=0`、最大误差 `0.000286`；这证明生命周期和提交语义，不宣称跨算子端到端性能收益。该回归已接入 UMA 与离散模拟正式入口。
- 2026-08-28 resident MLP 图批处理增量完成：新增带权重缓存的 `vkssbo_resident_graph_linear`，允许在同一 `ResidentGraph` 中录制 host-weight `linear`；ComfyUI CLIP GELU MLP 在 resident 输入命中时按 `linear -> GELU -> linear` 一次提交，QuickGELU 和不满足条件的路径保留原有回退。新增 `test_resident_graph_linear_runtime.py` 与 `test_comfy_resident_mlp_graph_runtime.py`，固定 FP16 形状下提交次数由 `3` 降为 `1`，最大误差 `0.001640/0.000444`，输入上传 `0`、`queue_wait_idle=0`。占用旧 DLL 的 8195/8196 验证服务在队列为空后已结束，实验 DLL 已提升为正式 `vulkan_ssbo.dll` 并在不设置 `VKSSBO_DLL_NAME` 时完成定向、UMA 与离散策略模拟复验。
- 2026-08-28 resident GEGLU 图批处理增量完成：`ResidentGraph` 新增单一 `geglu()` 接口，内部复用现有 fused GEGLU kernel，并在强制 128-row 路径时把 `matmul -> gate` 两个 dispatch 保留在同一 command buffer；ComfyUI 标准推理态 `FeedForward(GEGLU -> Dropout -> Linear)` 在 resident 输入命中时合并为一次提交。`VKSSBO_RESIDENT_GEGLU_GRAPH=0` 可关闭；训练态非零 dropout、非标准模块布局、梯度、旧 DLL、非法 shape 或预算不足均走原回退。新增 `test_resident_graph_geglu_runtime.py` 与 `test_comfy_resident_geglu_graph_runtime.py`，覆盖 64-row fused、强制 128-row、bias/no-bias、训练 dropout 回退、清理和 ComfyUI 适配：提交 `2 -> 1`，两种 kernel 的 graph/eager 最大误差分别为 `0.000787/0.000710`、交叉误差 `0`，ComfyUI 最大误差 `0.000459`、输入上传 `0`。正式 UMA 入口共 `46` 项、离散策略模拟入口共 `31` 项，均无非零退出并以 `RESULT: PASS` 结束；正式 DLL SHA-256 为 `6102749D341A2633038E402E4DDE165D5E2CD0DAB1B07A77FE8D846804B88213`。该结果只证明兼容性和单段提交减少，不宣称整个 UNet 已图捕获或真实独显性能收益。
- 2026-08-28 resident LayerNorm 图批处理增量完成：`ResidentGraph` 新增 `layernorm()`，复用已有并行归约 shader（256/128 workgroup），支持 2D/3D token 激活、affine/no-affine 和非零 epsilon；`LayerNorm -> Linear` 在同一 graph 中由两次提交降为一次，graph/eager 最大误差 `0.000193/0.000264`，输入上传 `0`。ComfyUI `_layer_norm_ssbo` 仅在 FP16 resident、连续 2D/3D、推理态且最后一维匹配时命中，4D 输入继续走 eager 回退；新增 `test_resident_graph_layernorm_runtime.py` 与 `test_comfy_resident_layernorm_graph_runtime.py`，ComfyUI 命中误差 `0.000481`，回退路径上传 `2048` bytes。`VKSSBO_RESIDENT_LAYERNORM_GRAPH=0` 可关闭；旧 DLL、梯度、非法参数、预算不足均保持回退。该增量仍是迁移期 sidecar graph segment，不代表统一 PyTorch device/storage 或完整 UNet 图捕获。
- 2026-08-28 resident GroupNorm 图批处理增量完成：C ABI、`ResidentTensor` 和 `ResidentGraph` 新增 NCHW `groupnorm()`，复用 `groupnorm_par` 的 FP16 storage/FP32 accumulation 路径，覆盖 affine/no-affine 与不同合法 group 数。ComfyUI `_group_norm_ssbo` 可直接复用 resident 输入；ResBlock 只对实例内已证明为固定前缀的 `GroupNorm -> SiLU` 安装局部 forward，不修改全局 `torch.nn.Sequential`，后续 Conv/其他模块仍走原路径。固定回归提交由 eager `2 -> graph 1`，affine/no-affine 最大误差分别为 `0.000291/0.001997`、交叉误差为 `0`、输入上传为 `0`；ComfyUI GroupNorm/组合链最大误差为 `0.000179/0.000122`，后续模块调用次数为 `1`。`VKSSBO_RESIDENT_GROUPNORM_GRAPH=0` 可关闭；旧 DLL、非 4D、非法 groups、非 resident、inplace SiLU、训练/梯度和预算不足保持原回退。新增 `test_resident_graph_groupnorm_runtime.py` 与 `test_comfy_resident_groupnorm_graph_runtime.py`，本轮 UMA 正式入口共 `50` 项、离散策略模拟共 `35` 项，均无非零退出且只有一个 `RESULT: PASS`。`vulkan_ssbo.dll` 与 `vulkan_ssbo_next.dll` SHA-256 均为 `ED220E7F89548A2A4477D6C5E2EA9F56A2050645DDCBB1828664F505F41D8B97`；未构建 wheel、未更新真实独显性能包。该结果证明数值兼容性和局部提交合并，不代表完整 ResBlock/UNet 图捕获或真实独显性能收益。
- 2026-08-28 resident Conv2D 图批处理增量完成：新增 `vkssbo_resident_graph_conv2d`、`ResidentTensor.conv2d()` 与 `ResidentGraph.conv2d()`，复用现有 FP16 storage、权重缓存和 Conv2D shader 选择；ComfyUI `_conv2d_ssbo` 可消费 resident 输入，并将实例级 ResBlock 固定前缀扩展为 `GroupNorm -> SiLU -> Conv2D`，推理态 `GroupNorm -> SiLU -> Dropout -> Conv2D` 在证明 Dropout 为恒等后也可合并，不修改全局 `torch.nn.Sequential`。3x3、batch-2 1x1、grouped/stride 三种 native 回归均由 eager `3 -> graph 1` 次提交，graph/eager 交叉误差为 `0`，相对手工 float64 参考最大误差分别为 `0.000039/0.000025/0.000029`，bias-free 最大误差不超过 `0.000479`，输入上传为 `0`；ComfyUI direct/ResBlock/eval-Dropout 最大误差为 `0.000454/0.000035/0.000035`。训练 Dropout、梯度、非对称 stride/padding、dilation、非零 padding mode、LoRA weight/bias function、非法 groups/shape、旧 DLL 和预算不足继续回退；`VKSSBO_RESIDENT_CONV2D_GRAPH=0` 可关闭。同步修复两项兼容性边界：eager 1x1 MatMul 特例只在已验证的 batch=1 使用，batch>1 走通用 Conv2D；shader 编译入口补齐 `conv2d_256/tiled8/simple` 的正常 FP16-output SPIR-V 目标。新增 `test_resident_graph_conv2d_runtime.py` 与 `test_comfy_resident_conv2d_graph_runtime.py`；UMA 正式入口共 `54` 个退出记录、离散策略模拟入口共 `39` 个退出记录，全部为 `rc=0`，两份日志各只有一个 `RESULT: PASS`。正式 `vulkan_ssbo.dll` 与 `vulkan_ssbo_next.dll` SHA-256 均为 `76A00DF6244CE53C0363BDD324B176318DB6FA96BB78D4FA082876AF23A00796`；快照位于 `D:\PyTorch for Vulkan\backups\pre-resident-conv2d-20260828-115624`。本轮未构建 wheel、未更新真实独显性能包；这些结果只证明 UMA/离散策略模拟兼容性和局部提交减少，不是完整 ResBlock/UNet 图捕获或真实独显性能结论。
- 2026-08-28 FP16 resident projected-attention 图像验收：实验 DLL 在独立 8197 服务上启用 `VKSSBO_FUSED_XATTN_FP16=1`、resident activation 和 MLP graph，运行 512x512、8 步、seed=42 猫咪 smoke test 成功；profile 记录 `projected_attention_hits=630`、`graph_mlp_hits=64`、`graph_mlp_fallbacks=0`、resident `fallbacks=0`，输出 `std=83.462420`，视觉检查通过。该结果证明注意力 resident 输入复用没有立即造成坏图，但仍保留显式开关，尚未宣称默认 FP16 cross-attention 或端到端性能收益。
- 2026-08-28 FP16 resident projected-attention 尺寸扩展：同一实验配置运行 768x768、8 步、seed=42 猫咪 smoke test，profile 记录 `projected_attention_hits=980`、`graph_mlp_hits=64`、`graph_mlp_fallbacks=0`、resident `fallbacks=0`，输出 `std=63.434698`，视觉检查通过。该结果只扩大数值/图像覆盖，不改变默认 `VKSSBO_FUSED_XATTN_FP16=0` 的安全策略，也不作为性能收益结论。
- 2026-08-28 FP16 fused-attention graph on/off 图像 A/B：以 512x512、8 步、seed=42 猫咪结果为对照，仅切换 `VKSSBO_FUSED_XATTN_FP16`，得到 NCC=`0.947250`、mean absolute pixel difference=`13.263454`，高于阶段 G 的 `0.85` 图像门槛。该结果支持继续保留显式实验开关，但暂不改为所有工作流的默认配置。
- 2026-08-28 GEGLU 正式 DLL 图像验收：在独立 8197 服务上启用 FP16 fused cross-attention、resident activation、MLP graph 和 GEGLU graph，以 512x512、8 步、seed=42 生成窗边猫咪，输出 `vulkan_resident_geglu_formal_smoke_20260828_00001_.png`，像素 mean/std=`137.438960/84.227608`；相对 `vulkan_resident_xattn_next_smoke_20260828_00001_.png` 的 NCC=`0.885711`，通过 `0.85` 门槛且目检未见黑图、无规律色块或鬼图。profile 记录 `graph_geglu_hits=560`、`graph_geglu_fallbacks=0`、`graph_mlp_hits=64`、`projected_attention_hits=560`。服务关闭后用默认正式 DLL 再跑两项 GEGLU 定向测试及 UMA 46 项、离散策略模拟 31 项完整入口，全部通过且无非零退出；`vulkan_ssbo.dll` 与 `vulkan_ssbo_next.dll` SHA-256 均为 `6102749D341A2633038E402E4DDE165D5E2CD0DAB1B07A77FE8D846804B88213`。
- 2026-08-28 离散模拟入口日志可靠性修复：定位到 BAT 混合 `CRLF/LF` 导致 Windows `cmd` 的 `call :label` 返回偏移错误，前 24 项结束后会错误续跑后 8 项，形成两个 `RESULT` 并把主测试数误计为 32。`run_uma_test.bat` 与 `run_discrete_simulation_test.bat` 已统一为纯 CRLF；离散入口新增原子运行锁，并先写每次运行独立的 `*.running.log`，完成后才提升为固定日志。修复后入口按实际存在的脚本连续运行 31 项（可选 `test_conv2d_silu_fusion_runtime.py` 当前不存在），日志恰有一个 `Started/Finished/RESULT`、非零退出为 0、最终 `RESULT: PASS`，锁和临时日志均自动清理。
- 2026-08-28 resident Conv1D 图批处理增量完成：新增 `vkssbo_resident_graph_conv1d` C ABI、`ResidentTensor.conv1d()`/`ResidentGraph.conv1d()` 和 ComfyUI `_conv1d_ssbo` resident 路由。实现复用已验证的通用 Conv2D shader，将 NCL 与 `H=1` NCHW 做布局等价映射，不新增未经验证的 Conv1D shader；覆盖连续 FP16 推理态的 dense、grouped、stride、padding 场景，`dilation!=1`、训练/梯度、无 resident 输入、非法 shape、旧 DLL 或预算不足均保持原 eager/fallback。新增 `test_resident_graph_conv1d_runtime.py` 与 `test_comfy_resident_conv1d_runtime.py`：dense/grouped-stride 最大误差 `0.000209/0.000119`，ComfyUI 路由最大误差 `0.000119`，resident 图均单次 compute submit 且激活 `upload_bytes=0`；不支持 dilation=2 的回退计数正确。更新后的正式 UMA 入口共 `62` 个 `[run]`、离散策略模拟入口共 `47` 个 `[run]`，两者非零退出均为 `0` 且最终均为唯一 `RESULT: PASS`。本轮构建并同步三处运行时 DLL，SHA-256 均为 `8712DDE6CD275AD6650763A1C35959BCC2874D4E6A7C91DF3D5C1529B96C07EC`；备份位于 `D:\PyTorch for Vulkan\backups\pre-conv1d-resident-20260828-191103`。该增量完成 Conv1D resident 兼容性基础，不代表音频 encoder 完整组合图、训练反向或真实独显性能收益。
- 2026-08-28 resident ConvTranspose1D 图批处理增量完成：新增 `conv_transpose1d` FP16 shader、`vkssbo_resident_graph_conv_transpose1d` C ABI、`ResidentTensor/ResidentGraph.conv_transpose1d()` 与 ComfyUI 路由。覆盖连续 NCL 激活、host-weight、bias、stride、padding、`output_padding`、dilation、groups；单次 graph 为一次 compute submit，输入上传 `0`。动态/梯度参数、weight norm、训练态、requires-grad、非连续输入、缺 sidecar、非法几何、旧 DLL 和预算不足均安全回退。独立 resident dense/grouped+dilated 最大误差 `0.000239/0.000176`，ComfyUI 路由 `0.000211`，命中 `1`、回退 `0`。本轮正式 UMA 日志为单次 `72` 项、离散策略模拟为单次 `57` 项，均无非零退出且唯一 `RESULT: PASS`；离散侧验证了强制 staging、离散拓扑和异步请求，仅作为兼容性证据，不代表真实独显性能。测试参考改为独立标量公式，规避本机构建 CPU `conv_transpose1d` 偶发污染 workspace 导致的假误差。三份 DLL SHA-256 一致为 `77DC3F358F6B0D9AFED59659404FC2CDF064CF6793FC497025D920D69603BCE4`。
- 2026-08-28 resident scale-shift ResBlock 图批处理增量完成：新增 `channelwise_scale_shift` FP16 shader/C ABI，直接读取 resident `[B,2*C]` embedding，在 NCHW 激活上执行 `h * (1 + scale) + shift`，不创建 `[B,C,H,W]` 临时张量。标准推理态 2D ResBlock 在 `use_scale_shift_norm=True` 且输入/embedding 均 resident 时录制 `GroupNorm -> scale-shift -> SiLU -> Conv2D -> skip -> residual add`，identity 与 1x1 channel-changing skip 均为单次提交，最大误差 `0.001085/0.000936`，输入上传 `0`。训练 Dropout、up/down、VideoResBlock、LoRA 动态权重、梯度、非法 `[B,2*C]` embedding、旧 DLL 和预算不足保持原回退；`test_comfy_resident_resblock_scale_shift_runtime.py` 已接入 UMA/离散模拟入口，两入口分别 `56/41` 个 run、全部 `rc=0` 且唯一 `RESULT: PASS`。本轮未构建 wheel、未更新真实独显性能包。
- 2026-08-28 resident upsample 图批处理增量完成：新增 `vkssbo_resident_graph_upsample` 和 `ResidentGraph.upsample()`，复用 nearest/bilinear FP16 shader，支持显式二维 size 或 scalar/二维 scale factor；ComfyUI `F.interpolate` 在 4D、无梯度、resident 输入和 `align_corners=False` 时直接走 graph，其他 mode/参数继续走原回退。nearest 最大误差 `0`，bilinear 最大误差 `0.000558`，两种模式均单次提交且输入上传 `0`；`test_resident_graph_upsample_runtime.py` 已接入 UMA/离散模拟入口，两入口随后继续通过 up/down ResBlock 扩展回归。VideoResBlock 和固定 2x2 下采样在本条记录时尚未迁移，后续状态见 up/down 增量记录。

- 2026-08-28 ComfyUI 标准 LoRA resident 图批处理增量完成：新增受限 `Linear + LoRA` 路径，仅接受单一标准 `LoRAAdapter`、二维 FP16 resident 输入、无 mid/DoRA/reshape/offset/自定义函数且无 bias function 的推理场景；基础 Linear 和低秩 `down -> up` 增量在同一个 resident graph 中提交，LoRA 因子按源 Tensor 身份和版本缓存，原地更新会替换旧 resident 参数。复杂补丁、卷积 LoRA、训练/梯度、非二维输入、非 CPU 参数、非法 shape 和旧 DLL 均保持原始 ComfyUI 权重函数回退；`VKSSBO_RESIDENT_LORA=0` 可关闭。新增 `test_comfy_resident_lora_runtime.py`，覆盖首次数值、参数更新、缓存失效以及 mid/自定义函数/bias function 回退，定向最大误差 `0.003842/0.005835`。旧 AST 隔离测试的清理夹具已改为兼容缺失的新缓存全局变量；修复后 UMA 正式入口和离散策略模拟入口分别完成 `60` 和 `45` 个运行项，全部 `rc=0` 且各自只有一个 `RESULT: PASS`。本轮未构建 wheel、未更新真实独显性能包。

本轮增量（2026-08-28，音频 Conv1D 激活组合）：新增受限的 resident
`Conv1D -> SiLU/GELU` 组合回归，复用 `ResidentGraph.conv1d()` 与
`unary_chain()`，两种激活均保持单次 compute submit、激活上传为 `0`，FP16
输出相对 float32 参考的最大误差小于 `0.001`。ComfyUI 仅对音频编码器
`wav2vec2.ConvNoNorm` 的确定性 `Conv1D -> GELU` 结构安装实例级 forward 路由，
不修改全局 `Sequential`；FP16、推理态、无梯度且输入有 resident sidecar 时命中，
非 resident、sidecar 失效、训练/梯度、非法卷积几何或旧 DLL 均继续走原始实现。
新增 `test_resident_graph_conv1d_activation_runtime.py` 与
`test_comfy_resident_conv1d_activation_runtime.py`，并接入 UMA/离散策略模拟正式
入口；ComfyUI 定向回归 `max_abs=0.001254`、命中 `1`、回退 `1`。该增量只覆盖
音频常见 Conv1D 激活组合，不代表 LayerNorm/GroupNorm、GLU、Snake、ConvTranspose1d
或完整音频模型已图捕获；本轮不构建 wheel、不更新真实独显性能包。

本轮增量（2026-08-28，音频 Conv1D 归一化组合）：新增受限的 resident
`Conv1D -> LayerNorm -> GELU` 与 `Conv1D -> GroupNorm -> GELU` 图路径。LayerNorm
路径在同一 resident graph 中显式完成 NCL/NLC 重排、最后维度归一化和 GELU；GroupNorm
路径复用 NCHW GroupNorm shader 的 `H=1,W=L` 等价布局，并通过独立
`vkssbo_resident_graph_groupnorm1d` ABI 保持形状契约清晰。两条路径均为单次 compute
submit、激活上传为 `0`；standalone 定向最大误差为 `0.000063/0.000471`，ComfyUI
路由最大误差为 `0.001341/0.002177`。仅对 `wav2vec2.LayerNormConv` 与
`LayerGroupNormConv` 的 FP16、eval/no-grad、连续 NCL、CPU 浮点参数、标准 stride/padding
和 dilation=1 结构安装实例级 forward，不修改全局 `Sequential`；训练/梯度、非 resident、
动态权重、非法几何、旧 DLL 或预算不足均回退原始实现。新增
`test_resident_graph_audio_norm_conv1d_runtime.py` 与
`test_comfy_resident_audio_norm_conv1d_runtime.py` 并接入 UMA/离散策略模拟正式入口。
通用 chunk/mul、完整音频模型端到端仍未迁移；标准 GLU/SnakeBeta 已在后续增量中
完成，具体支持范围和验收数据见下方同日记录；ConvTranspose1d
已在后续增量中完成，具体支持范围和验收数据见上方同日记录；本轮
不构建 wheel、不更新真实独显性能包。

已知风险：

- 当前 stock Vulkan runtime 与 SSBO runtime 使用独立 Vulkan device，不能假设两个 runtime 可以直接共享 `VkBuffer`。
- host lease 是带弱引用和版本检查的过渡型 mirror 方案，代表性连续算子链已通过 A/B；它仍不等价于真正的 resident Tensor，view、in-place 和 autograd 扩展前必须继续保留 materialize fallback。
- producer metadata 同样是过渡层：普通 Tensor 使用 `_version()` 检查 in-place 失效；inference Tensor 没有版本计数，只能记录为 advisory metadata，任何融合实现必须在无法证明无别名/无 in-place 时回退。
- Intel 驱动需要显式 dispatch barrier；不能用无同步的多 dispatch 取代现有同步。
- fp16 存储会产生可见舍入误差，必须持续报告 `max_abs`、均值误差和图像 NCC。

本轮增量（2026-08-28）：新增完整标准 ResBlock resident graph。原生侧增加
`channelwise_add` 专用 NCHW `[B,C]` 广播 shader/C ABI，避免把时间嵌入展开为
`[B,C,H,W]` 临时张量；ComfyUI 对标准 2D、推理态、无 scale-shift、无上下采样、无
LoRA 动态权重、eval Dropout 且 `x/emb` 均有 resident sidecar 的 `ResBlock` 录制
`GroupNorm -> SiLU -> Conv2D -> (SiLU -> Linear -> channel add) -> GroupNorm ->
SiLU -> Conv2D -> skip -> residual add`，整块一次提交。identity skip 与 1x1
channel-changing skip 的 batch-2 回归最大误差为 `0.000971/0.001117`，提交数为 `1`、
输入上传为 `0`；通道广播独立回归最大误差为 `0.000977`。训练 Dropout、缺少 embedding
sidecar、动态 weight/bias function、梯度、VideoResBlock、scale-shift、up/down、旧
DLL、非法 shape 和预算不足均回退；`VKSSBO_RESIDENT_RESBLOCK_GRAPH=0` 可关闭。新增
`test_resident_graph_channelwise_add_runtime.py` 和
`test_comfy_resident_resblock_graph_runtime.py`，并接入 UMA/离散策略模拟入口；全量入口
已完成：UMA `56` 个退出记录、离散策略模拟 `41` 个退出记录，均为 `rc=0` 且各只有一个
`RESULT: PASS`。正式 `vulkan_ssbo.dll` SHA-256 为
`E2288A7C85EAA13C8B619A95DCD06FB690D2C4632BA9C467699A5B3E57D680D4`，
`vulkan_ssbo_next.dll` SHA-256 为
`8132ACB867A18544A01C0992A43DE0F497D273C73EBE8C8638CAB07999682916`。
本轮未构建 wheel、未更新真实独显性能包。

本轮增量（2026-08-28，scale-shift）：新增
`channelwise_scale_shift` 专用 NCHW `[B,2*C]` 参数 shader/C ABI，避免将 scale 和
shift 展开为 `[B,C,H,W]` 临时张量；ComfyUI 标准 2D、推理态、无上下采样、无 LoRA
动态权重、eval Dropout 且 `x/emb` 均有 resident sidecar 的 ResBlock 在
`use_scale_shift_norm=True` 时录制 `GroupNorm -> scale-shift -> SiLU -> Conv2D ->
skip -> residual add`，identity 与 1x1 skip 均为一次 graph 提交。训练 Dropout、
VideoResBlock、up/down、梯度、旧 DLL、非法 embedding shape 和预算不足均回退；新增
`test_comfy_resident_resblock_scale_shift_runtime.py` 并接入两条正式入口，定向最大误差
`0.001085/0.000936`，两入口最终均为 `RESULT: PASS`。

本轮增量（2026-08-28，up/down）：新增固定 2x2/stride-2 `avgpool2d` FP16 shader、
C ABI、`ResidentTensor.avgpool2d()` 和 `ResidentGraph.avgpool2d()`；ComfyUI 标准 2D、
推理态、无 LoRA 动态权重且 `x/emb` 均有 resident sidecar 的上/下采样 ResBlock 现在
分别在主分支与 skip 分支使用 resident nearest 2x 或 avgpool2d，并可与 scale-shift
组合。上采样、下采样、上采样加 scale-shift 的定向回归最大误差为
`0.001419/0.000675/0.001306`，三种路径均单次 graph 提交、输入上传 `0`、无
`queue_wait_idle`；`test_comfy_resident_resblock_updown_runtime.py` 已加入 UMA 与离散
策略模拟正式入口。非 2D/非固定 2x 参数、VideoResBlock、训练/梯度、LoRA 动态权重、
旧 DLL、非法尺寸和预算不足均保持原回退；`VKSSBO_RESIDENT_RESBLOCK_GRAPH=0` 仍可
关闭整条路径。完整复验：UMA `57` 个 run、离散策略模拟 `42` 个 run，均无非零退出，
各自日志只有一个 `RESULT: PASS`。本次正式 `vulkan_ssbo.dll` SHA-256 为
`804072198A1C35885EDB74D2685EA3B8D88A8077F38D9CA20EE5C985130A8B2C`；快照位于
`D:\PyTorch for Vulkan\backups\post-resident-updown-20260828-160403`。本轮未构建 wheel、
未更新真实独显性能包。

本轮增量（2026-08-28，resident Conv3D）：新增 `vkssbo_resident_graph_conv3d` 原生
C ABI、`ResidentTensor.conv3d()` 和 `ResidentGraph.conv3d()`，复用现有 Conv3D
FP16 shader、权重缓存和 resident 生命周期管理；支持 NCDHW、stride、padding、
groups 及无 bias/有 bias，dilation 仅接受全 1，其他 dilation 明确回退到原始
PyTorch Conv3D。standalone dense 与 grouped/stride 回归的 graph/eager 交叉误差均为
`0`，相对 eager 最大误差分别为 `0.000460/0.000457`，无 bias 最大误差为
`0.000486/0.000260`；ComfyUI resident 路由回归 `max_abs=0.000487`、命中 `1`、
不支持 dilation 回退 `1`。新增 `test_resident_graph_conv3d_runtime.py` 和
`test_comfy_resident_conv3d_runtime.py`，已接入两条正式入口；UMA 完整日志
`D:\PyTorch for Vulkan\pytorch-src\uma_test.log` 共 `59` 个 `[run]`、唯一
`RESULT: PASS`，离散策略模拟日志 `D:\PyTorch for Vulkan\pytorch-src\discrete_simulation_test.log`
共 `44` 个 `[run]`、唯一 `RESULT: PASS`，均无非零退出。正式 standalone DLL
（`tools\vulkan_ssbo\vulkan_ssbo.dll`、`vulkan_ssbo_next.dll` 及 ComfyUI backend
副本）SHA-256 为 `63B2AB67309214EF8D844D977940D00F1BCE3F8195F67D5B249B1781284D90F1`。
完整 `VideoResBlock` 的 4D/5D view/rearrange、时间堆栈、`AlphaBlender` 及 resident
别名生命周期尚未证明安全，仍保持原始路径回退；本轮未构建 wheel、未更新真实独显性能包。

本轮增量（2026-08-28，完整 VideoResBlock）：完成标准推理态 `VideoResBlock` 的
resident graph 路由，覆盖空间 ResBlock、4D/5D layout rearrange、5D GroupNorm、
时间 Conv3D、时间嵌入重排与 channelwise add、AlphaBlender 混合和最终布局恢复。
仅在 FP16 resident activation、eval/no-grad、无 `image_only_indicator`、固定时间帧数、
非 up/down、标准模块结构、无动态 weight/bias function 且 `fixed/learned` alpha 时命中；
`learned_with_images`、训练/梯度、异常布局或旧 DLL 均可靠回退。定向回归
`test_comfy_resident_video_resblock_runtime.py` 通过，`max_abs=0.001344`、
`graph_video_resblock_hits=1`、单次提交、输入上传为 `0`；布局/5D GroupNorm 与既有
Conv2D 回归继续通过。期间修正测试参考中的本机构建 CPU 多批次 Conv2D 错误，未放宽
数值门槛。临时诊断脚本已清理；本轮仍未构建 wheel、未更新真实独显性能包，正式 UMA/
离散策略模拟全量入口待随后复验。

正式入口复验（2026-08-28）：上述 VideoResBlock 回归已接入两条正式入口。UMA
`run_uma_test.bat` 共 `60` 个 `[run]`、所有退出码为 `0`、唯一最终
`RESULT: PASS`；其中 VideoResBlock `max_abs=0.001332`、单次提交、命中 `1`。
离散策略模拟 `run_discrete_simulation_test.bat` 共 `45` 个 `[run]`、所有退出码为
`0`、唯一最终 `RESULT: PASS`；其中 VideoResBlock `max_abs=0.001324`、单次提交、
命中 `1`。本轮正式 standalone DLL 与 backend 副本 SHA-256 均为
`6751DEFD2FE17CE8F17DAD890E9BA310ADA54A97B01229870BE3953480CDF15A`；备份位于
`D:\PyTorch for Vulkan\backups\pre-video-formal-20260828-184639` 和
`D:\PyTorch for Vulkan\backups\pre-video-runner-integration-20260828-184616`。

本轮增量（2026-08-29，T10 真正 FP32 resident Conv2D）：完成独立的 FP32
storage 变体和边界契约。`Dispatch::run_variant()` 按 resident storage dtype
选择 `conv2d_resident_fp32.spv` 或 `conv2d_resident_fp16.spv`；FP32 resident
绕过旧 FP16 权重缓存，Python 层禁止 FP32 resident 与 FP16 activation 更新路径
混用，并对 bias dtype、非连续输入、原地修改和旧 DLL 保留明确回退。两种 shader
均使用 FP32 累加，FP16 变体通过 `-DFP16_STORAGE` 编译。

定向验收全部通过：FP32 resident Conv2D `max_abs=0.00000024`，FP32 边界、
FP16 resident 兼容性、三种 resident Conv2D 形状以及 ComfyUI Conv2D/ResBlock
路由均通过，Python 源码编译检查通过。正式 10 轮交错 A/B（warmup=2、repeat=5、
seed=42；误差门为 `resident_vs_eager <= 2 * eager_vs_CPU_reference`）中，
`conv320` eager/FP32 resident 平均 median 为 `42.562/204.320 ms`（4.805x），
`conv640` 为 `34.385/205.324 ms`（5.989x）；两个 shape 的误差门均通过，
但 FP32 resident 性能明显变慢。因此正式决策为
`keep_fp32_conv_resident_capability_closed`：保留为显式实验能力，默认关闭，
不扩展 FP32 Linear 或 FP32 GroupNorm。若重新开启，必须先实现 tiled/vectorized
FP32 kernel，并按同一 10x2x5 规则复测。基准记录为
`tools\\vulkan_ssbo\\bench_fp32_conv_ab_study_20260829-084607.json`；当前正式
`vulkan_ssbo.dll` SHA-256 为
`A78E159225FE9EF6C112A55086DEC5F4B84079700AF8BCAF67978FEF0D2CB1A7`。
本轮未构建 wheel、未更新真实独显性能包。

本轮增量（2026-08-29，T10 FP32 tiled/vector 候选）：在同一
`conv2d_x32v.comp` 中增加 `F32_STORAGE` 变体，使用真正 FP32 的输入、权重、
bias、输出 storage，并保留 16x16 workgroup、共享 tile、32 路输出通道和 FP32
累加；编译产物为 `conv2d_x32v4_f32storage_fp32.spv`。`VKSSBO_RESIDENT_FP32_TILED=1`
只对已验证的 3x3/stride-1/groups=1 FP32 resident Conv2D 选择该候选，未设置时
继续使用标量 `conv2d_resident_fp32.spv`；其他几何和默认 FP16 路径不变。

独立子进程契约测试确认标量与 tiled 两种变体均为
`max_abs=0.00000024`，FP32 storage、清理和边界检查均通过；`test_torch.py`
保持 `14 OK, 0 FAIL`，native SSBO regression 通过。正式 10 轮交错 A/B（每轮
warmup=2、repeat=5、seed=42）结果：`conv320` tiled 胜出 `7/10`，平均 median
`42.624 -> 41.158 ms`（约 `1.036x`）；`conv640` 胜出 `0/10`，平均 median
`33.720 -> 42.296 ms`（约 `0.798x`）。两种 shape 的误差门全部通过，但跨 shape
收益不稳定，不能作为通用默认或自动调优候选。因此决策为继续
`keep_fp32_conv_resident_capability_closed`：tiled 版本仅保留显式实验开关，
不扩展 FP32 Linear/GroupNorm，也不改变现有 FP16 默认 kernel。基准记录为
`tools\\vulkan_ssbo\\bench_fp32_tiled_ab_study_20260829.json`。

本轮正式入口复验：`run_uma_test.bat` 共 `74` 个 `[run]`，
`run_discrete_simulation_test.bat` 共 `59` 个 `[run]`；两者非零退出均为 `0`，
且各自只有一个 `RESULT: PASS`。两条入口均实际执行 FP32 resident 标量/tiled
双变体契约。本轮未构建 wheel、未更新真实独显性能包。

本轮增量（2026-08-29，T10 FP32 tiled 能力门控）：为显式 FP32 tiled
resident Conv2D 增加 Vulkan `maxComputeSharedMemorySize` 能力检查。该变体的
实际 shared-memory 需求为 `39,168` bytes（18x18x16 FP32 输入 tile 加
8x16x9 vec4 FP32 权重 tile）；低于该上限时，即使设置
`VKSSBO_RESIDENT_FP32_TILED=1` 也自动回退到标量 `conv2d_resident_fp32`，并在
`VKSSBO_TRACE_RESIDENT_FP32=1` 下记录回退原因。`device_info()` 新增
`resident_fp32_tiled_shared_bytes` 与 `resident_fp32_tiled_capable` 字段。
当前 Arc UMA 报告 `max_shared_memory=49,152`，因此显式实验请求实际选择 tiled；
默认 FP32 resident 能力门仍关闭，不改变 FP16 默认路径。

本轮定向契约、shader 编译、standalone device-info、native SSBO、`test_torch.py`
（14/14）均通过；完整 `run_uma_test.bat` 共 `74` 项、
`run_discrete_simulation_test.bat` 共 `59` 项，所有退出码为 `0` 且各自唯一
`RESULT: PASS`。本轮 standalone DLL SHA-256 为
`94350A54C23A9EB7AB36A5D492007EE41319F734BE32DA96E78D103AE31B457D`，
`conv2d_x32v4_f32storage.spv` SHA-256 为
`2CF39E218190345A6CD0366A6C0DC1C2AC41FA3F60BF566AA15664E3745593F1`。
备份位于 `D:\PyTorch for Vulkan\backups\pre-fp32-tiled-gate-20260829-093859`
及 `pre-fp32-tiled-taskbook-20260829-095912`。本轮未构建 wheel、未更新真实独显性能包。

本轮增量（2026-08-29，4x resident 超分链路兼容性契约）：新增
`tools\vulkan_ssbo\test_resident_4x_superres_runtime.py`，固定验证
`Conv2D -> SiLU -> bilinear 2x -> Conv2D -> SiLU -> bilinear 2x` 的
FP16 resident graph。输入 `(1,4,8,8)` 最终输出 `(1,3,32,32)`，相对独立 FP32
参考的 `max_abs=0.024822`、`mean_abs=0.001258`；整条链只有 `1` 次 compute
submission 和 `6` 个 dispatch，中间激活上传为 `0`，最终 materialize 下载
`6,144` bytes。UMA 与离散策略模拟定向回归均通过。该测试只证明 4x Vulkan SSBO
resident 链路的数值、提交、传输和清理契约，不代表真实超分模型画质，也不构成
1080P/4x 或真实独显性能结论。

同时补齐 resident materialize 下载 profiling：`resident_download_impl` 和
`resident_download_fp16_impl` 现在都记录下载调用次数、字节数及 host 耗时；上述
4x 回归确认最终 FP16 输出为 `download_calls=1`、`download_bytes=6144`，避免 profile
继续把真实 resident 下载误报为零。更新后的正式 standalone
`vulkan_ssbo.dll` SHA-256 为
`327AAECF69B8E9A6825B7AE01211216307B7C4909D12BFE34AC9A546FCBC8AAC`。

本轮完整入口复验：`run_uma_test.bat` 共 `75` 个 `[run]`、`77` 条退出记录；
`run_discrete_simulation_test.bat` 共 `60` 个 `[run]`、`62` 条退出记录。两份日志
所有退出码均为 `0`，无 traceback/AssertionError，且各自只有一个最终
`RESULT: PASS`。离散入口仍只验证强制 staging、预算、timeline、能力门和 fallback
等兼容性语义，不报告真实性能。本轮未构建 wheel、未更新真实独显一次性性能包；
任务书修改前备份位于
`D:\PyTorch for Vulkan\backups\pre-4x-taskbook-final-20260829-102442`。

本轮增量（2026-08-29，T11 迷你 RRDBNet 风格 4x 端到端兼容性）：按已锁定的
`3->64`、2 个 RRDB block（每块 2 层 dense Conv2D/LeakyReLU/cat 残差，scale=0.2）、
两级 `64->256 + PixelShuffle(2)` 和 `64->3` 拓扑，新增
`tools\vulkan_ssbo\test_rrdbnet_4x_e2e_runtime.py`。未新增 native shader/API；
Conv2D、cat 和残差算术走 standalone Vulkan SSBO，LeakyReLU 与 PixelShuffle 在
测试适配层显式 CPU fallback，并记录每种路径计数。

256x256 与 512x512 参考档的 FP32 tiled/eager/CPU 相对误差门均通过，FP16 按 T11
约定只记录；1080P 档输入 `(1,3,1080,1920)` 在 `tile_core=256、halo=8` 下完整
输出 `(1,3,4320,7680)`，FP32/FP16 均无 NaN/Inf、40 tiles、内存估算分别约
`681,836,544`/`340,918,272` bytes，均低于驱动 heap 80% 安全余量，清理后
`cache/resident/staging/pool` bytes 与 entries 全部归零。1080P 定向结果分别为
FP32 `172317.75 ms`、FP16 `85178.63 ms`；耗时只作 UMA 记录，不设性能门。
四个固定 64x64 抽检坐标为输入 `[138,863]`、`[861,1744]`、`[478,998]`、
`[146,1640]`，官方两档 JSON 均 PASS。

为支持独立复核，导出正式 `seed=42` 合成 state_dict：
`D:\ComfyUI Vulkan beat\tools\vulkan_ssbo\t11_rrdb_mini_weights.pt`，文件
`2,087,463` bytes，文件 SHA-256 为
`154308D18FAEDB4D2298565F10C493AD0F9AB494DD3C15D670A1BAE75E30832F`，规范化
state_dict SHA-256 为
`BE06CA42B7EDF08BB2701047D504725EF28C7F0118CFD6797A206BA08645F58F`。CPU 单线程
FP32 参考整图摘要和四 tile 摘要固化在
`tools\vulkan_ssbo\t11_rrdb_reference_digests.json`；1080P 整图摘要为
`DD373556EF6B7EBE706676C616020B63E88E9D3069827006EB820F199E522014`。ZCode 在
`D:\PyTorch for Vulkan\zcode-scratch` 独立复算 256 档并确认官方误差处于同量级，
确认差异来自残差算术实现顺序而非缺陷；未修改 `tools\vulkan_ssbo`。

T11 已接入两个正式入口并完成复验：UMA `uma_test.log` 共 `76` 个 `[run]`、
`78` 条退出记录，新增 RRDB 项 `rc=0`；离散策略模拟
`discrete_simulation_test.log` 共 `61` 个 `[run]`、`63` 条退出记录，新增 RRDB
项 `rc=0`。两份日志均无非零退出且各自只有一个 `RESULT: PASS`；离散结果只证明
强制 staging/预算/timeline/fallback 兼容性，不代表真实独显性能。正式 standalone
DLL SHA-256 仍为 `327AAECF69B8E9A6825B7AE01211216307B7C4909D12BFE34AC9A546FCBC8AAC`。
T11 备份位于 `D:\PyTorch for Vulkan\backups\pre-t11-bat-integration-20260829-182200`
和 `pre-t11-taskbook-final-20260829-185100`；8 通道性能实验已隔离至
`D:\PyTorch for Vulkan\backups\t11-c8-experiment-20260829-181300`，不作为验收依据。

本轮增量（2026-08-29，T12R 离散策略兼容性矩阵最终复验）：在确认脚本写入已停止后，
使用当前便携运行时重新执行 `tools\vulkan_ssbo\bench_discrete_matrix_json.py`，生成
`D:\ComfyUI Vulkan beat\tools\vulkan_ssbo\discrete_matrix_20260829-200131.json`。
矩阵覆盖 26 个单元格（64/1024/4096 MiB、strict、async request、force staging 对照，
FP32/FP16），共 182 条 workload 记录：178 条 `PASS`，4 条为严格 64 MiB 下
`probe_big` 超过 80% reserve 的 `EXPECTED_REJECTION`。`conv_silu`、`linear`、
`matmul_softmax`（3D `[8,1024,128]` -> `[8,1024,1024]`）、`group_norm`、`upsample`
和 `probe_small` 在 26/26 单元格均返回有限输出；`probe_big` 在 1/4/4 GiB 以及非严格
对照中通过。异步请求在模拟离散策略下正确启用 timeline 资源，force staging 对照的
staging 计数为零；所有单元格 `empty_cache()` 后模拟显存与 resident/pool/staging/weight
条目均为零。`test_discrete_matrix_schema_runtime.py --matrix` 和 live smoke 均通过，
最终 schema 输出为 `PASS`。本轮仅验证 UMA 宿主上的离散策略模拟兼容性，不构成真实独显
性能结论；未重建 DLL、未构建 wheel、未更新独显一次性性能整合包。脚本备份位于
`D:\PyTorch for Vulkan\backups\pre-t12r-final-matrix-20260829-194937`。

本轮增量（2026-08-29，T13 跨算子状态稳定性筛查）：为复核此前
`matmul_softmax` storage-less/未定义输出与 FP16 upsample 非有限值是否存在共同的
运行时状态污染，使用固定 `seed=424242`、独立子进程、强制离散策略模拟、异步请求和
强制 staging，覆盖 `matmul`、`upsample`、`matmul -> upsample`、`upsample -> matmul`
四种顺序及 FP32/FP16 两种 dtype。每个配置运行 20 个独立 trial，每个 trial 内连续
调用 10 次；结果写入
`D:\PyTorch for Vulkan\zcode-scratch\t13_trials_20260829-200832.json`。
共完成 2,400 次调用，子进程成功率 `160/160`，storage-less `0`、非有限输出 `0`、
异常 `0`；所有调用均返回预期 shape，策略模拟下未出现跨算子状态复现。该结果支持
将此前问题限定为已修复/已隔离的历史运行状态，不宣称真实独显上的并行性能收益；本轮
未重建 DLL、未构建 wheel、未更新独显一次性性能整合包。

本轮增量（2026-08-29，音频激活 resident graph 补齐）：在已有 Conv1D、归一化和
ConvTranspose1D 基础上，完成标准 gated projection 的 resident `GLU`/`GLU gate`
及通道参数化 `SnakeBeta` 路由，并完成 ComfyUI 音频模块实例级适配。定向回归覆盖
resident GLU 的 SiLU/GELU/Sigmoid/Sin 激活（含 bias/no-bias）、ComfyUI 音频 GLU
三种激活、SnakeBeta 的 log-scale 参数和训练/非连续输入回退；所有回归均为
`PASS`，代表性最大误差分别为 `0.000244`（resident GLU）、`0.000244`
（ComfyUI 音频 GLU）和 `0.001953`（SnakeBeta），命中路径均为单次 compute submit、
输入上传 `0`，清理后 resident 条目归零。该增量只覆盖已约束的标准音频组合，不宣称
通用 `chunk/mul` 图捕获或完整音频模型端到端完成；本轮未重建 DLL、未构建 wheel、
未更新独显一次性性能整合包。

随后完成正式入口复验：`run_uma_test.bat` 共 `76` 个 `[run]`、`78` 条退出记录，
`run_discrete_simulation_test.bat` 共 `61` 个 `[run]`、`63` 条退出记录；两份临时日志
均只有一个 `RESULT: PASS`，所有退出码为 `0`，无 traceback/AssertionError。离散入口
结束时固定日志提升曾因 Windows 文件占用返回归档错误，已在确认临时日志完整且测试成功
后恢复为 `D:\PyTorch for Vulkan\pytorch-src\discrete_simulation_test.log`；该归档问题
不影响测试结果。两条入口仍分别代表 UMA 真实性能/兼容性和离散策略模拟兼容性，未据此
宣称真实独显性能，也未构建 wheel 或更新独显一次性性能整合包。

本轮增量（2026-08-29，受限通用 resident chunk/split + mul）：新增
`chunk_last` FP16 shader、`vkssbo_resident_graph_chunk` C ABI，以及
`ResidentTensor/ResidentGraph.chunk()` 和等价的 `split()` 适配。当前版本明确只接受
连续 2D/3D FP16 推理张量、沿最后一维等分为 2 份；两个输出各自保持 resident，随后
可在同一 graph 中使用现有二元 elementwise 乘法，不发生 host materialize。定向回归
最大误差 `0.000062`，单次 command submit、3 个内部 dispatch、输入上传 `0`，释放后
`resident_entries=0`；奇数宽度、非最后维、非二等分、旧 DLL、FP32 resident、训练/梯度
和预算不足均在边界处报错并由调用方保留 eager/fallback。新回归已接入两条正式入口：
UMA `77` 个运行项、离散策略模拟 `62` 个运行项，均无非零退出且各自只有一个
`RESULT: PASS`。本轮同步的三处 DLL SHA-256 为
`70B11FDE2E77FCF911E424DFE2B080D20AA3F61A000E656535ECAF29D9C77153`；备份位于
`D:\PyTorch for Vulkan\backups\pre-resident-chunk-mul-20260829-213638`、
`pre-resident-chunk-mul-dll-20260829-215802` 和
`pre-resident-chunk-mul-taskbook-20260829-223130`。该增量完成受限通用 split/mul
兼容性基础，不代表任意维度/不等分 view、训练反向或完整音频模型端到端迁移；本轮
仍未构建 wheel、未更新真实独显一次性性能整合包。

本轮增量（2026-08-30，Wav2Vec2 Transformer resident 结构级闭环）：新增
`vkssbo_resident_graph_projected_attention` C ABI，以及
`ResidentTensor.projected_attention()`/`ResidentGraph.projected_attention()`。Q/K/V
投影及 bias、多头布局变换、FP32 attention score、Softmax、value 聚合、输出投影及
output bias 均保留在同一 resident graph 提交内，激活使用 FP16 storage。ComfyUI
对标准 stable LayerNorm Wav2Vec2 编码层接入
`LayerNorm -> Attention -> Residual -> LayerNorm -> GELU MLP -> Residual` 实例级路由，
由 `VKSSBO_RESIDENT_AUDIO_TRANSFORMER_GRAPH` 控制；仅 FP16、eval/no-grad、无 mask、
CPU 静态参数和标准模块结构命中，训练/梯度、动态权重、mask、非标准结构、旧 DLL 和
预算不足均回退原实现。

新增 `test_resident_graph_audio_transformer_runtime.py`、
`test_comfy_resident_audio_transformer_runtime.py` 和
`test_resident_audio_encoder_mini_e2e_runtime.py`。最终正式 UMA/离散策略模拟结果一致：
底层 Transformer 图 `max_abs=0.000276`、单次提交、`16` 个 dispatch、激活上传 `0`；
ComfyUI 路由 `max_abs=0.000352`、命中 `1`、requires-grad 回退 `1`、单次提交；缩小版
完整音频 encoder 覆盖 ConvFeatureEncoder、FeatureProjection、分组位置卷积和一个
Transformer block，输出 `(1,96,32)`，`max_abs=0.000422`、单次提交、`29` 个
dispatch、激活上传 `0`。三项测试的 CPU oracle 均固定 PyTorch intra-op/inter-op
单线程，并将误差门收紧为 `0.01`；修复前已证明相同输入/权重下 Vulkan 中间结果稳定、
漂移来自本机多线程 oneDNN/OMP 参考路径，修复后三项各 6 个独立进程的误差跨度均为
`0`，mini encoder 额外 10 进程稳定性循环同样通过。

最终完整入口复验：`run_uma_test.bat` 共 `80` 个 `[run]`、`82` 条退出记录；
`run_discrete_simulation_test.bat` 共 `65` 个 `[run]`、`67` 条退出记录。两份日志的
非零退出均为 `0`，各自只有一个最终 `RESULT: PASS`。三处正式 DLL SHA-256 均为
`D8CA66EE9145A02C9B51393B450E6948F285AC6C434755EFD5CCAF2EC203AB94`；最终文档备份位于
`D:\PyTorch for Vulkan\backups\audio-transformer-final-docs-20260830-001810`。
本轮完成的是合成权重下的结构级 encoder smoke 和实际 Wav2Vec2 Transformer 层路由；
本机没有真实预训练音频 encoder 权重，因此不得宣称真实权重语义/音质端到端验收完成。
本轮未构建 wheel、未更新真实独显性能包，离散模拟结果仍不构成真实独显性能结论。

本轮增量（2026-08-30，F1 集成侧 fallback 诊断门控）：在
`aten/src/ATen/native/vulkan/ops/ssbo/SsboBackend.cpp` 的
`matmul_softmax` 路由增加默认关闭的 `VKSSBO_TRACE_FALLBACK=1` 诊断开关。
开启后以单行 stderr 记录 `op=matmul_softmax`、`q_shape`/`k_shape`、dtype、
transpose/scale 和明确原因，原因枚举统一为 `input_guard`、`scale_guard`、
`native_undefined`、`native_null_fallback`、`exception_std`；覆盖输入守卫、非法
scale、native 返回空、非融合 fallback 中间结果未定义及
`std::exception`；未知异常只记录 `unknown_exception_rethrow` 后继续抛出，保持原有传播
语义。默认关闭时成功路径、未定义结果和异常 fallback 行为均未改变；本轮不扩展为全局
日志重构，也未触碰 standalone DLL。

新增 `test/test_vulkan_ssbo_fallback_trace_runtime.py` 与便携测试根目录中的
`test_matmul_softmax_fallback_trace_runtime.py`，验证默认静默、`input_guard`、
`scale_guard`、shape/dtype 字段及返回行为。增量构建后的集成 `torch_cpu.dll` 已同步至源码和
`tools/Python313` 两个实际运行时，SHA-256 均为
`8BE92FB682ADF03702A980EDD0A2038EC1918B9E29D4BEAEBE16D4E810A59E4C`。
定向 MatMul+Scale+Softmax 回归 `max_abs=0.000849`、scheduler lifecycle
`submitted_delta=1/in_flight=0` 均 PASS；T16 严格门禁
`t16_contract_20260830-012238.json` 为 `10 PASS`，两份 fallback trace 定向回归均 PASS；
正式 `run_uma_test.bat` 共 `81` 个运行项、
`83` 条退出记录，离散策略模拟共 `66` 个运行项、`68` 条退出记录，退出码均为 `0`，
各自唯一 `RESULT: PASS`，正式日志无未授权 trace 行。备份位于
`D:\PyTorch for Vulkan\backups\pre-f1-catch-trace-20260830-003119` 和
`D:\PyTorch for Vulkan\backups\pre-f1-catch-trace-final-20260830-010416`；本轮契约修正备份位于
`D:\PyTorch for Vulkan\backups\f1-trace-contract-align-20260830-011414`、
`D:\PyTorch for Vulkan\backups\t16-contract-fix-20260830-011200` 和
`D:\PyTorch for Vulkan\backups\taskbook-f1-contract-20260830-012336`。
本轮未构建 wheel、未启动真实独显性能测试；离散模拟仍只作为兼容性证据。

T17 fp32 upsample 边缘映射调查已闭环：`zcode-scratch/t17_findings.md` 证明 nearest
输出逐像素遵循 `floor(i/2)` 映射，差异来自 `upsample_f32o_fp16` 的 fp16 输入量化，
不是边缘或半像素坐标错误。fp16 存储输入不受影响；若未来需要保持 fp32 输入精度，需
单独设计并验证真正 fp32 计算 shader。此前 bilinear 探针误用了 nearest 模式，旧结论
仅代表当时证据不足，不再作为当前状态。

T17b bilinear 边界坐标修复已完成并经 Codex 与 ZCode 双方复测通过：
`upsample.comp` 的 FP16/FP32 bilinear 分支在计算权重前 clamp `fx/fy`，并重新生成
`upsample_fp16.spv`、`upsample_fp32.spv`、`upsample_f32o_fp16.spv` 和
`upsample_f32o_fp32.spv`。`align_corners=False` native 探针输出相对 PyTorch 参考
`max_abs=0`（修复前 `75.75`），角点不再错误采样 `in[1,1]`；FP16 内部量化相对
FP16 参考 `max_abs=0`、相对 FP32 参考 `0.0001104`，连续 5 次调用稳定，nearest
路径逐位不变。resident nearest/bilinear 分别为 `0`/`0.000558`，4x resident
超分链 `max_abs=0.000033` 且单次提交、激活上传为 `0`；相关 UMA native 回归
全部 PASS。bilinear 模式现为已修复可用，仍受既有能力检测、FP16 精度和 fallback
约束；本轮未构建 wheel、未启动真实独显性能测试。

2026-08-30 resident 异步 host-read 边界修复已完成并通过复验：FP32/FP16 resident
下载在映射前等待 `max(output.ready_value, ComputeManager.last_submitted())`，因此先提交的
图在后提交图仍 in-flight 时也不会暴露 Intel UMA 的部分写入；图提交本身保持异步，等待只
发生在显式 CPU 下载边界。当前正式 standalone DLL `vulkan_ssbo.dll` SHA-256 为
`ADA41A675D78EEB5C2A27B0B26C0315B85C40FB847DF0CF880473EBFF0A7DBF7`。
`test_resident_async_pipeline_runtime.py` 双图回归通过（两次提交无 host wait，最大误差
`0.000217/0.000147`）；新增 `test_resident_host_read_boundary_runtime.py` 连续 100 轮通过，
两路最大误差均为 `6.7055e-08`，最终 `in_flight=0/resident_entries=0`。使用该 DLL 的完整
 UMA 门禁于 17:32:17 开始、17:43:35 结束，`14 OK, 0 FAIL`，最终唯一 `RESULT: PASS`；
本轮仍未构建 wheel、未启动真实独显性能测试，离散 GPU 仍只使用策略模拟证据。

2026-08-30 离散策略模拟最终复验已完成：`run_discrete_simulation_test.bat` 于
17:48:14 启动、17:58:43 完成，正式日志 `D:\PyTorch for Vulkan\pytorch-src\discrete_simulation_test.log`
共 `67` 个 `[run]`、`69` 条 `[exit]`，非零退出数为 `0`，且只有一个最终
`RESULT: PASS`。本轮覆盖 native SSBO staging、预算/async/timeline/fallback 语义以及
resident graph、4x 超分和音频/视频结构级链路；该结果仅证明离散策略在 UMA 宿主上的
兼容性模拟，不构成真实独显性能结论。日志中的 Vulkan loader 可选 layer JSON 缺失提示
不影响用例结果。本轮未构建 wheel、未启动真实独显性能测试。

2026-08-30 SSBO/stock Vulkan bridge 观测契约已补齐：新增 schema 2 的
`torch.ops.vulkan_ssbo.bridge_stats()`/`bridge_reset()`，记录 host lease 命中/失配、
SSBO->CPU materialize、CPU->stock Vulkan materialize 以及 stock Vulkan `copy_` 调用中
来自 SSBO producer 的次数和字节数，并单独记录 stock Vulkan->CPU 回退次数/字节数。
`direct_copy_eligible/enabled` 当前明确为 `0`，
因为两条 runtime 仍由不同 `VkDevice` 所有，不能仅凭 vendor/device 属性共享 `VkBuffer`。
新增 `test_vulkan_ssbo_bridge_runtime.py`：旧运行时缺少接口时清晰 `SKIP`，新运行时
分别以 `VKSSBO_HOST_LEASE=1/0` 检查 lease 复用、CPU 回退和结果一致性；当前结果
`lease_hits=1`、回退计数 `1/2`、`ssbo_to_cpu=2`、`stock_copy_from_ssbo=1`，并已接入
UMA 与离散策略模拟入口。增量 `torch_cpu.dll` SHA-256 为
`AFA4B58A398D6ECE2C0067670A9704E0620D9AB595F818964CAF58C1ED21087C`，已同步源码和
`tools/Python313` 运行时。本轮未构建 wheel、未启动真实独显性能测试。

同日新增 `bench_vulkan_ssbo_bridge.py` 与
`test_vulkan_ssbo_bridge_benchmark.py`，对固定 `conv2d -> SiLU -> conv2d` 链路进行
进程隔离的 `VKSSBO_HOST_LEASE=1/0` A/B。Intel Arc UMA 两次独立诊断的 median 分别为
`1.631/2.399 ms` 与 `3.237/2.056 ms`，说明短样本时延受系统噪声影响，当前不能宣称
lease 带来稳定速度收益；stock Vulkan->CPU 回退计数稳定为 `15/21`，输出 checksum/std
和 finite 检查一致。这只说明 host lease 减少回退，不能外推为真实独显性能收益。基准已
接入 UMA 与离散策略模拟入口。

运行时版本矩阵（2026-08-30）：源码集成运行时 `pytorch-src/torch/lib/torch_cpu.dll`、
构建产物 `build/bin/torch_cpu.dll` 和实际门禁运行时 `tools/Python313/.../torch_cpu.dll`
保持同一 SHA-256 `8BE92FB682ADF03702A980EDD0A2038EC1918B9E29D4BEAEBE16D4E810A59E4C`；
ComfyUI 生产目录 `ComfyUI_windows_portable/python_embeded` 保持旧版
`torch_cpu.dll`（SHA-256 `AFF72D202A5268791C61C1A6B5861C8967BDB7FDD05980A12561637E9EA8AC02`，
PyTorch `gitcf30153`），因为当前 custom node 执行的是 standalone `vulkan_ssbo.dll`
（两处 SHA-256 `D8CA66EE9145A02C9B51393B450E6948F285AC6C434755EFD5CCAF2EC203AB94`）。
该隔离避免只替换单个 `torch_cpu.dll` 导致 `torch_python`/扩展组件混用；若将来切换到
集成 dispatcher，必须在所有 ComfyUI 进程退出后整体配套升级并重新跑门禁。

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

当前执行 **阶段 G（全链路验证和发布准备）**：

1. 保留现有改动并创建局部备份。
2. 保持 resident mirror cache 的 LRU、版本检查和 fallback；不把 SSBO `VkBuffer` 伪装成 stock Vulkan storage。
3. 已完成四条代表性链路的 resident 命中和 A/B 正确性覆盖，持续作为回归测试运行。
4. MatMul 的设备/驱动/拓扑/shape keyed autotune 已完成；保持 3 轮交错 median、3% 晋升门槛、内存预算门控和原子持久化缓存作为后续算子的公共设计基线。
5. MatMul 与 Conv2D 的设备感知自动调优已完成；后续自动调优必须复用独立状态、文件锁、原子持久化、3% 晋升门槛和 budget 门控，新增候选前先做多设备能力边界与数值 A/B。
6. 每次增量构建 `torch_cpu.dll` 后同步两个实际测试运行时；持续运行 native regression、`test_torch.py` 14/14、resident 链和各 autotune focused/concurrent 回归。
7. 已完成 MatMul+Scale+Softmax 显式融合及 A/B 验收；阶段 D 的能力探测、安全门控、fence、timeline 骨架以及 UMA/离散模拟入口验收完成。阶段 F 的 FP16 resident、BF16 FP32 fallback、INT8、INT4、FP8、FP4 格式契约均已完成并通过 UMA/离散策略模拟回归；保持显式误差门槛、fallback 和独立基准，不实现 SSD backing、不实现投机采样、不构建 wheel。
8. 真实独显性能暂缓。后续一次性实测时再显式设置 `VKSSBO_ENABLE_REAL_DGPU_TEST=1` 运行 `run_discrete_gpu_test.bat`；软件模拟结果只用于兼容性，不作为 dGPU 性能结论。
9. FP16 的 view、in-place、fallback 和 resident 生命周期边界已通过；BF16 能力探测已完成并确认走 FP32 fallback，native BF16 storage 保持关闭。INT8 weight-only 首版数据格式、scale/zero-point 语义、安全 fallback 契约、`shaderInt8` 和 8-bit storage 能力探测已通过回归测试锁定；最小 `int8_t` SSBO shader 只完成编译探针，INT8 feature、storage、pipeline 和 kernel 仍未启用，默认仍走 FP32 fallback。INT4 v1 格式契约和 FP32 fallback 已通过 UMA/离散模拟回归，但 INT4 Vulkan feature、storage、shader、pipeline、resident 和性能路径仍未启用。FP8 E4M3FN/E5M2 v1 格式契约和 FP32 fallback 已通过 UMA/离散模拟回归。FP4 E2M1 v1 格式契约和 FP32 fallback 已通过 UMA/离散模拟回归；FP8/FP4 的 Vulkan feature、storage、shader、pipeline、resident 和性能路径仍未启用。

10. 阶段 G 先完成整合包清单、哈希和复核；真实独显一次性性能脚本只在用户提供 NVIDIA/离散设备后执行。交付前不得把本机模拟耗时、UMA 性能或单次样本写成真实 dGPU 性能结论。

11. 音频常见 Conv1D/归一化/GLU/SnakeBeta/ConvTranspose1D、受限 chunk/mul 和 stable Wav2Vec2 Transformer resident 结构级链路已完成并纳入 UMA/离散模拟门禁；在取得真实预训练音频 encoder 权重并完成固定输入的语义输出对比前，只能宣称结构级兼容性完成，不能宣称完整真实音频模型端到端验收完成。

12. 2026-08-30 已完成独显一次性性能测试整合包刷新和静态交付复核。包内强制离散策略模拟日志为 `71` 个 `[run]`、`73` 条 `[exit]`、非零退出 `0`、最终唯一 `RESULT: PASS`；便携包不携带完整 `ComfyUI_windows_portable` 开发树，依赖该开发夹具的 `19` 项 ComfyUI 路由契约测试均明确记录 `[skip]`，不计为失败。`MANIFEST_SHA256.txt` 覆盖 `23,963` 个静态文件，复核结果为 `missing=0、mismatch=0、unlisted=0`，清单文件 SHA-256 为 `AFAD35C2D3FD0E29D30D2E5D20CC14FFE921FFC1F8C5902FEA09FEE56A60100C`，包内 standalone `vulkan_ssbo.dll` SHA-256 为 `ADA41A675D78EEB5C2A27B0B26C0315B85C40FB847DF0CF880473EBFF0A7DBF7`。本次结果只证明 UMA 宿主上的强制离散策略、包布局、运行时依赖和回退契约可用，不是实际独显性能数据；真实独显吞吐、显存占用和 1080P 4x 超分验收仍待目标离散 GPU 执行一次性 runner。

13. 2026-08-30 完整 UMA 门禁首轮仅因 `test_matmul_autotune_runtime.py` 把“缓存条目非空”误判为“当前 key 命中”而失败；当本地缓存含旧驱动条目时，测试实际发生的是合法 miss + tune。测试已改为每个进程使用独立临时缓存，并显式验证首次调优、第二次命中和 override 不改统计。修复后的 UMA 正式入口（21:48:04--22:00:51）共 `85` 个 `[run]`、`87` 条 `[exit]`、非零退出 `0`、唯一 `RESULT: PASS`；随后离散策略模拟入口（22:01:38--22:13:07）共 `69` 个 `[run]`、`71` 条 `[exit]`、非零退出 `0`、唯一 `RESULT: PASS`。后者仍只证明 UMA 宿主上的强制离散策略和 staging/fallback 兼容性，不构成真实独显性能结论。

14. 2026-09-05 收到目标 RTX 4060 Laptop（8GB）的 2026-09-01 一次性测试日志：`71` 个运行项中仅 MatMul autotune 非零退出，最终 `RESULT: FAIL`。报错为旧测试第 43 行缓存命中断言，随后退出码为 `0xC0000005`；不能据此宣称底层 native 退出崩溃已修复。E 盘交付包中该脚本确实仍与旧版一致，已同步并进一步改为独立临时目录、清除继承的 kernel override、显式验证首次 miss/tune 与第二次 hit、finally 清理缓存及恢复环境。CPU-only fake-Torch 接缝可在旧包脚本上精确复现原断言，新版 `3` 个回归通过；不作为 GPU 内核或 native teardown 的验证证据。

15. 同轮修正真实独显 runner 的 NVIDIA 自动识别：CMD/FOR 会拆分未加引号的 `=` 和 `,` 参数，原脚本因而把 `nvidia-smi` 报错写入了设备名。改由纯标准库 Python 使用参数数组查询，失败、不安全输出或多卡歧义不自动选择；保留手动名称/索引，策略模拟跳过自动检测。开发和包内 BAT 的 host-only 选择回归均通过（`5` 个测试，含多个子用例）。新增静态 SHA-256 清单验证器及 `13` 个 CPU-only 回归，预检和完整 runner 都强制验证包完整性，漏复制、损坏或额外静态文件会返回失败。包脚本修订号为 `2026-09-05-host-fixes`。本轮不替换 DLL/runtime、不构建 wheel、不运行真实 GPU 或整套模拟门禁；局部备份位于 `D:\PyTorch for Vulkan\backups\dgpu-package-fix-20260905`，自动调优开发脚本备份位于 `D:\PyTorch for Vulkan\backups\autotune-harness-20260905-210253571`。

16. 该真实日志的约 10 分 40 秒是套件总耗时，不是完整出图耗时。`32x32` 的 4x 链和 `--skip-real` 的 RRDBNet 不能替代 Flash 1080P 加 4x 验收；还缺完整工作流画质、耗时、显存峰值和 XPU 同配置对照。日志中标记 async 的小型 transfer 基准两侧均 `transfer_submissions=0`，standalone resident 基准两侧均 `async_transfer_enabled=0`，故未证明真实异步重叠收益。新的 runner 标记 `async requested` 并明确输出验收范围；接下来先在目标机复核新版失败项与退出码，再另行进行实际启用异步路径及完整工作流验收。

17. 同轮按用户要求新增包根目录 `START_DGPU_TEST.bat`：双击后顺序执行整包/环境预检与真实独显 runner，预检非零立即停止；清除继承的离散模拟变量，保留阶段退出码与日志，默认结束后 pause。`test_oneclick_dgpu.py` 使用两个 BAT 桩而不调用 GPU，验证成功、预检失败短路和 runner 失败保留，`3` 个测试通过。此一键入口只编排现有独显基础回归，不包含 Flash 完整工作流；本轮没有在真实包上启动该入口。最终包内四组 host-only 回归共 `24` 个测试通过，实际 `preflight_environment.bat` 返回 `PASS`；刷新后的清单验证 `verified=23970、missing=0、mismatch=0、unlisted=0`，清单 SHA-256 为 `97B46ACC6040C7D19C1A64732B0B77E758D8D3BBB28F1B09E40E4FD26402BDB6`。包内 standalone DLL 和集成 `torch_cpu.dll` 哈希仍分别为 `ADA41A675D78EEB5C2A27B0B26C0315B85C40FB847DF0CF880473EBFF0A7DBF7`、`8BE92FB682ADF03702A980EDD0A2038EC1918B9E29D4BEAEBE16D4E810A59E4C`，未发生运行时替换。

完成当前阶段验收项后，更新本文件的“当前状态”和数据，再开始下一阶段。

## 6. 明确不直接采用的方案

- 不把 UMA 等同于普通 CPU 指针可被 shader 直接访问。
- 不假设 `DEVICE_LOCAL` 一定 `HOST_VISIBLE`。
- 不假设独显只有两个简单内存池。
- 不因为支持双缓冲就宣称有异步收益。
- 不把 SSD 当作默认显存扩展。
- 不把 FP4、2-bit 或 cooperative matrix 当作所有 Vulkan GPU 都具备的硬件能力。
- 不把投机采样作为 SDXL 优化；它只在适合的自回归音频/文本模型阶段单独评估。
