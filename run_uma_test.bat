@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul

rem ============================================================================
rem PyTorch for Vulkan - UMA compatibility and performance runner
rem
rem This entry point intentionally selects the UMA device and never enables the
rem discrete-GPU simulation or real discrete-GPU performance benchmarks.
rem Override VKSSBO_UMA_DEVICE_NAME only when the target machine exposes a
rem different UMA Vulkan name.
rem ============================================================================

set "ROOT=%~dp0"
set "FINAL_LOG=%ROOT%uma_test.log"
set "RUN_TOKEN=%RANDOM%-%RANDOM%"
set "LOG=%ROOT%uma_test.%RUN_TOKEN%.running.log"
set "RUN_LOCK=%ROOT%.run_uma_test.lock"
2>nul mkdir "%RUN_LOCK%"
if errorlevel 1 (
    echo ERROR: another UMA compatibility/performance test is already running.
    echo Lock: %RUN_LOCK%
    exit /b 2
)
set "RUN_LOCK_HELD=1"

if not defined VKSSBO_PYTHON (
    if exist "%ROOT%..\..\vk-test-env\Scripts\python.exe" (
        set "VKSSBO_PYTHON=%ROOT%..\..\vk-test-env\Scripts\python.exe"
    ) else if exist "%ROOT%..\..\ComfyUI Vulkan beat\tools\Python313\python.exe" (
        set "VKSSBO_PYTHON=%ROOT%..\..\ComfyUI Vulkan beat\tools\Python313\python.exe"
    ) else if exist "%ROOT%..\..\ComfyUI Vulkan beat\ComfyUI_windows_portable\python_embeded\python.exe" (
        set "VKSSBO_PYTHON=%ROOT%..\..\ComfyUI Vulkan beat\ComfyUI_windows_portable\python_embeded\python.exe"
    ) else (
        set "VKSSBO_PYTHON=python"
    )
)
if not defined VKSSBO_TORCH_PYTHON set "VKSSBO_TORCH_PYTHON=%VKSSBO_PYTHON%"
if not defined VKSSBO_TEST_ROOT (
    if exist "%ROOT%..\..\ComfyUI Vulkan beat\tools\vulkan_ssbo\test_torch.py" (
        set "VKSSBO_TEST_ROOT=%ROOT%..\..\ComfyUI Vulkan beat\tools\vulkan_ssbo"
    ) else (
        set "VKSSBO_TEST_ROOT=%ROOT%tools\vulkan_ssbo"
    )
)

if defined VKSSBO_UMA_DEVICE_NAME (
    set "VKSSBO_DEVICE_NAME=%VKSSBO_UMA_DEVICE_NAME%"
) else (
    set "VKSSBO_DEVICE_NAME=Intel(R) Arc(TM) Graphics"
)
set "VKSSBO_DEVICE_INDEX="
if defined VKSSBO_UMA_DEVICE_INDEX set "VKSSBO_DEVICE_INDEX=%VKSSBO_UMA_DEVICE_INDEX%"
set "VKSSBO_SIMULATE_DISCRETE="
set "VKSSBO_SIMULATED_VRAM_MB="
set "VKSSBO_SIMULATION_FORCE_STAGING="
set "VKSSBO_SIMULATION_STRICT_BUDGET="
set "VKSSBO_SIMULATION_TRANSFER_LATENCY_US="
set "VKSSBO_ASYNC_TRANSFER=0"
if not defined VKSSBO_BENCH_SHAPE set "VKSSBO_BENCH_SHAPE=64,128,64"
if not defined VKSSBO_BENCH_WARMUP set "VKSSBO_BENCH_WARMUP=3"
if not defined VKSSBO_BENCH_REPEATS set "VKSSBO_BENCH_REPEATS=9"
if not defined VKSSBO_RESIDENT_DEPTH set "VKSSBO_RESIDENT_DEPTH=8"
if not defined VKSSBO_RESIDENT_WARMUP set "VKSSBO_RESIDENT_WARMUP=2"
if not defined VKSSBO_RESIDENT_REPEATS set "VKSSBO_RESIDENT_REPEATS=7"
set "FAILED=0"

> "%LOG%" echo ============================================================
>> "%LOG%" echo PyTorch for Vulkan UMA test
>> "%LOG%" echo Started: %DATE% %TIME%
>> "%LOG%" echo Host: %COMPUTERNAME%
>> "%LOG%" echo User: %USERNAME%
>> "%LOG%" echo Root: %ROOT%
>> "%LOG%" echo Python: %VKSSBO_PYTHON%
>> "%LOG%" echo TestRoot: %VKSSBO_TEST_ROOT%
>> "%LOG%" echo VKSSBO_DEVICE_INDEX=%VKSSBO_DEVICE_INDEX%
>> "%LOG%" echo VKSSBO_DEVICE_NAME=%VKSSBO_DEVICE_NAME%
>> "%LOG%" echo VKSSBO_TORCH_PYTHON=%VKSSBO_TORCH_PYTHON%
>> "%LOG%" echo VKSSBO_ASYNC_TRANSFER=%VKSSBO_ASYNC_TRANSFER%
>> "%LOG%" echo VKSSBO_BENCH_SHAPE=%VKSSBO_BENCH_SHAPE%
>> "%LOG%" echo VKSSBO_BENCH_WARMUP=%VKSSBO_BENCH_WARMUP%
>> "%LOG%" echo VKSSBO_BENCH_REPEATS=%VKSSBO_BENCH_REPEATS%
>> "%LOG%" echo VKSSBO_RESIDENT_DEPTH=%VKSSBO_RESIDENT_DEPTH%
>> "%LOG%" echo VKSSBO_RESIDENT_WARMUP=%VKSSBO_RESIDENT_WARMUP%
>> "%LOG%" echo VKSSBO_RESIDENT_REPEATS=%VKSSBO_RESIDENT_REPEATS%
>> "%LOG%" echo ============================================================

call :run_optional "Python version" "%VKSSBO_PYTHON%" --version
>> "%LOG%" echo.
>> "%LOG%" echo [preflight] PyTorch import
"%VKSSBO_PYTHON%" -c "import torch; print(torch.__file__); print(torch.__version__)" >> "%LOG%" 2>&1
set "TORCH_IMPORT_RC=%ERRORLEVEL%"
>> "%LOG%" echo [exit] PyTorch import rc=!TORCH_IMPORT_RC!
if not "!TORCH_IMPORT_RC!"=="0" (
    >> "%LOG%" echo ERROR: selected Python cannot import PyTorch.
    set "FAILED=1"
    goto :finish
)

>> "%LOG%" echo.
>> "%LOG%" echo [environment] Vulkan summary
where vulkaninfo.exe >> "%LOG%" 2>&1
if errorlevel 1 (>> "%LOG%" echo vulkaninfo.exe not found on PATH) else (vulkaninfo.exe --summary >> "%LOG%" 2>&1)

if not exist "%VKSSBO_TEST_ROOT%\test_device_selection_runtime.py" (
    >> "%LOG%" echo ERROR: test root does not contain Vulkan SSBO tests
    set "FAILED=1"
    goto :finish
)

call :run_required "device selection" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_device_selection_runtime.py"
call :run_required "async capabilities" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_async_transfer_capabilities_runtime.py"
set "VKSSBO_TRACE_SUBMIT=1"
call :run_required "fence submit" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_fence_submit_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_scheduler_lifecycle_torch_runtime.py" call :run_required "PyTorch scheduler lifecycle" "%VKSSBO_TORCH_PYTHON%" "%VKSSBO_TEST_ROOT%\test_scheduler_lifecycle_torch_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_matmul_softmax_fallback_trace_runtime.py" call :run_required "MatMul Softmax fallback trace" "%VKSSBO_TORCH_PYTHON%" "%VKSSBO_TEST_ROOT%\test_matmul_softmax_fallback_trace_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_scheduler_lifecycle_runtime.py" call :run_required "standalone scheduler lifecycle" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_scheduler_lifecycle_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_profile_route_contract.py" call :run_required "ComfyUI profile route contract" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_profile_route_contract.py"
if exist "%VKSSBO_TEST_ROOT%\test_vulkan_ssbo_bridge_runtime.py" call :run_required "SSBO stock-Vulkan bridge contract" "%VKSSBO_TORCH_PYTHON%" "%VKSSBO_TEST_ROOT%\test_vulkan_ssbo_bridge_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_vulkan_ssbo_bridge_benchmark.py" call :run_required "SSBO stock-Vulkan bridge benchmark" "%VKSSBO_TORCH_PYTHON%" "%VKSSBO_TEST_ROOT%\test_vulkan_ssbo_bridge_benchmark.py"
if exist "%VKSSBO_TEST_ROOT%\test_standalone_device_info_runtime.py" call :run_required "standalone device info contract" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_standalone_device_info_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_operator_profile_contract.py" call :run_required "standalone operator profile contract" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_operator_profile_contract.py"
set "VKSSBO_TRACE_SUBMIT=0"
 call :run_required "native SSBO regression" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_vulkan_ssbo_native_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_attention_score_precision_runtime.py" call :run_required "attention score precision" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_attention_score_precision_runtime.py"
call :run_required "14-test torch regression" "%VKSSBO_TORCH_PYTHON%" "%VKSSBO_TEST_ROOT%\test_torch.py"
if exist "%VKSSBO_TEST_ROOT%\test_fp16_resident_runtime.py" call :run_required "FP16 resident compatibility" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_fp16_resident_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_fp16_boundaries_runtime.py" call :run_required "FP16 boundary compatibility" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_fp16_boundaries_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_resident_fp32_boundaries_runtime.py" call :run_required "FP32 resident boundary compatibility" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_resident_fp32_boundaries_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_resident_fp32_conv2d_runtime.py" call :run_required "FP32 resident Conv2D compatibility" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_resident_fp32_conv2d_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_bf16_fallback_runtime.py" call :run_required "BF16 fallback compatibility" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_bf16_fallback_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_int8_capability_runtime.py" call :run_required "INT8 shader capability" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_int8_capability_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_int8_storage_probe_runtime.py" call :run_required "INT8 SSBO shader probe" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_int8_storage_probe_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_int8_weight_only_contract_runtime.py" call :run_required "INT8 weight-only contract" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_int8_weight_only_contract_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_int4_weight_only_contract_runtime.py" call :run_required "INT4 weight-only contract" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_int4_weight_only_contract_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_fp8_weight_only_contract_runtime.py" call :run_required "FP8 weight-only contract" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_fp8_weight_only_contract_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_fp4_weight_only_contract_runtime.py" call :run_required "FP4 weight-only contract" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_fp4_weight_only_contract_runtime.py"

rem Compatibility coverage for the memory, resident, autotune and fusion paths.
if exist "%VKSSBO_TEST_ROOT%\test_memory_budget.py" call :run_required "memory budget" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_memory_budget.py"
set "VKSSBO_TRANSIENT_POOL=1"
if exist "%VKSSBO_TEST_ROOT%\test_transient_pool.py" call :run_required "transient pool" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_transient_pool.py"
set "VKSSBO_TRANSIENT_POOL="
if exist "%VKSSBO_TEST_ROOT%\test_resident.py" call :run_required "resident regression" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_resident.py"
if exist "%VKSSBO_TEST_ROOT%\test_resident_linear_runtime.py" call :run_required "resident linear compatibility" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_resident_linear_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_resident_graph_linear_runtime.py" call :run_required "resident graph linear compatibility" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_resident_graph_linear_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_comfy_resident_mlp_graph_runtime.py" call :run_required "ComfyUI resident MLP graph" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_comfy_resident_mlp_graph_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_resident_graph_geglu_runtime.py" call :run_required "resident graph GEGLU compatibility" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_resident_graph_geglu_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_comfy_resident_geglu_graph_runtime.py" call :run_required "ComfyUI resident GEGLU graph" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_comfy_resident_geglu_graph_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_resident_graph_layernorm_runtime.py" call :run_required "resident graph LayerNorm compatibility" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_resident_graph_layernorm_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_comfy_resident_layernorm_graph_runtime.py" call :run_required "ComfyUI resident LayerNorm graph" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_comfy_resident_layernorm_graph_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_resident_graph_groupnorm_runtime.py" call :run_required "resident graph GroupNorm compatibility" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_resident_graph_groupnorm_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_comfy_resident_groupnorm_graph_runtime.py" call :run_required "ComfyUI resident GroupNorm graph" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_comfy_resident_groupnorm_graph_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_resident_graph_conv2d_runtime.py" call :run_required "resident graph Conv2D compatibility" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_resident_graph_conv2d_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_resident_graph_channelwise_add_runtime.py" call :run_required "resident graph channelwise add compatibility" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_resident_graph_channelwise_add_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_resident_graph_upsample_runtime.py" call :run_required "resident graph upsample compatibility" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_resident_graph_upsample_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_resident_4x_superres_runtime.py" call :run_required "resident 4x super-resolution chain" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_resident_4x_superres_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_rrdbnet_4x_e2e_runtime.py" call :run_required "RRDBNet 4x end-to-end compatibility" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_rrdbnet_4x_e2e_runtime.py" --skip-real
if exist "%VKSSBO_TEST_ROOT%\test_comfy_resident_conv2d_graph_runtime.py" call :run_required "ComfyUI resident Conv2D graph" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_comfy_resident_conv2d_graph_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_resident_graph_conv1d_runtime.py" call :run_required "resident graph Conv1D compatibility" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_resident_graph_conv1d_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_comfy_resident_conv1d_runtime.py" call :run_required "ComfyUI resident Conv1D routing" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_comfy_resident_conv1d_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_resident_graph_conv_transpose1d_runtime.py" call :run_required "resident graph ConvTranspose1D compatibility" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_resident_graph_conv_transpose1d_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_comfy_resident_conv_transpose1d_runtime.py" call :run_required "ComfyUI resident ConvTranspose1D routing" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_comfy_resident_conv_transpose1d_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_resident_graph_conv1d_activation_runtime.py" call :run_required "resident graph Conv1D activation compatibility" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_resident_graph_conv1d_activation_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_comfy_resident_conv1d_activation_runtime.py" call :run_required "ComfyUI audio Conv1D activation routing" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_comfy_resident_conv1d_activation_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_resident_graph_audio_norm_conv1d_runtime.py" call :run_required "resident audio Conv1D normalization graph" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_resident_graph_audio_norm_conv1d_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_comfy_resident_audio_norm_conv1d_runtime.py" call :run_required "ComfyUI audio normalization Conv1D routing" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_comfy_resident_audio_norm_conv1d_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_resident_graph_glu_runtime.py" call :run_required "resident graph GLU compatibility" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_resident_graph_glu_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_resident_graph_chunk_mul_runtime.py" call :run_required "resident graph chunk/mul compatibility" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_resident_graph_chunk_mul_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_resident_graph_audio_transformer_runtime.py" call :run_required "resident audio Transformer graph compatibility" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_resident_graph_audio_transformer_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_comfy_resident_audio_transformer_runtime.py" call :run_required "ComfyUI audio Transformer routing" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_comfy_resident_audio_transformer_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_resident_audio_encoder_mini_e2e_runtime.py" call :run_required "mini Wav2Vec2 resident encoder smoke" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_resident_audio_encoder_mini_e2e_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_comfy_resident_audio_glu_runtime.py" call :run_required "ComfyUI audio GLU routing" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_comfy_resident_audio_glu_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_comfy_resident_glu_gate_runtime.py" call :run_required "ComfyUI functional GLU gate routing" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_comfy_resident_glu_gate_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_resident_graph_snake_beta_runtime.py" call :run_required "resident graph SnakeBeta compatibility" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_resident_graph_snake_beta_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_comfy_resident_conv3d_runtime.py" call :run_required "ComfyUI resident Conv3D routing" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_comfy_resident_conv3d_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_comfy_resident_resblock_graph_runtime.py" call :run_required "ComfyUI resident full ResBlock graph" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_comfy_resident_resblock_graph_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_comfy_resident_resblock_scale_shift_runtime.py" call :run_required "ComfyUI resident scale-shift ResBlock graph" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_comfy_resident_resblock_scale_shift_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_comfy_resident_resblock_updown_runtime.py" call :run_required "ComfyUI resident up/down ResBlock graph" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_comfy_resident_resblock_updown_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_comfy_resident_video_resblock_runtime.py" call :run_required "ComfyUI resident VideoResBlock graph" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_comfy_resident_video_resblock_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_comfy_resident_lora_runtime.py" call :run_required "ComfyUI resident standard LoRA graph" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_comfy_resident_lora_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_resident_async_runtime.py" call :run_required "resident async graph compatibility" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_resident_async_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_resident_async_pipeline_runtime.py" call :run_required "resident async multi-graph pipeline" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_resident_async_pipeline_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_resident_host_read_boundary_runtime.py" call :run_required "resident host-read boundary" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_resident_host_read_boundary_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_matmul_autotune_runtime.py" call :run_required "matmul autotune" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_matmul_autotune_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_matmul_autotune_focused_runtime.py" call :run_required "matmul autotune focused" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_matmul_autotune_focused_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_matmul_autotune_concurrent_runtime.py" call :run_required "matmul autotune concurrent" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_matmul_autotune_concurrent_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_conv2d_autotune_focused_runtime.py" call :run_required "conv2d autotune focused" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_conv2d_autotune_focused_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_conv2d_autotune_concurrent_runtime.py" call :run_required "conv2d autotune concurrent" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_conv2d_autotune_concurrent_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_producer_metadata_runtime.py" call :run_required "producer metadata" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_producer_metadata_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_conv2d_silu_fusion_runtime.py" call :run_required "conv2d silu fusion" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_conv2d_silu_fusion_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_linear_silu_fusion_runtime.py" call :run_required "linear silu fusion" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_linear_silu_fusion_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_linear_gelu_fusion_runtime.py" call :run_required "linear gelu fusion" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_linear_gelu_fusion_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_matmul_softmax_fusion_runtime.py" call :run_required "matmul softmax fusion" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_matmul_softmax_fusion_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_resident_benchmark_contract.py" call :run_required "resident benchmark contract" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_resident_benchmark_contract.py"

rem UMA performance: resident graph plus the existing fixed-shape benchmarks.
set "VKSSBO_ASYNC_TRANSFER=0"
if exist "%VKSSBO_TEST_ROOT%\bench_resident_json.py" call :run_resident_benchmark "resident graph benchmark"
if exist "%VKSSBO_TEST_ROOT%\bench_cached.py" call :run_benchmark "cached operator benchmark" "%VKSSBO_TEST_ROOT%\bench_cached.py"
set "VKSSBO_ACT_FP16=1"
if exist "%VKSSBO_TEST_ROOT%\bench_cached.py" call :run_benchmark "FP16 cached operator benchmark" "%VKSSBO_TEST_ROOT%\bench_cached.py"
set "VKSSBO_ACT_FP16="
if exist "%VKSSBO_TEST_ROOT%\bench_vae.py" call :run_benchmark "VAE convolution benchmark" "%VKSSBO_TEST_ROOT%\bench_vae.py" --runs 5 --warmup 2
if exist "%VKSSBO_TEST_ROOT%\bench_fixed.py" call :run_benchmark "fixed overhead benchmark" "%VKSSBO_TEST_ROOT%\bench_fixed.py"
if exist "%VKSSBO_TEST_ROOT%\bench_operator_profile.py" call :run_profile "operator profile matrix" "%VKSSBO_TEST_ROOT%\bench_operator_profile.py" --profile-set representative --warmup 1 --repeats 3
set "VKSSBO_ACT_FP16=1"
if exist "%VKSSBO_TEST_ROOT%\bench_operator_profile.py" call :run_profile "FP16 operator profile matrix" "%VKSSBO_TEST_ROOT%\bench_operator_profile.py" --profile-set representative --warmup 1 --repeats 3
set "VKSSBO_ACT_FP16="

:finish
>> "%LOG%" echo.
>> "%LOG%" echo ============================================================
>> "%LOG%" echo Finished: %DATE% %TIME%
if "%FAILED%"=="0" (
    >> "%LOG%" echo RESULT: PASS
    echo UMA test completed: PASS
) else (
    >> "%LOG%" echo RESULT: FAIL
    echo UMA test completed: FAIL
)
move /y "%LOG%" "%FINAL_LOG%" >nul
if errorlevel 1 (
    if defined RUN_LOCK_HELD rmdir "%RUN_LOCK%" >nul 2>&1
    echo ERROR: failed to promote the completed UMA log.
    echo Temporary log: %LOG%
    exit /b 3
)
if defined RUN_LOCK_HELD rmdir "%RUN_LOCK%" >nul 2>&1
echo Log: %FINAL_LOG%
exit /b %FAILED%

:run_required
set "LABEL=%~1"
set "COMMAND=%~2"
set "ARG1=%~3"
set "ARG2=%~4"
>> "%LOG%" echo.
>> "%LOG%" echo [run] %LABEL%
>> "%LOG%" echo Command: "%COMMAND%" "%ARG1%" "%ARG2%"
if "%ARG1%"=="" ("%COMMAND%" >> "%LOG%" 2>&1) else if "%ARG2%"=="" ("%COMMAND%" "%ARG1%" >> "%LOG%" 2>&1) else ("%COMMAND%" "%ARG1%" "%ARG2%" >> "%LOG%" 2>&1)
set "RC=%ERRORLEVEL%"
>> "%LOG%" echo [exit] %LABEL% rc=!RC!
if not "!RC!"=="0" set "FAILED=1"
exit /b 0

:run_optional
set "LABEL=%~1"
set "COMMAND=%~2"
set "ARG1=%~3"
>> "%LOG%" echo.
>> "%LOG%" echo [optional] %LABEL%
"%COMMAND%" "%ARG1%" >> "%LOG%" 2>&1
>> "%LOG%" echo [exit] %LABEL% rc=!ERRORLEVEL!
exit /b 0

:run_script
set "LABEL=%~1"
set "SCRIPT=%~2"
set "ARG1=%~3"
set "ARG2=%~4"
>> "%LOG%" echo.
>> "%LOG%" echo [run] %LABEL%
>> "%LOG%" echo Command: "%VKSSBO_TORCH_PYTHON%" "%SCRIPT%" "%ARG1%" "%ARG2%"
if "%ARG1%"=="" ("%VKSSBO_TORCH_PYTHON%" "%SCRIPT%" >> "%LOG%" 2>&1) else if "%ARG2%"=="" ("%VKSSBO_TORCH_PYTHON%" "%SCRIPT%" "%ARG1%" >> "%LOG%" 2>&1) else ("%VKSSBO_TORCH_PYTHON%" "%SCRIPT%" "%ARG1%" "%ARG2%" >> "%LOG%" 2>&1)
set "RC=%ERRORLEVEL%"
>> "%LOG%" echo [exit] %LABEL% rc=!RC!
if not "!RC!"=="0" set "FAILED=1"
exit /b 0

:run_benchmark
set "LABEL=%~1"
set "SCRIPT=%~2"
set "ARG1=%~3"
set "ARG2=%~4"
>> "%LOG%" echo.
>> "%LOG%" echo [run] %LABEL% (diagnostic benchmark)
>> "%LOG%" echo Command: "%VKSSBO_TORCH_PYTHON%" "%SCRIPT%" "%ARG1%" "%ARG2%"
if "%ARG1%"=="" ("%VKSSBO_TORCH_PYTHON%" "%SCRIPT%" >> "%LOG%" 2>&1) else if "%ARG2%"=="" ("%VKSSBO_TORCH_PYTHON%" "%SCRIPT%" "%ARG1%" >> "%LOG%" 2>&1) else ("%VKSSBO_TORCH_PYTHON%" "%SCRIPT%" "%ARG1%" "%ARG2%" >> "%LOG%" 2>&1)
set "RC=%ERRORLEVEL%"
>> "%LOG%" echo [exit] %LABEL% rc=!RC! (benchmark result retained; does not gate compatibility)
exit /b 0

:run_profile
set "LABEL=%~1"
set "SCRIPT=%~2"
set "ARG1=%~3"
set "ARG2=%~4"
set "ARG3=%~5"
set "ARG4=%~6"
set "ARG5=%~7"
set "ARG6=%~8"
>> "%LOG%" echo.
>> "%LOG%" echo [run] %LABEL% (diagnostic profile)
>> "%LOG%" echo Command: "%VKSSBO_TORCH_PYTHON%" "%SCRIPT%" "%ARG1%" "%ARG2%" "%ARG3%" "%ARG4%" "%ARG5%" "%ARG6%"
"%VKSSBO_TORCH_PYTHON%" "%SCRIPT%" "%ARG1%" "%ARG2%" "%ARG3%" "%ARG4%" "%ARG5%" "%ARG6%" >> "%LOG%" 2>&1
set "RC=%ERRORLEVEL%"
>> "%LOG%" echo [exit] %LABEL% rc=!RC! (profile result retained; does not gate compatibility)
exit /b 0

:run_resident_benchmark
set "LABEL=%~1"
>> "%LOG%" echo.
>> "%LOG%" echo [run] %LABEL%
>> "%LOG%" echo Command: "%VKSSBO_TORCH_PYTHON%" "%VKSSBO_TEST_ROOT%\bench_resident_json.py" --depth %VKSSBO_RESIDENT_DEPTH% --warmup %VKSSBO_RESIDENT_WARMUP% --repeats %VKSSBO_RESIDENT_REPEATS%
"%VKSSBO_TORCH_PYTHON%" "%VKSSBO_TEST_ROOT%\bench_resident_json.py" --depth %VKSSBO_RESIDENT_DEPTH% --warmup %VKSSBO_RESIDENT_WARMUP% --repeats %VKSSBO_RESIDENT_REPEATS% >> "%LOG%" 2>&1
set "RC=%ERRORLEVEL%"
>> "%LOG%" echo [exit] %LABEL% rc=!RC!
if not "!RC!"=="0" set "FAILED=1"
exit /b 0
