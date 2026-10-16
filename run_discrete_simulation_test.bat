@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul

rem ============================================================================
rem PyTorch for Vulkan - discrete-GPU policy simulation compatibility runner
rem
rem This entry point uses the real UMA Vulkan device only as an execution host,
rem while forcing the discrete policy, staging, budget and timeline branches.
rem It deliberately does not run transfer/resident performance benchmarks.
rem Simulation results are compatibility evidence, never discrete-GPU timing.
rem ============================================================================

set "ROOT=%~dp0"
set "FINAL_LOG=%ROOT%discrete_simulation_test.log"
set "RUN_TOKEN=%RANDOM%-%RANDOM%"
set "LOG=%ROOT%discrete_simulation_test.%RUN_TOKEN%.running.log"
set "RUN_LOCK=%ROOT%.run_discrete_simulation_test.lock"
2>nul mkdir "%RUN_LOCK%"
if errorlevel 1 (
    echo ERROR: another discrete simulation compatibility test is already running.
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
if defined VKSSBO_SIMULATION_DEVICE_NAME (
    set "VKSSBO_DEVICE_NAME=%VKSSBO_SIMULATION_DEVICE_NAME%"
) else (
    set "VKSSBO_DEVICE_NAME=Intel(R) Arc(TM) Graphics"
)
set "VKSSBO_DEVICE_INDEX="
set "VKSSBO_SIMULATE_DISCRETE=1"
if not defined VKSSBO_SIMULATED_VRAM_MB set "VKSSBO_SIMULATED_VRAM_MB=4096"
set "VKSSBO_SIMULATION_FORCE_STAGING=1"
set "VKSSBO_SIMULATION_STRICT_BUDGET=0"
if not defined VKSSBO_SIMULATION_TRANSFER_LATENCY_US set "VKSSBO_SIMULATION_TRANSFER_LATENCY_US=0"
set "VKSSBO_ASYNC_TRANSFER=1"
set "VKSSBO_TRACE_SUBMIT=1"
set "FAILED=0"

> "%LOG%" echo ============================================================
>> "%LOG%" echo PyTorch for Vulkan discrete GPU simulation compatibility test
>> "%LOG%" echo Started: %DATE% %TIME%
>> "%LOG%" echo Host: %COMPUTERNAME%
>> "%LOG%" echo User: %USERNAME%
>> "%LOG%" echo Root: %ROOT%
>> "%LOG%" echo Python: %VKSSBO_PYTHON%
>> "%LOG%" echo TestRoot: %VKSSBO_TEST_ROOT%
>> "%LOG%" echo VKSSBO_DEVICE_NAME=%VKSSBO_DEVICE_NAME%
>> "%LOG%" echo VKSSBO_SIMULATE_DISCRETE=%VKSSBO_SIMULATE_DISCRETE%
>> "%LOG%" echo VKSSBO_SIMULATED_VRAM_MB=%VKSSBO_SIMULATED_VRAM_MB%
>> "%LOG%" echo VKSSBO_SIMULATION_FORCE_STAGING=%VKSSBO_SIMULATION_FORCE_STAGING%
>> "%LOG%" echo VKSSBO_SIMULATION_STRICT_BUDGET=%VKSSBO_SIMULATION_STRICT_BUDGET%
>> "%LOG%" echo VKSSBO_SIMULATION_TRANSFER_LATENCY_US=%VKSSBO_SIMULATION_TRANSFER_LATENCY_US%
>> "%LOG%" echo VKSSBO_ASYNC_TRANSFER=%VKSSBO_ASYNC_TRANSFER%
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

if not exist "%VKSSBO_TEST_ROOT%\test_discrete_simulation_runtime.py" (
    >> "%LOG%" echo ERROR: test root does not contain test_discrete_simulation_runtime.py
    set "FAILED=1"
    goto :finish
)

call :run_required "discrete policy simulation" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_discrete_simulation_runtime.py"
set "VKSSBO_ASYNC_TRANSFER=0"
call :run_required "fence submit under simulation" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_fence_submit_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_scheduler_lifecycle_torch_runtime.py" call :run_required "PyTorch scheduler lifecycle simulation" "%VKSSBO_TORCH_PYTHON%" "%VKSSBO_TEST_ROOT%\test_scheduler_lifecycle_torch_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_matmul_softmax_fallback_trace_runtime.py" call :run_required "MatMul Softmax fallback trace simulation" "%VKSSBO_TORCH_PYTHON%" "%VKSSBO_TEST_ROOT%\test_matmul_softmax_fallback_trace_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_scheduler_lifecycle_runtime.py" call :run_required "standalone scheduler lifecycle simulation" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_scheduler_lifecycle_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_profile_route_contract.py" call :run_required "ComfyUI profile route contract simulation" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_profile_route_contract.py"
if exist "%VKSSBO_TEST_ROOT%\test_vulkan_ssbo_bridge_runtime.py" call :run_required "SSBO stock-Vulkan bridge contract simulation" "%VKSSBO_TORCH_PYTHON%" "%VKSSBO_TEST_ROOT%\test_vulkan_ssbo_bridge_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_vulkan_ssbo_bridge_benchmark.py" call :run_required "SSBO stock-Vulkan bridge benchmark simulation" "%VKSSBO_TORCH_PYTHON%" "%VKSSBO_TEST_ROOT%\test_vulkan_ssbo_bridge_benchmark.py"
if exist "%VKSSBO_TEST_ROOT%\test_standalone_device_info_runtime.py" call :run_required "standalone device info contract simulation" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_standalone_device_info_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_operator_profile_contract.py" call :run_required "standalone operator profile contract simulation" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_operator_profile_contract.py"
set "VKSSBO_ASYNC_TRANSFER=1"
 call :run_required "native SSBO staging simulation" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_vulkan_ssbo_native_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_attention_score_precision_runtime.py" call :run_required "attention score precision simulation" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_attention_score_precision_runtime.py"
set "VKSSBO_TRACE_SUBMIT=0"
if exist "%VKSSBO_TEST_ROOT%\test_fp16_resident_runtime.py" call :run_required "FP16 resident simulation" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_fp16_resident_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_fp16_boundaries_runtime.py" call :run_required "FP16 boundary simulation" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_fp16_boundaries_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_resident_fp32_boundaries_runtime.py" call :run_required "FP32 resident boundary simulation" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_resident_fp32_boundaries_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_resident_fp32_conv2d_runtime.py" call :run_required "FP32 resident Conv2D simulation" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_resident_fp32_conv2d_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_bf16_fallback_runtime.py" call :run_required "BF16 fallback simulation" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_bf16_fallback_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_int8_capability_runtime.py" call :run_required "INT8 shader capability simulation" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_int8_capability_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_int8_storage_probe_runtime.py" call :run_required "INT8 SSBO shader probe simulation" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_int8_storage_probe_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_int8_weight_only_contract_runtime.py" call :run_required "INT8 weight-only simulation" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_int8_weight_only_contract_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_int4_weight_only_contract_runtime.py" call :run_required "INT4 weight-only simulation" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_int4_weight_only_contract_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_fp8_weight_only_contract_runtime.py" call :run_required "FP8 weight-only simulation" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_fp8_weight_only_contract_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_fp4_weight_only_contract_runtime.py" call :run_required "FP4 weight-only simulation" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_fp4_weight_only_contract_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_memory_budget.py" call :run_required "memory budget simulation" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_memory_budget.py"
set "VKSSBO_TRANSIENT_POOL=1"
if exist "%VKSSBO_TEST_ROOT%\test_transient_pool.py" call :run_required "transient pool simulation" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_transient_pool.py"
set "VKSSBO_TRANSIENT_POOL="
if exist "%VKSSBO_TEST_ROOT%\test_resident.py" call :run_required "resident simulation" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_resident.py"
if exist "%VKSSBO_TEST_ROOT%\test_resident_linear_runtime.py" call :run_required "resident linear simulation" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_resident_linear_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_resident_graph_linear_runtime.py" call :run_required "resident graph linear simulation" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_resident_graph_linear_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_comfy_resident_mlp_graph_runtime.py" call :run_required "ComfyUI resident MLP graph simulation" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_comfy_resident_mlp_graph_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_resident_graph_geglu_runtime.py" call :run_required "resident graph GEGLU simulation" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_resident_graph_geglu_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_comfy_resident_geglu_graph_runtime.py" call :run_required "ComfyUI resident GEGLU graph simulation" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_comfy_resident_geglu_graph_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_resident_graph_layernorm_runtime.py" call :run_required "resident graph LayerNorm simulation" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_resident_graph_layernorm_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_comfy_resident_layernorm_graph_runtime.py" call :run_required "ComfyUI resident LayerNorm graph simulation" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_comfy_resident_layernorm_graph_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_resident_graph_groupnorm_runtime.py" call :run_required "resident graph GroupNorm simulation" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_resident_graph_groupnorm_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_comfy_resident_groupnorm_graph_runtime.py" call :run_required "ComfyUI resident GroupNorm graph simulation" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_comfy_resident_groupnorm_graph_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_resident_graph_conv2d_runtime.py" call :run_required "resident graph Conv2D simulation" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_resident_graph_conv2d_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_resident_graph_channelwise_add_runtime.py" call :run_required "resident graph channelwise add simulation" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_resident_graph_channelwise_add_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_resident_graph_upsample_runtime.py" call :run_required "resident graph upsample simulation" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_resident_graph_upsample_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_resident_4x_superres_runtime.py" call :run_required "resident 4x super-resolution simulation" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_resident_4x_superres_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_rrdbnet_4x_e2e_runtime.py" call :run_required "RRDBNet 4x end-to-end simulation" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_rrdbnet_4x_e2e_runtime.py" --skip-real
if exist "%VKSSBO_TEST_ROOT%\test_comfy_resident_conv2d_graph_runtime.py" call :run_required "ComfyUI resident Conv2D graph simulation" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_comfy_resident_conv2d_graph_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_resident_graph_conv1d_runtime.py" call :run_required "resident graph Conv1D simulation" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_resident_graph_conv1d_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_comfy_resident_conv1d_runtime.py" call :run_required "ComfyUI resident Conv1D routing simulation" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_comfy_resident_conv1d_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_resident_graph_conv_transpose1d_runtime.py" call :run_required "resident graph ConvTranspose1D simulation" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_resident_graph_conv_transpose1d_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_comfy_resident_conv_transpose1d_runtime.py" call :run_required "ComfyUI resident ConvTranspose1D routing simulation" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_comfy_resident_conv_transpose1d_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_resident_graph_conv1d_activation_runtime.py" call :run_required "resident graph Conv1D activation simulation" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_resident_graph_conv1d_activation_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_comfy_resident_conv1d_activation_runtime.py" call :run_required "ComfyUI audio Conv1D activation simulation" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_comfy_resident_conv1d_activation_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_resident_graph_audio_norm_conv1d_runtime.py" call :run_required "resident audio Conv1D normalization simulation" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_resident_graph_audio_norm_conv1d_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_comfy_resident_audio_norm_conv1d_runtime.py" call :run_required "ComfyUI audio normalization Conv1D simulation" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_comfy_resident_audio_norm_conv1d_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_resident_graph_glu_runtime.py" call :run_required "resident graph GLU simulation" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_resident_graph_glu_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_resident_graph_chunk_mul_runtime.py" call :run_required "resident graph chunk/mul simulation" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_resident_graph_chunk_mul_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_resident_graph_audio_transformer_runtime.py" call :run_required "resident audio Transformer graph simulation" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_resident_graph_audio_transformer_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_comfy_resident_audio_transformer_runtime.py" call :run_required "ComfyUI audio Transformer simulation" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_comfy_resident_audio_transformer_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_resident_audio_encoder_mini_e2e_runtime.py" call :run_required "mini Wav2Vec2 resident encoder simulation" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_resident_audio_encoder_mini_e2e_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_comfy_resident_audio_glu_runtime.py" call :run_required "ComfyUI audio GLU simulation" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_comfy_resident_audio_glu_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_comfy_resident_glu_gate_runtime.py" call :run_required "ComfyUI functional GLU gate simulation" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_comfy_resident_glu_gate_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_resident_graph_snake_beta_runtime.py" call :run_required "resident graph SnakeBeta simulation" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_resident_graph_snake_beta_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_comfy_resident_conv3d_runtime.py" call :run_required "ComfyUI resident Conv3D routing simulation" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_comfy_resident_conv3d_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_comfy_resident_resblock_graph_runtime.py" call :run_required "ComfyUI resident full ResBlock graph simulation" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_comfy_resident_resblock_graph_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_comfy_resident_resblock_scale_shift_runtime.py" call :run_required "ComfyUI resident scale-shift ResBlock graph simulation" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_comfy_resident_resblock_scale_shift_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_comfy_resident_resblock_updown_runtime.py" call :run_required "ComfyUI resident up/down ResBlock graph simulation" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_comfy_resident_resblock_updown_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_comfy_resident_video_resblock_runtime.py" call :run_required "ComfyUI resident VideoResBlock graph simulation" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_comfy_resident_video_resblock_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_comfy_resident_lora_runtime.py" call :run_required "ComfyUI resident standard LoRA graph simulation" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_comfy_resident_lora_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_resident_async_runtime.py" call :run_required "resident async graph simulation" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_resident_async_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_resident_async_pipeline_runtime.py" call :run_required "resident async multi-graph pipeline simulation" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_resident_async_pipeline_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_torch.py" call :run_required "14-test torch simulation" "%VKSSBO_TORCH_PYTHON%" "%VKSSBO_TEST_ROOT%\test_torch.py"
if exist "%VKSSBO_TEST_ROOT%\test_linear_silu_fusion_runtime.py" call :run_required "linear silu simulation" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_linear_silu_fusion_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_linear_gelu_fusion_runtime.py" call :run_required "linear gelu simulation" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_linear_gelu_fusion_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_matmul_softmax_fusion_runtime.py" call :run_required "matmul softmax simulation" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_matmul_softmax_fusion_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_conv2d_silu_fusion_runtime.py" call :run_required "conv2d silu simulation" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_conv2d_silu_fusion_runtime.py"

:finish
>> "%LOG%" echo.
>> "%LOG%" echo ============================================================
>> "%LOG%" echo Finished: %DATE% %TIME%
if "%FAILED%"=="0" (
    >> "%LOG%" echo RESULT: PASS
    echo Discrete simulation compatibility test completed: PASS
) else (
    >> "%LOG%" echo RESULT: FAIL
    echo Discrete simulation compatibility test completed: FAIL
)
move /y "%LOG%" "%FINAL_LOG%" >nul
if errorlevel 1 (
    if defined RUN_LOCK_HELD rmdir "%RUN_LOCK%" >nul 2>&1
    echo ERROR: failed to promote the completed simulation log.
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
