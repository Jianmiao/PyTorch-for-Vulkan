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
set "LOG=%ROOT%discrete_simulation_test.log"

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
set "VKSSBO_ASYNC_TRANSFER=1"
call :run_required "native SSBO staging simulation" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_vulkan_ssbo_native_runtime.py"
set "VKSSBO_TRACE_SUBMIT=0"
if exist "%VKSSBO_TEST_ROOT%\test_fp16_resident_runtime.py" call :run_required "FP16 resident simulation" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_fp16_resident_runtime.py"
if exist "%VKSSBO_TEST_ROOT%\test_fp16_boundaries_runtime.py" call :run_required "FP16 boundary simulation" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_fp16_boundaries_runtime.py"
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
echo Log: %LOG%
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
