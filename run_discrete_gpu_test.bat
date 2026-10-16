@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul

rem ============================================================================
rem PyTorch for Vulkan - legacy real discrete-GPU diagnostics/benchmark runner
rem
rem This runner is reserved for the later one-shot real dGPU pass. The current
rem workflow uses run_discrete_simulation_test.bat for compatibility only.
rem Set VKSSBO_ENABLE_REAL_DGPU_TEST=1 explicitly before using this file.
rem Run this file on the target machine after rebuilding torch_cpu.dll there.
rem Edit the defaults below, or set the variables before launching this file.
rem The complete log is written beside this BAT file.
rem Example for a different machine:
rem   set VKSSBO_PYTHON=C:\path\to\python.exe
rem   set VKSSBO_TEST_ROOT=C:\path\to\tools\vulkan_ssbo
rem   set VKSSBO_DEVICE_NAME=NVIDIA
rem   run_discrete_gpu_test.bat
rem ============================================================================

set "ROOT=%~dp0"
set "LOG=%ROOT%discrete_gpu_test.log"
if not defined VKSSBO_ENABLE_REAL_DGPU_TEST set "VKSSBO_ENABLE_REAL_DGPU_TEST=0"
if /i not "%VKSSBO_ENABLE_REAL_DGPU_TEST%"=="1" (
    > "%LOG%" echo Real discrete-GPU runner is disabled for the current test phase.
    >> "%LOG%" echo Use run_discrete_simulation_test.bat for compatibility or set VKSSBO_ENABLE_REAL_DGPU_TEST=1 for the later one-shot pass.
    echo Real discrete-GPU runner skipped. Use run_discrete_simulation_test.bat for current compatibility tests.
    echo Log: %LOG%
    exit /b 0
)

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

if not defined VULKAN_SDK set "VULKAN_SDK="
if not defined VKSSBO_DEVICE_INDEX set "VKSSBO_DEVICE_INDEX="
if not defined VKSSBO_BENCH_SHAPE set "VKSSBO_BENCH_SHAPE=64,128,64"
if not defined VKSSBO_BENCH_WARMUP set "VKSSBO_BENCH_WARMUP=3"
if not defined VKSSBO_BENCH_REPEATS set "VKSSBO_BENCH_REPEATS=9"
if not defined VKSSBO_RESIDENT_DEPTH set "VKSSBO_RESIDENT_DEPTH=8"
if not defined VKSSBO_RESIDENT_WARMUP set "VKSSBO_RESIDENT_WARMUP=2"
if not defined VKSSBO_RESIDENT_REPEATS set "VKSSBO_RESIDENT_REPEATS=7"
set "FAILED=0"
rem BEGIN HOST DEVICE SELECTION
if not defined VKSSBO_DEVICE_NAME if not defined VKSSBO_DEVICE_INDEX if /i not "%VKSSBO_PACKAGE_SIMULATION%"=="1" (
    for /f "usebackq delims=" %%G in (`""%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\detect_discrete_gpu.py""`) do set "VKSSBO_DEVICE_NAME=%%G"
    if not defined VKSSBO_DEVICE_NAME set "FAILED=1"
)
rem END HOST DEVICE SELECTION

> "%LOG%" echo ============================================================
>> "%LOG%" echo PyTorch for Vulkan discrete GPU test
>> "%LOG%" echo Started: %DATE% %TIME%
>> "%LOG%" echo Host: %COMPUTERNAME%
>> "%LOG%" echo User: %USERNAME%
>> "%LOG%" echo Root: %ROOT%
>> "%LOG%" echo Python: %VKSSBO_PYTHON%
>> "%LOG%" echo TestRoot: %VKSSBO_TEST_ROOT%
>> "%LOG%" echo VKSSBO_DEVICE_INDEX=%VKSSBO_DEVICE_INDEX%
>> "%LOG%" echo VKSSBO_DEVICE_NAME=%VKSSBO_DEVICE_NAME%
>> "%LOG%" echo VKSSBO_TORCH_PYTHON=%VKSSBO_TORCH_PYTHON%
>> "%LOG%" echo VKSSBO_BENCH_SHAPE=%VKSSBO_BENCH_SHAPE%
>> "%LOG%" echo VKSSBO_BENCH_WARMUP=%VKSSBO_BENCH_WARMUP%
>> "%LOG%" echo VKSSBO_BENCH_REPEATS=%VKSSBO_BENCH_REPEATS%
>> "%LOG%" echo VKSSBO_RESIDENT_DEPTH=%VKSSBO_RESIDENT_DEPTH%
>> "%LOG%" echo VKSSBO_RESIDENT_WARMUP=%VKSSBO_RESIDENT_WARMUP%
>> "%LOG%" echo VKSSBO_RESIDENT_REPEATS=%VKSSBO_RESIDENT_REPEATS%
>> "%LOG%" echo ============================================================

if "%FAILED%"=="1" (
    >> "%LOG%" echo ERROR: automatic GPU selection failed or was ambiguous. Set VKSSBO_DEVICE_NAME or VKSSBO_DEVICE_INDEX explicitly.
    goto :finish
)

call :run_optional "Python version" "%VKSSBO_PYTHON%" --version

>> "%LOG%" echo.
>> "%LOG%" echo [preflight] PyTorch import
"%VKSSBO_PYTHON%" -c "import torch; print(torch.__file__); print(torch.__version__)" >> "%LOG%" 2>&1
set "TORCH_IMPORT_RC=%ERRORLEVEL%"
>> "%LOG%" echo [exit] PyTorch import rc=!TORCH_IMPORT_RC!
if not "!TORCH_IMPORT_RC!"=="0" (
    >> "%LOG%" echo ERROR: selected Python cannot import PyTorch.
    >> "%LOG%" echo Install the Vulkan wheel into this interpreter or set VKSSBO_PYTHON explicitly.
    set "FAILED=1"
    goto :finish
)

if /i not "%VKSSBO_TORCH_PYTHON%"=="%VKSSBO_PYTHON%" (
    >> "%LOG%" echo.
    >> "%LOG%" echo [preflight] regression-test PyTorch import
    "%VKSSBO_TORCH_PYTHON%" -c "import torch; print(torch.__file__); print(torch.__version__)" >> "%LOG%" 2>&1
    set "TORCH_REGRESSION_IMPORT_RC=%ERRORLEVEL%"
    >> "%LOG%" echo [exit] regression-test PyTorch import rc=!TORCH_REGRESSION_IMPORT_RC!
    if not "!TORCH_REGRESSION_IMPORT_RC!"=="0" (
        >> "%LOG%" echo ERROR: VKSSBO_TORCH_PYTHON cannot import PyTorch.
        set "FAILED=1"
        goto :finish
    )
)

>> "%LOG%" echo.
>> "%LOG%" echo [environment] Vulkan summary
where vulkaninfo.exe >> "%LOG%" 2>&1
if errorlevel 1 (
    >> "%LOG%" echo vulkaninfo.exe not found on PATH
) else (
    vulkaninfo.exe --summary >> "%LOG%" 2>&1
)

>> "%LOG%" echo.
>> "%LOG%" echo [environment] NVIDIA summary
where nvidia-smi.exe >> "%LOG%" 2>&1
if errorlevel 1 (
    >> "%LOG%" echo nvidia-smi.exe not found on PATH
) else (
    nvidia-smi.exe "--query-gpu=name,driver_version,memory.total" "--format=csv,noheader" >> "%LOG%" 2>&1
)

if not exist "%VKSSBO_TEST_ROOT%\test_device_selection_runtime.py" (
    >> "%LOG%" echo ERROR: test root does not contain test_device_selection_runtime.py
    >> "%LOG%" echo Set VKSSBO_TEST_ROOT to the directory containing the Vulkan SSBO tests.
    set "FAILED=1"
) else (
    call :run_required "device selection" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_device_selection_runtime.py"
    call :run_required "async capabilities disabled" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_async_transfer_capabilities_runtime.py"

    set "VKSSBO_ASYNC_TRANSFER=0"
    set "VKSSBO_TRACE_SUBMIT=1"
    call :run_required "native SSBO baseline" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_vulkan_ssbo_native_runtime.py"
    call :run_required "fence submit" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_fence_submit_runtime.py"

    set "VKSSBO_ASYNC_TRANSFER=1"
    set "VKSSBO_TRACE_SUBMIT=1"
    call :run_required "async capabilities requested" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_async_transfer_capabilities_runtime.py"
    if exist "%VKSSBO_TEST_ROOT%\test_async_transfer_threshold_runtime.py" call :run_required "async transfer threshold" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_async_transfer_threshold_runtime.py"
    if exist "%VKSSBO_TEST_ROOT%\test_staging_pool_runtime.py" call :run_required "staging pool reuse" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_staging_pool_runtime.py"
    call :run_required "native SSBO async request" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_vulkan_ssbo_native_runtime.py"

    set "VKSSBO_TRACE_SUBMIT=0"
    if exist "%VKSSBO_TEST_ROOT%\test_int8_capability_runtime.py" call :run_required "INT8 shader capability" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_int8_capability_runtime.py"
    if exist "%VKSSBO_TEST_ROOT%\test_int8_storage_probe_runtime.py" call :run_required "INT8 SSBO shader probe" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_int8_storage_probe_runtime.py"
    if exist "%VKSSBO_TEST_ROOT%\test_int8_weight_only_contract_runtime.py" call :run_required "INT8 weight-only contract" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_int8_weight_only_contract_runtime.py"
    if exist "%VKSSBO_TEST_ROOT%\test_int4_weight_only_contract_runtime.py" call :run_required "INT4 weight-only contract" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_int4_weight_only_contract_runtime.py"
    if exist "%VKSSBO_TEST_ROOT%\test_fp8_weight_only_contract_runtime.py" call :run_required "FP8 weight-only contract" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_fp8_weight_only_contract_runtime.py"
    if exist "%VKSSBO_TEST_ROOT%\test_fp4_weight_only_contract_runtime.py" call :run_required "FP4 weight-only contract" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_fp4_weight_only_contract_runtime.py"
    call :run_required "14-test torch regression" "%VKSSBO_TORCH_PYTHON%" "%VKSSBO_TEST_ROOT%\test_torch.py"

    if exist "%VKSSBO_TEST_ROOT%\test_discrete_simulation_runtime.py" call :run_required "discrete policy simulation" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_discrete_simulation_runtime.py"

    if exist "%VKSSBO_TEST_ROOT%\test_vulkan_ssbo_native_runtime.py" (
        set "VKSSBO_SIMULATE_DISCRETE=1"
        set "VKSSBO_SIMULATED_VRAM_MB=4096"
        set "VKSSBO_SIMULATION_FORCE_STAGING=1"
        set "VKSSBO_SIMULATION_STRICT_BUDGET=0"
        set "VKSSBO_ASYNC_TRANSFER=1"
        call :run_required "native SSBO simulation" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_vulkan_ssbo_native_runtime.py"
        set "VKSSBO_SIMULATE_DISCRETE="
        set "VKSSBO_SIMULATED_VRAM_MB="
        set "VKSSBO_SIMULATION_FORCE_STAGING="
        set "VKSSBO_SIMULATION_STRICT_BUDGET="
        set "VKSSBO_ASYNC_TRANSFER=0"
    )

    if exist "%VKSSBO_TEST_ROOT%\test_matmul_softmax_fusion_runtime.py" call :run_required "matmul softmax fusion" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_matmul_softmax_fusion_runtime.py"
    if exist "%VKSSBO_TEST_ROOT%\test_linear_silu_fusion_runtime.py" call :run_required "linear silu fusion" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_linear_silu_fusion_runtime.py"
    if exist "%VKSSBO_TEST_ROOT%\test_linear_gelu_fusion_runtime.py" call :run_required "linear gelu fusion" "%VKSSBO_PYTHON%" "%VKSSBO_TEST_ROOT%\test_linear_gelu_fusion_runtime.py"

    rem The transfer benchmark lives in the PyTorch source tree and is run
    rem once per mode in a fresh process. It is diagnostic-only: a single
    rem matmul cannot demonstrate queue overlap, so the log is evidence rather
    rem than an automatic default-policy switch.
    if exist "%ROOT%test\bench_vulkan_ssbo_transfer.py" (
        set "VKSSBO_PROFILE=1"
        set "VKSSBO_ASYNC_TRANSFER=0"
        call :run_transfer_benchmark "transfer benchmark sync"
        set "VKSSBO_ASYNC_TRANSFER=1"
        call :run_transfer_benchmark "transfer benchmark async"
        set "VKSSBO_PROFILE="
    ) else (
        >> "%LOG%" echo [skip] transfer benchmark source runner not found at "%ROOT%test\bench_vulkan_ssbo_transfer.py"
    )

    if exist "%VKSSBO_TEST_ROOT%\bench_resident_json.py" (
        set "VKSSBO_ASYNC_TRANSFER=0"
        call :run_resident_benchmark "resident graph benchmark sync"
        set "VKSSBO_ASYNC_TRANSFER=1"
        call :run_resident_benchmark "resident graph benchmark async"
    ) else (
        >> "%LOG%" echo [skip] resident benchmark runner not found at "%VKSSBO_TEST_ROOT%\bench_resident_json.py"
    )
)

:finish
>> "%LOG%" echo.
>> "%LOG%" echo ============================================================
>> "%LOG%" echo Finished: %DATE% %TIME%
if "%FAILED%"=="0" (
    >> "%LOG%" echo RESULT: PASS
    echo Discrete GPU test completed: PASS
) else (
    >> "%LOG%" echo RESULT: FAIL
    echo Discrete GPU test completed: FAIL
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
if "%ARG1%"=="" (
    "%COMMAND%" >> "%LOG%" 2>&1
) else if "%ARG2%"=="" (
    "%COMMAND%" "%ARG1%" >> "%LOG%" 2>&1
) else (
    "%COMMAND%" "%ARG1%" "%ARG2%" >> "%LOG%" 2>&1
)
set "RC=%ERRORLEVEL%"
>> "%LOG%" echo [exit] %LABEL% rc=!RC!
if not "!RC!"=="0" set "FAILED=1"
exit /b 0

:run_optional
set "LABEL=%~1"
set "COMMAND=%~2"
set "ARG1=%~3"
set "ARG2=%~4"
>> "%LOG%" echo.
>> "%LOG%" echo [optional] %LABEL%
if "%ARG1%"=="" (
    "%COMMAND%" >> "%LOG%" 2>&1
) else if "%ARG2%"=="" (
    "%COMMAND%" "%ARG1%" >> "%LOG%" 2>&1
) else (
    "%COMMAND%" "%ARG1%" "%ARG2%" >> "%LOG%" 2>&1
)
>> "%LOG%" echo [exit] %LABEL% rc=!ERRORLEVEL!
exit /b 0

:run_transfer_benchmark
set "LABEL=%~1"
>> "%LOG%" echo.
>> "%LOG%" echo [run] %LABEL%
>> "%LOG%" echo Command: "%VKSSBO_TORCH_PYTHON%" "%ROOT%test\bench_vulkan_ssbo_transfer.py" --warmup %VKSSBO_BENCH_WARMUP% --repeats %VKSSBO_BENCH_REPEATS% --shape %VKSSBO_BENCH_SHAPE%
"%VKSSBO_TORCH_PYTHON%" "%ROOT%test\bench_vulkan_ssbo_transfer.py" --warmup %VKSSBO_BENCH_WARMUP% --repeats %VKSSBO_BENCH_REPEATS% --shape %VKSSBO_BENCH_SHAPE% >> "%LOG%" 2>&1
set "RC=%ERRORLEVEL%"
>> "%LOG%" echo [exit] %LABEL% rc=!RC!
if not "!RC!"=="0" set "FAILED=1"
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
