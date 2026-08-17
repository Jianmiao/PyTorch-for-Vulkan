@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 >nul 2>&1
set USE_VULKAN=1
set USE_VULKAN_SHADERC_RUNTIME=1
set USE_VULKAN_WRAPPER=0
set USE_VULKAN_FP16_INFERENCE=1
set USE_CUDA=0
set USE_DISTRIBUTED=0
set VULKAN_SDK=D:\ComfyUI Vulkan beat\tools\VulkanSDK\1.4.357.0
set PYTHONIOENCODING=utf-8
cd /d "D:\PyTorch for Vulkan\pytorch-src"
"D:\ComfyUI Vulkan beat\tools\Python312\python.exe" setup.py bdist_wheel 2>&1
echo BUILD_EXIT=%ERRORLEVEL%
