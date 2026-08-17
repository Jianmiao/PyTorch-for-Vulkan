@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 >nul 2>&1
dumpbin /exports "C:\Windows\System32\vulkan-1.dll" > "C:\Users\21294\AppData\Local\Temp\opencode\vulkan_exports.txt"
"D:\ComfyUI Vulkan beat\tools\Python313\python.exe" "C:\Users\21294\AppData\Local\Temp\opencode\gen_def.py"
mkdir "D:\PyTorch for Vulkan\VulkanSDK\Lib" 2>nul
lib /def:"C:\Users\21294\AppData\Local\Temp\opencode\vulkan.def" /out:"D:\PyTorch for Vulkan\VulkanSDK\Lib\vulkan-1.lib" /machine:x64
echo LIB_EXIT=%ERRORLEVEL%
