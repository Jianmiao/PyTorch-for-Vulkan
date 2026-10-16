if(NOT INTERN_BUILD_ATEN_OPS)
  return()
endif()

# Find modules
if(NOT INTERN_BUILD_MOBILE)
  list(APPEND CMAKE_MODULE_PATH /usr/lib/x86_64-linux-gnu/)
  list(APPEND CMAKE_LIBRARY_PATH /usr/lib/x86_64-linux-gnu/ /usr/lib/aarch64-linux-gnu/)
endif()

list(APPEND CMAKE_MODULE_PATH
  ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/Modules
  ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/public
  ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/Modules_CUDA_fix)

#############################################

set(ATen_CPU_SRCS)
set(ATen_MTIA_SRCS)
set(ATen_XPU_SRCS)
set(ATen_XPU_INCLUDE)
set(ATen_CPU_TEST_SRCS)
set(ATen_CPU_INCLUDE)
set(ATen_THIRD_PARTY_INCLUDE)
set(ATen_CUDA_CPP_SRCS)
set(ATen_CUDA_CU_SRCS)
set(ATen_CUDA_LINALG_SRCS)
set(ATen_CUDA_SRCS_W_SORT_BY_KEY)
set(ATen_CUDA_TEST_SRCS)
set(ATen_CUDA_INCLUDE)
set(ATen_NVRTC_STUB_SRCS)
set(ATen_HIP_SRCS)
set(ATen_HIP_SRCS_W_SORT_BY_KEY)
set(ATen_HIP_TEST_SRCS)
set(ATen_HIP_INCLUDE)
set(ATen_MPS_SRCS)
set(ATen_MPS_TEST_SRCS)
set(ATen_XPU_TEST_SRCS)
set(ATen_VULKAN_TEST_SRCS)
set(ATen_CPU_DEPENDENCY_LIBS)
set(ATen_XPU_DEPENDENCY_LIBS)
set(ATen_CUDA_DEPENDENCY_LIBS)
set(ATen_HIP_DEPENDENCY_LIBS)
set(ATen_PUBLIC_CUDA_DEPENDENCY_LIBS)
set(ATen_PUBLIC_HIP_DEPENDENCY_LIBS)
set(ATEN_INSTALL_BIN_SUBDIR "bin" CACHE PATH "ATen install binary subdirectory")
set(ATEN_INSTALL_LIB_SUBDIR "lib" CACHE PATH "ATen install library subdirectory")
set(ATEN_INSTALL_INCLUDE_SUBDIR "include" CACHE PATH "ATen install include subdirectory")
set(MEM_EFF_ATTENTION_CUDA_SOURCES)

set(TH_CPU_INCLUDE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${CMAKE_CURRENT_BINARY_DIR}/src
  ${CMAKE_BINARY_DIR}/aten/src)
list(APPEND ATen_CPU_INCLUDE ${TH_CPU_INCLUDE})

if(USE_VULKAN)
  list(APPEND ATen_CPU_INCLUDE ${CMAKE_BINARY_DIR}/vulkan ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/VulkanMemoryAllocator)

  # The SSBO memory policy is a pure-Rust static library. It receives
  # accounting snapshots only; C++/VMA owns every Vulkan allocation.
  set(VKSSBO_MEMORY_POLICY_DIR
      "${CMAKE_CURRENT_SOURCE_DIR}/src/ATen/native/vulkan/ops/ssbo/memory_policy")
  find_program(VKSSBO_CARGO_EXECUTABLE cargo)
  if(VKSSBO_CARGO_EXECUTABLE)
    set(VKSSBO_RUST_TARGET_DIR "${CMAKE_BINARY_DIR}/vkssbo_memory_policy")
    if(WIN32)
      set(VKSSBO_MEMORY_POLICY_LIB
          "${VKSSBO_RUST_TARGET_DIR}/release/vkssbo_memory_policy.lib")
    else()
      set(VKSSBO_MEMORY_POLICY_LIB
          "${VKSSBO_RUST_TARGET_DIR}/release/libvkssbo_memory_policy.a")
    endif()
    add_custom_command(
      OUTPUT "${VKSSBO_MEMORY_POLICY_LIB}"
      COMMAND ${CMAKE_COMMAND} -E env
              "CARGO_TARGET_DIR=${VKSSBO_RUST_TARGET_DIR}"
              "${VKSSBO_CARGO_EXECUTABLE}" build
              --manifest-path "${VKSSBO_MEMORY_POLICY_DIR}/Cargo.toml"
              --release
      DEPENDS
        "${VKSSBO_MEMORY_POLICY_DIR}/Cargo.toml"
        "${VKSSBO_MEMORY_POLICY_DIR}/Cargo.lock"
        "${VKSSBO_MEMORY_POLICY_DIR}/include/memory_policy.h"
        "${VKSSBO_MEMORY_POLICY_DIR}/src/lib.rs"
      COMMENT "Building the vkssbo Rust memory policy"
      VERBATIM)
    add_custom_target(vkssbo_memory_policy_build
                      DEPENDS "${VKSSBO_MEMORY_POLICY_LIB}")
    add_library(vkssbo_memory_policy STATIC IMPORTED GLOBAL)
    set_target_properties(vkssbo_memory_policy PROPERTIES
      IMPORTED_LOCATION "${VKSSBO_MEMORY_POLICY_LIB}")
    add_dependencies(vkssbo_memory_policy vkssbo_memory_policy_build)
    list(APPEND ATen_CPU_DEPENDENCY_LIBS vkssbo_memory_policy)
    message(STATUS "Vulkan SSBO Rust memory policy enabled")
  else()
    message(STATUS "Cargo not found; Vulkan SSBO Rust memory policy disabled")
  endif()
endif()

# Find the HIP package, set the HIP paths, load the HIP CMake.
if(USE_ROCM)
  include(LoadHIP)
  if(NOT PYTORCH_FOUND_HIP)
    set(USE_ROCM OFF)
  endif()
endif()

# Both CUDA and ROCM are enabled and found. Report an error.
if(USE_CUDA AND USE_ROCM)
  message(FATAL_ERROR "Both CUDA and ROCm are enabled and found. PyTorch can only be built with either of them. Please turn one off by using either USE_CUDA=OFF or USE_ROCM=OFF.")
endif()

if(USE_ROCM)
  # TODO: AT_HIP_ENABLED (change this once we represent HIP as HIP in
  # ATen proper)
  set(AT_CUDA_ENABLED 1)
  add_subdirectory(src/THH)
  message("ROCm is enabled.")
elseif(USE_CUDA)
  set(AT_CUDA_ENABLED 1)
  add_subdirectory(src/THC)
else()
  message("disabling CUDA because USE_CUDA is set false")
  set(AT_CUDA_ENABLED 0)
endif()

if(NOT USE_NNPACK)
  set(AT_NNPACK_ENABLED 0)
else()
  set(AT_NNPACK_ENABLED 1)
endif()

if(NOT USE_CUSPARSELT)
  set(AT_CUSPARSELT_ENABLED 0)
else()
  set(AT_CUSPARSELT_ENABLED 1)
endif()

# Add hipSPARSELt support flag if the package is available.
if(USE_ROCM AND hipsparselt_FOUND)
  set(AT_HIPSPARSELT_ENABLED 1)
else()
  set(AT_HIPSPARSELT_ENABLED 0)
endif()

list(APPEND ATen_CPU_INCLUDE
  ${CMAKE_CURRENT_SOURCE_DIR}/src)
add_subdirectory(src/ATen)

# Pass source, includes, and libs to parent
set(ATen_CPU_SRCS ${ATen_CPU_SRCS} PARENT_SCOPE)
set(ATen_CORE_SRCS ${ATen_CORE_SRCS} PARENT_SCOPE)
set(ATen_MTIA_SRCS ${ATen_MTIA_SRCS} PARENT_SCOPE)
set(ATen_XPU_SRCS ${ATen_XPU_SRCS} PARENT_SCOPE)
set(ATen_XPU_INCLUDE ${ATen_XPU_INCLUDE} PARENT_SCOPE)
set(ATen_CUDA_CU_SRCS ${ATen_CUDA_CU_SRCS} PARENT_SCOPE)
set(ATen_CUDA_CPP_SRCS ${ATen_CUDA_CPP_SRCS} PARENT_SCOPE)
set(ATen_CUDA_LINALG_SRCS ${ATen_CUDA_LINALG_SRCS} PARENT_SCOPE)
set(ATen_CUDA_SRCS_W_SORT_BY_KEY ${ATen_CUDA_SRCS_W_SORT_BY_KEY} PARENT_SCOPE)
set(ATen_CUDA_CU_SRCS_W_SORT_BY_KEY ${ATen_CUDA_CU_SRCS_W_SORT_BY_KEY} PARENT_SCOPE)
set(ATen_HIP_SRCS ${ATen_HIP_SRCS} PARENT_SCOPE)
set(ATen_MPS_SRCS ${ATen_MPS_SRCS} PARENT_SCOPE)
set(ATen_MPS_TEST_SRCS ${ATen_MPS_TEST_SRCS} PARENT_SCOPE)
set(ATen_HIP_SRCS_W_SORT_BY_KEY ${ATen_HIP_SRCS_W_SORT_BY_KEY} PARENT_SCOPE)
set(ATen_XPU_TEST_SRCS ${ATen_XPU_TEST_SRCS} PARENT_SCOPE)
set(ATen_NVRTC_STUB_SRCS ${ATen_NVRTC_STUB_SRCS} PARENT_SCOPE)
set(ATen_CPU_TEST_SRCS ${ATen_CPU_TEST_SRCS} PARENT_SCOPE)
set(ATen_CUDA_TEST_SRCS ${ATen_CUDA_TEST_SRCS} PARENT_SCOPE)
set(ATen_HIP_TEST_SRCS ${ATen_HIP_TEST_SRCS} PARENT_SCOPE)
set(ATen_VULKAN_TEST_SRCS ${ATen_VULKAN_TEST_SRCS} PARENT_SCOPE)
set(ATen_MOBILE_BENCHMARK_SRCS ${ATen_MOBILE_BENCHMARK_SRCS} PARENT_SCOPE)
set(ATen_MOBILE_TEST_SRCS ${ATen_MOBILE_TEST_SRCS} PARENT_SCOPE)
set(ATen_VEC_TEST_SRCS ${ATen_VEC_TEST_SRCS} PARENT_SCOPE)
set(ATen_CPU_INCLUDE ${ATen_CPU_INCLUDE} PARENT_SCOPE)
set(ATen_CUDA_INCLUDE ${ATen_CUDA_INCLUDE} PARENT_SCOPE)
set(ATen_HIP_INCLUDE ${ATen_HIP_INCLUDE} PARENT_SCOPE)
set(ATen_XPU_INCLUDE ${ATen_XPU_INCLUDE} PARENT_SCOPE)
set(ATen_THIRD_PARTY_INCLUDE ${ATen_THIRD_PARTY_INCLUDE} PARENT_SCOPE)
set(ATen_CPU_DEPENDENCY_LIBS ${ATen_CPU_DEPENDENCY_LIBS} PARENT_SCOPE)
set(ATen_XPU_DEPENDENCY_LIBS ${ATen_XPU_DEPENDENCY_LIBS} PARENT_SCOPE)
set(ATen_CUDA_DEPENDENCY_LIBS ${ATen_CUDA_DEPENDENCY_LIBS} PARENT_SCOPE)
set(ATen_HIP_DEPENDENCY_LIBS ${ATen_HIP_DEPENDENCY_LIBS} PARENT_SCOPE)
set(ATen_CORE_TEST_SRCS ${ATen_CORE_TEST_SRCS} PARENT_SCOPE)
set(FLASH_ATTENTION_CUDA_SOURCES ${FLASH_ATTENTION_CUDA_SOURCES} PARENT_SCOPE)
set(MEM_EFF_ATTENTION_CUDA_SOURCES ${MEM_EFF_ATTENTION_CUDA_SOURCES} PARENT_SCOPE)
set(ATen_ATTENTION_KERNEL_SRCS ${ATen_ATTENTION_KERNEL_SRCS} PARENT_SCOPE)
