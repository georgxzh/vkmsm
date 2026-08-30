cmake_minimum_required(VERSION 3.20)
project(vkmsm CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Vulkan REQUIRED)
find_program(GLSLC glslc HINTS "$ENV{VULKAN_SDK}/Bin")
if(NOT GLSLC)
    message(FATAL_ERROR "glslc not found - is the Vulkan SDK installed and VULKAN_SDK set?")
endif()

# Compiles shaders/<name> to ${CMAKE_BINARY_DIR}/shaders/<name>.spv and
# sets ${out_var} to that output path in the caller's scope. Extra args are
# additional dependencies (e.g. an #include'd .glsl file) that should also
# trigger a recompile when changed.
function(compile_shader name out_var)
    set(src ${CMAKE_SOURCE_DIR}/shaders/${name})
    set(spv ${CMAKE_BINARY_DIR}/shaders/${name}.spv)
    add_custom_command(
        OUTPUT ${spv}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/shaders
        COMMAND ${GLSLC} ${src} -o ${spv}
        DEPENDS ${src} ${ARGN}
        COMMENT "Compiling ${name} -> SPIR-V"
        VERBATIM
    )
    add_custom_target(shader_${name} ALL DEPENDS ${spv})
    set(${out_var} ${spv} PARENT_SCOPE)
endfunction()

set(FP_ARITH_GLSL ${CMAKE_SOURCE_DIR}/shaders/fp_arith.glsl)
set(POINT_ARITH_GLSL ${CMAKE_SOURCE_DIR}/shaders/point_arith.glsl)
compile_shader(add.comp ADD_SHADER_SPV)
compile_shader(fp_ops.comp FP_OPS_SHADER_SPV ${FP_ARITH_GLSL})
compile_shader(point_ops.comp POINT_OPS_SHADER_SPV ${FP_ARITH_GLSL} ${POINT_ARITH_GLSL})
compile_shader(pippenger_buckets.comp PIPPENGER_BUCKETS_SHADER_SPV ${FP_ARITH_GLSL} ${POINT_ARITH_GLSL})

add_executable(stage0_vecadd src/stage0_vecadd.cpp)
add_dependencies(stage0_vecadd shader_add.comp)
target_link_libraries(stage0_vecadd PRIVATE Vulkan::Vulkan)
target_compile_definitions(stage0_vecadd PRIVATE SHADER_PATH="${ADD_SHADER_SPV}")

# --- blst: CPU-side oracle library for known-answer test vectors ---
# blst has no CMakeLists.txt of its own; it ships build.bat (Windows) /
# build.sh (Unix) which invoke the platform toolchain directly. We fetch
# its source with FetchContent, then drive its own build script ourselves.
include(FetchContent)
FetchContent_Declare(
    blst
    GIT_REPOSITORY https://github.com/supranational/blst.git
    GIT_TAG        v0.3.17
    GIT_SHALLOW    TRUE
)
FetchContent_GetProperties(blst)
if(NOT blst_POPULATED)
    FetchContent_Populate(blst)
endif()

if(WIN32)
    # build.bat needs cl.exe/ml64.exe/lib.exe on PATH *and* the INCLUDE/LIB
    # env vars set (for standard headers and the CRT/Windows SDK libs) -
    # that's only true automatically inside a "Developer Command Prompt".
    # We don't assume one is active; instead we load the same vcvars64.bat
    # script Visual Studio itself uses, then run build.bat in that shell.
    # (Also: cmd requires an explicit ".\" prefix to find a batch file by
    # relative name when invoked non-interactively via /c.)
    # Derive the VS install root from the compiler CMake already found
    # (its path looks like <VSROOT>/VC/Tools/MSVC/<ver>/bin/Hostx64/x64/cl.exe),
    # so this locates vcvars64.bat from the exact same install/edition in use.
    string(REGEX REPLACE "/VC/Tools/MSVC/.*" "" VS_INSTALL_DIR "${CMAKE_CXX_COMPILER}")
    set(VCVARS64_BAT "${VS_INSTALL_DIR}/VC/Auxiliary/Build/vcvars64.bat")
    if(NOT EXISTS "${VCVARS64_BAT}")
        message(FATAL_ERROR "Could not find vcvars64.bat at ${VCVARS64_BAT}")
    endif()
    set(BLST_LIB "${blst_SOURCE_DIR}/blst.lib")
    # Nested quoting through CMake -> generated .vcxproj -> cmd.exe is
    # fragile for an inline "call ... && ..." string, so we generate a
    # tiny standalone wrapper .bat instead and just invoke that directly.
    set(BLST_BUILD_WRAPPER "${CMAKE_BINARY_DIR}/build_blst.bat")
    file(WRITE "${BLST_BUILD_WRAPPER}"
"@echo off\r\n\
call \"${VCVARS64_BAT}\"\r\n\
if errorlevel 1 exit /b 1\r\n\
cd /d \"${blst_SOURCE_DIR}\"\r\n\
call .\\build.bat\r\n"
    )
    add_custom_command(
        OUTPUT ${BLST_LIB}
        COMMAND cmd /c ".\\build_blst.bat"
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        DEPENDS "${BLST_BUILD_WRAPPER}"
        COMMENT "Building blst (CPU reference library) via build.bat"
        VERBATIM
    )
else()
    set(BLST_LIB "${blst_SOURCE_DIR}/libblst.a")
    add_custom_command(
        OUTPUT ${BLST_LIB}
        COMMAND ./build.sh
        WORKING_DIRECTORY ${blst_SOURCE_DIR}
        COMMENT "Building blst (CPU reference library) via build.sh"
        VERBATIM
    )
endif()
add_custom_target(blst_build DEPENDS ${BLST_LIB})

add_library(blst STATIC IMPORTED GLOBAL)
set_target_properties(blst PROPERTIES IMPORTED_LOCATION ${BLST_LIB})
target_include_directories(blst INTERFACE ${blst_SOURCE_DIR}/bindings)
add_dependencies(blst blst_build)

add_executable(stage1_field_test src/stage1_field_test.cpp src/fp.cpp)
target_link_libraries(stage1_field_test PRIVATE blst)

add_executable(stage2_point_test src/stage2_point_test.cpp src/point.cpp src/fp.cpp)
target_link_libraries(stage2_point_test PRIVATE blst)

add_executable(stage3a_msm_naive_test src/stage3a_msm_naive_test.cpp src/msm.cpp src/point.cpp src/fp.cpp)
target_link_libraries(stage3a_msm_naive_test PRIVATE blst)

add_executable(stage3b_pippenger_test src/stage3b_pippenger_test.cpp src/msm.cpp src/point.cpp src/fp.cpp)
target_link_libraries(stage3b_pippenger_test PRIVATE blst)

add_executable(stage4_field_shader_test src/stage4_field_shader_test.cpp src/fp.cpp)
add_dependencies(stage4_field_shader_test shader_fp_ops.comp)
target_link_libraries(stage4_field_shader_test PRIVATE Vulkan::Vulkan)
target_compile_definitions(stage4_field_shader_test PRIVATE SHADER_PATH="${FP_OPS_SHADER_SPV}")

add_executable(stage5_point_shader_test src/stage5_point_shader_test.cpp src/point.cpp src/fp.cpp)
add_dependencies(stage5_point_shader_test shader_point_ops.comp)
target_link_libraries(stage5_point_shader_test PRIVATE Vulkan::Vulkan blst)
target_compile_definitions(stage5_point_shader_test PRIVATE SHADER_PATH="${POINT_OPS_SHADER_SPV}")

add_executable(stage6_pippenger_shader_test
    src/stage6_pippenger_shader_test.cpp src/gpu_pippenger.cpp src/msm.cpp src/point.cpp src/fp.cpp)
add_dependencies(stage6_pippenger_shader_test shader_pippenger_buckets.comp)
target_link_libraries(stage6_pippenger_shader_test PRIVATE Vulkan::Vulkan blst)
target_compile_definitions(stage6_pippenger_shader_test PRIVATE
    SHADER_PATH_PIPPENGER_BUCKETS="${PIPPENGER_BUCKETS_SHADER_SPV}")

add_executable(stage7_benchmark
    src/stage7_benchmark.cpp src/gpu_pippenger.cpp src/msm.cpp src/point.cpp src/fp.cpp)
add_dependencies(stage7_benchmark shader_pippenger_buckets.comp)
target_link_libraries(stage7_benchmark PRIVATE Vulkan::Vulkan blst)
target_compile_definitions(stage7_benchmark PRIVATE
    SHADER_PATH_PIPPENGER_BUCKETS="${PIPPENGER_BUCKETS_SHADER_SPV}")
