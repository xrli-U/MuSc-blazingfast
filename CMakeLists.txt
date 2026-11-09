cmake_minimum_required(VERSION 3.24)
project(MuSc LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT WIN32 OR NOT MSVC OR NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(FATAL_ERROR "MuSc requires 64-bit Windows and MSVC.")
endif()
find_path(LIBTORCH_ROOT NAMES lib/torch_cpu.lib
    HINTS ENV LIBTORCH_ROOT "${CMAKE_SOURCE_DIR}/third_party/libtorch" "$ENV{LOCALAPPDATA}/MuSc/deps/libtorch"
    NO_DEFAULT_PATH DOC "Root of the Windows x64 LibTorch distribution")
if(NOT EXISTS "${LIBTORCH_ROOT}/lib/torch_cpu.lib")
    message(FATAL_ERROR "Set LIBTORCH_ROOT to an extracted Windows x64 LibTorch distribution")
endif()
find_package(yaml-cpp 0.8 CONFIG QUIET HINTS
    "${CMAKE_SOURCE_DIR}/third_party/yaml-cpp" "$ENV{LOCALAPPDATA}/MuSc/deps/yaml-cpp")
if(NOT yaml-cpp_FOUND)
    include(FetchContent)
    set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(yaml-cpp
        URL https://codeload.github.com/jbeder/yaml-cpp/tar.gz/refs/tags/0.8.0
        URL_HASH SHA256=fbe74bbdcee21d656715688706da3c8becfd946d92cd44705cc6098bb23b3a16)
    FetchContent_MakeAvailable(yaml-cpp)
endif()
add_executable(musc examples/musc_main.cpp configs/config.cpp datasets/datasets.cpp datasets/image.cpp
    models/backbone/vision.cpp models/modules/_LNAMD.cpp models/modules/_MSM.cpp
    models/modules/_RsCIN.cpp models/RsCIN_features/rescore.cpp models/musc.cpp utils/metrics.cpp utils/report.cpp)
target_include_directories(musc PRIVATE "." "${LIBTORCH_ROOT}/include" "${LIBTORCH_ROOT}/include/torch/csrc/api/include")
target_link_directories(musc PRIVATE "${LIBTORCH_ROOT}/lib")
target_compile_definitions(musc PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN UNICODE _UNICODE)
target_compile_options(musc PRIVATE /bigobj /utf-8 /permissive- /EHsc)
target_link_libraries(musc PRIVATE torch torch_cpu c10 yaml-cpp::yaml-cpp windowscodecs ole32)
target_precompile_headers(musc PRIVATE <ATen/ATen.h>)
file(GLOB nvrtc_dll "${LIBTORCH_ROOT}/lib/nvrtc64_*.dll")
file(GLOB nvrtc_builtins "${LIBTORCH_ROOT}/lib/nvrtc-builtins64_*.dll")
if(nvrtc_dll AND nvrtc_builtins)
    list(GET nvrtc_dll 0 nvrtc)
    list(GET nvrtc_builtins 0 builtins)
    add_executable(musc_compile_ptx scripts/compile_ptx.cpp)
    set(kernel "${CMAKE_CURRENT_SOURCE_DIR}/models/modules/_MSM_tiled.cu")
    set(kernel_header "${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/generated/musc_match_ptx.hpp")
    add_custom_command(OUTPUT "${kernel_header}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/generated"
        COMMAND musc_compile_ptx "${nvrtc}" "${builtins}" "${kernel}" "${kernel_header}"
        DEPENDS musc_compile_ptx "${kernel}" "${nvrtc}" "${builtins}"
        VERBATIM)
    target_sources(musc PRIVATE models/modules/_MSM_tiled.cpp "${kernel_header}")
    target_include_directories(musc PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/generated")
    target_compile_definitions(musc PRIVATE MUSC_TILED_CUDA)
endif()
add_custom_command(TARGET musc POST_BUILD COMMAND "${CMAKE_COMMAND}"
    "-Dsource=${LIBTORCH_ROOT}/lib" "-Ddest=$<TARGET_FILE_DIR:musc>" -P "${CMAKE_SOURCE_DIR}/scripts/runtime.cmake")
if(TARGET yaml-cpp::yaml-cpp)
    get_target_property(yaml_type yaml-cpp::yaml-cpp TYPE)
    if(yaml_type STREQUAL "SHARED_LIBRARY")
        add_custom_command(TARGET musc POST_BUILD COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "$<TARGET_FILE:yaml-cpp::yaml-cpp>" "$<TARGET_FILE_DIR:musc>")
    endif()
endif()
