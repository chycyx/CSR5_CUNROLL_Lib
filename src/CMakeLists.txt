cmake_minimum_required(VERSION 3.18)
project(CSR5_Lib LANGUAGES C CXX)

# 开关
option(CSR5_USE_OPENMP "Enable OpenMP for CPU code" ON)
option(CSR5_ENABLE_OPT "Enable -O3/-march=native (or /O2 on MSVC)" ON)

# 头文件库（CPU 版 cunroll 是 header-only）
add_library(csr5_cunroll INTERFACE)
add_library(csr5::cunroll ALIAS csr5_cunroll)

# 头文件目录：当前在 CSR5_cunroll/src/ 下，指到 ../include
target_include_directories(csr5_cunroll INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    $<INSTALL_INTERFACE:include>
)

# 传递 C++ 标准
target_compile_features(csr5_cunroll INTERFACE cxx_std_14)

# OpenMP（可选）
if(CSR5_USE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(csr5_cunroll INTERFACE OpenMP::OpenMP_CXX)
        message(STATUS "[csr5_cunroll] OpenMP enabled")
    else()
        message(WARNING "[csr5_cunroll] OpenMP requested but not found; building without it")
    endif()
endif()

# 优化选项（可选）
if(CSR5_ENABLE_OPT)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
        target_compile_options(csr5_cunroll INTERFACE -O3 -march=native)
    elseif(MSVC)
        target_compile_options(csr5_cunroll INTERFACE /O2)
    endif()
    message(STATUS "[csr5_cunroll] Host optimization flags enabled")
endif()
