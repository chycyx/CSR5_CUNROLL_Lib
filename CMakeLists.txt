cmake_minimum_required(VERSION 3.16)
project(csr5_cunroll LANGUAGES C CXX)

option(CSR5_USE_OPENMP  "Enable OpenMP for CPU kernels" ON)
option(CSR5_ENABLE_OPT  "Enable -O3/-march=native"      ON)

# 1) 头文件库
add_library(csr5_cunroll INTERFACE)
add_library(csr5::cunroll ALIAS csr5_cunroll)

target_include_directories(csr5_cunroll INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>
  $<INSTALL_INTERFACE:include>
)
target_compile_features(csr5_cunroll INTERFACE cxx_std_14)

if(CSR5_ENABLE_OPT)
  target_compile_options(csr5_cunroll INTERFACE -O3 -march=native)
endif()

if(CSR5_USE_OPENMP)
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(csr5_cunroll INTERFACE OpenMP::OpenMP_CXX)
  endif()
endif()

# 2) mmio.c 小静态库
add_library(csr5_mmio STATIC mmio.c)
target_include_directories(csr5_mmio PUBLIC ${CMAKE_CURRENT_LIST_DIR})
if(UNIX AND NOT APPLE)
  target_link_libraries(csr5_mmio PUBLIC m)
endif()

target_link_libraries(csr5_cunroll INTERFACE csr5_mmio)

# 3) 安装 & find_package
include(CMakePackageConfigHelpers)
set(PKG_NAME   csr5_cunroll)
set(PKG_EXPORT ${PKG_NAME}Targets)
set(PKG_DIR    lib/cmake/${PKG_NAME})

install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/
        DESTINATION include
        FILES_MATCHING PATTERN "*.h"
        PATTERN "build" EXCLUDE
        PATTERN "CMakeFiles" EXCLUDE
        PATTERN "CMakeLists.txt" EXCLUDE
        PATTERN "cmake" EXCLUDE
        PATTERN "Makefile" EXCLUDE
        PATTERN "main.cpp" EXCLUDE)

install(TARGETS csr5_mmio csr5_cunroll
        EXPORT ${PKG_EXPORT}
        ARCHIVE DESTINATION lib
        INCLUDES DESTINATION include)

install(EXPORT ${PKG_EXPORT}
        NAMESPACE csr5::
        FILE ${PKG_NAME}Targets.cmake
        DESTINATION ${PKG_DIR})

configure_package_config_file(
  "${CMAKE_CURRENT_LIST_DIR}/cmake/${PKG_NAME}Config.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/${PKG_NAME}Config.cmake"
  INSTALL_DESTINATION ${PKG_DIR}
)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/${PKG_NAME}Config.cmake"
        DESTINATION ${PKG_DIR})
