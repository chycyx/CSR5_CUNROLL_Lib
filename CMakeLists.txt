cmake_minimum_required(VERSION 3.18)

# 安装前缀
set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "Installation directory")

project(CSR5_CUNROLL VERSION 1.0.0 LANGUAGES C CXX)

# 选项
option(CSR5_USE_OPENMP "Enable OpenMP for CPU kernels" ON)
option(BUILD_TEST      "Build the test executable"     ON)

# 子目录
add_subdirectory(src)

if(BUILD_TEST)
    add_subdirectory(test)
endif()

# 安装库
install(TARGETS csr5_cunroll
    EXPORT csr5_cunrollTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

# 安装头文件
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
    DESTINATION include
)

# 导出 targets
install(EXPORT csr5_cunrollTargets
    FILE csr5_cunrollTargets.cmake
    NAMESPACE csr5_cunroll::
    DESTINATION lib/cmake/csr5_cunroll
)
# 新增：构建树导出（关键）
export(
    EXPORT csr5_cunrollTargets
    FILE "${CMAKE_CURRENT_BINARY_DIR}/csr5_cunrollTargets.cmake"
    NAMESPACE csr5_cunroll::
)

# 生成并安装 Config + Version
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/csr5_cunrollConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/csr5_cunrollConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/csr5_cunrollConfig.cmake"
    INSTALL_DESTINATION lib/cmake/csr5_cunroll
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/csr5_cunrollConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/csr5_cunrollConfigVersion.cmake"
    DESTINATION lib/cmake/csr5_cunroll
)
