# mmio 库（仅测试使用）
add_library(csr5_mmio STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/mmio.c
)
target_include_directories(csr5_mmio PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# 测试程序
set(CMAKE_CXX_STANDARD 17)
add_executable(csr5_cunroll_test
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp
)

target_link_libraries(csr5_cunroll_test
    PRIVATE
        csr5_cunroll
        csr5_mmio
)

target_include_directories(csr5_cunroll_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
