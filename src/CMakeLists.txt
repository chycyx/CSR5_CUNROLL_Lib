# 把 .c 当作 C++ 源来编译
set(_src ${CMAKE_CURRENT_SOURCE_DIR}/csr5_cunroll.c)
set_source_files_properties(${_src} PROPERTIES LANGUAGE CXX)

add_library(csr5_cunroll STATIC ${_src})

target_include_directories(csr5_cunroll
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    $<INSTALL_INTERFACE:include>
)

# 使用 C++17
set_property(TARGET csr5_cunroll PROPERTY CXX_STANDARD 17)
set_property(TARGET csr5_cunroll PROPERTY CXX_STANDARD_REQUIRED ON)

# 编译优化
target_compile_options(csr5_cunroll PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native>
)

# OpenMP
if(CSR5_USE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(csr5_cunroll PUBLIC OpenMP::OpenMP_CXX)
    endif()
    if(OpenMP_C_FOUND)
        target_link_libraries(csr5_cunroll PUBLIC OpenMP::OpenMP_C)
    endif()
endif()
