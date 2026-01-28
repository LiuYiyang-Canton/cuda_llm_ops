# ==============================================================================
# Author: Liu Yiyang
# Date:   2026-01-29
# Purpose:
# Build:
# ==============================================================================
cmake_minimum_required(VERSION 3.18)

set(cudaArchitecturesExplicit FALSE)
if(DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(cudaArchitecturesExplicit TRUE)
endif()

project(cuda_llm_ops LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT cudaArchitecturesExplicit)
    message(FATAL_ERROR "CMAKE_CUDA_ARCHITECTURES must be set explicitly before configuring. Example: -DCMAKE_CUDA_ARCHITECTURES=90")
endif()

find_package(CUDAToolkit REQUIRED)
find_package(OpenMP QUIET)

file(GLOB_RECURSE KERNEL_SOURCES CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cplusplus/*_kernel.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cplusplus/*_kernel.cpp
)

set(FAST_MATH_KERNEL_SOURCES ${KERNEL_SOURCES})
list(FILTER FAST_MATH_KERNEL_SOURCES INCLUDE REGEX ".*/[^/]+_kernel\\.cu$")
list(FILTER FAST_MATH_KERNEL_SOURCES EXCLUDE REGEX ".*/rope_kernel\\.cu$")

set(KERNEL_CPP_SOURCES ${KERNEL_SOURCES})
list(FILTER KERNEL_CPP_SOURCES INCLUDE REGEX ".*/[^/]+_kernel\\.cpp$")

add_library(cuda_ops_kernels STATIC
    ${KERNEL_SOURCES}
)

target_include_directories(cuda_ops_kernels PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include/cplusplus
)

target_compile_options(cuda_ops_kernels PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)

set_source_files_properties(${FAST_MATH_KERNEL_SOURCES} PROPERTIES
    COMPILE_OPTIONS "--use_fast_math"
)

set_source_files_properties(${KERNEL_CPP_SOURCES} PROPERTIES
    LANGUAGE CUDA
)

if(OpenMP_CXX_FOUND)
    target_compile_options(cuda_ops_kernels PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:${OpenMP_CXX_FLAGS}>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>
    )
    target_link_libraries(cuda_ops_kernels PRIVATE OpenMP::OpenMP_CXX)
endif()

file(GLOB EXAMPLE_TEST_SOURCES CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/examples/cplusplus/test_*.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/examples/cplusplus/test_*.cpp
)

set(TEST_CPP_SOURCES ${EXAMPLE_TEST_SOURCES})
list(FILTER TEST_CPP_SOURCES INCLUDE REGEX ".*/test_[^/]+\\.cpp$")

set(TEST_CUDA_SOURCES ${EXAMPLE_TEST_SOURCES})
list(FILTER TEST_CUDA_SOURCES INCLUDE REGEX ".*/test_[^/]+\\.cu$")

foreach(test_source ${EXAMPLE_TEST_SOURCES})
    get_filename_component(test_name ${test_source} NAME_WE)

    add_executable(${test_name} ${test_source})
    set_target_properties(${test_name} PROPERTIES
        OUTPUT_NAME ${test_name}
        SUFFIX ".o"
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/examples
    )

    target_include_directories(${test_name} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include/cplusplus
    )

    target_compile_options(${test_name} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-O3>
        $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    )

    target_link_libraries(${test_name} PRIVATE
        cuda_ops_kernels
        CUDA::cublas
        CUDA::cudart
    )

    if(OpenMP_CXX_FOUND)
        target_compile_options(${test_name} PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:${OpenMP_CXX_FLAGS}>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>
        )
        target_link_libraries(${test_name} PRIVATE OpenMP::OpenMP_CXX)
    endif()
endforeach()

set_source_files_properties(${TEST_CPP_SOURCES} PROPERTIES
    LANGUAGE CUDA
)

set_source_files_properties(${TEST_CUDA_SOURCES} PROPERTIES
    COMPILE_OPTIONS "--use_fast_math"
)

set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/examples/engram_golden.cpp PROPERTIES
    LANGUAGE CUDA
)
