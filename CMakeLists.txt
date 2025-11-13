cmake_minimum_required(VERSION 3.24)

project(fe_gpu
    VERSION 0.1.0
    DESCRIPTION "GPU-accelerated fixed-effects regression engine"
    LANGUAGES C CXX Fortran)

option(ENABLE_CUDA "Build CUDA acceleration support" ON)
option(ENABLE_TESTS "Build unit and integration tests" ON)

find_package(LAPACK REQUIRED)

if(ENABLE_CUDA)
    enable_language(CUDA OPTIONAL)
    set(CMAKE_CUDA_ARCHITECTURES 70 CACHE STRING "CUDA architectures" FORCE)
    if(NOT CMAKE_CUDA_COMPILER)
        message(WARNING "CUDA requested but no compiler found. Disabling CUDA support.")
        set(ENABLE_CUDA OFF CACHE BOOL "Build CUDA acceleration support" FORCE)
    endif()
endif()

add_subdirectory(src)

if(ENABLE_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
