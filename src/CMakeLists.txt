set(FE_GPU_SOURCES
    core/fe_logging.f90
    core/fe_types.f90
    core/fe_config.f90
    core/fe_grouping.f90
    core/fe_cluster_utils.f90
    core/fe_pipeline.f90
    cli/fe_cli.f90
    io/fe_data_io.f90
    gpu/fe_gpu_runtime.f90
    gpu/fe_gpu_data.f90
    gpu/fe_gpu_demean.f90
    gpu/fe_gpu_linalg.f90
    core/fe_solver.f90
    core/cluster_id_builder.c
)

add_library(fe_gpu_core STATIC ${FE_GPU_SOURCES})
target_include_directories(fe_gpu_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_BINARY_DIR}/modules)
set_target_properties(fe_gpu_core PROPERTIES
    Fortran_MODULE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/modules)
target_compile_features(fe_gpu_core PRIVATE cxx_std_17)
target_link_libraries(fe_gpu_core PUBLIC LAPACK::LAPACK)

find_package(OpenMP)
if(OpenMP_Fortran_FOUND)
    target_link_libraries(fe_gpu_core PUBLIC OpenMP::OpenMP_Fortran)
endif()

if(ENABLE_CUDA)
    add_library(fe_gpu_cuda STATIC gpu/fe_gpu_runtime.cu gpu/fe_gpu_linalg.cu)
    target_include_directories(fe_gpu_cuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/gpu)
    set_target_properties(fe_gpu_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON)
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(fe_gpu_cuda PUBLIC CUDA::cudart CUDA::cublas)
else()
    add_library(fe_gpu_cuda STATIC gpu/fe_gpu_runtime_stub.c)
    target_include_directories(fe_gpu_cuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/gpu)
endif()

add_executable(fe_gpu src_main.f90)
target_link_libraries(fe_gpu PRIVATE fe_gpu_core fe_gpu_cuda)

set_target_properties(fe_gpu PROPERTIES
    Fortran_MODULE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/modules
)
