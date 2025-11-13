add_executable(test_data_loader test_data_loader.f90)
target_link_libraries(test_data_loader PRIVATE fe_gpu_core fe_gpu_cuda)
target_include_directories(test_data_loader PRIVATE ${CMAKE_BINARY_DIR}/src/modules)

add_test(NAME data_loader COMMAND test_data_loader)

if(ENABLE_CUDA)
    add_executable(test_gpu_runtime test_gpu_runtime.f90)
    target_link_libraries(test_gpu_runtime PRIVATE fe_gpu_core fe_gpu_cuda)
    target_include_directories(test_gpu_runtime PRIVATE ${CMAKE_BINARY_DIR}/src/modules)
    add_test(NAME gpu_runtime COMMAND test_gpu_runtime)

    add_executable(test_gpu_demean test_gpu_demean.f90)
    target_link_libraries(test_gpu_demean PRIVATE fe_gpu_core fe_gpu_cuda)
    target_include_directories(test_gpu_demean PRIVATE ${CMAKE_BINARY_DIR}/src/modules)
    add_test(NAME gpu_demean COMMAND test_gpu_demean)

    add_executable(test_gpu_regression test_gpu_regression.f90)
    target_link_libraries(test_gpu_regression PRIVATE fe_gpu_core fe_gpu_cuda)
    target_include_directories(test_gpu_regression PRIVATE ${CMAKE_BINARY_DIR}/src/modules)
    add_test(NAME gpu_regression COMMAND test_gpu_regression)
endif()
