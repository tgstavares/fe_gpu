program test_gpu_runtime
    use iso_c_binding, only: c_loc, c_ptr
    use iso_fortran_env, only: real64, int64
    use fe_logging, only: set_log_threshold, LOG_LEVEL_ERROR
    use fe_gpu_runtime, only: fe_gpu_context, fe_device_buffer, fe_gpu_initialize, fe_gpu_finalize, &
        fe_device_alloc, fe_device_free, fe_memcpy_htod, fe_memcpy_dtoh, fe_gpu_backend_available
    implicit none

    type(fe_gpu_context) :: ctx
    type(fe_device_buffer) :: buffer
    real(real64), target :: host_values(4)
    real(real64), target :: roundtrip(4)
    integer :: i
    integer(int64) :: byte_count

    call set_log_threshold(LOG_LEVEL_ERROR)

    if (.not. fe_gpu_backend_available()) then
        print *, 'GPU backend unavailable; skipping runtime test.'
        stop 0
    end if

    do i = 1, size(host_values)
        host_values(i) = real(i, kind=real64)
        roundtrip(i) = 0.0_real64
    end do

    call fe_gpu_initialize(ctx)

    byte_count = int(size(host_values), int64) * int(storage_size(host_values(1)) / 8, int64)
    call fe_device_alloc(buffer, byte_count)

    call fe_memcpy_htod(buffer, c_loc(host_values(1)))
    call fe_memcpy_dtoh(c_loc(roundtrip(1)), buffer)

    if (maxval(abs(host_values - roundtrip)) > 1.0e-12_real64) then
        print *, 'Roundtrip copy mismatch between host and device buffers.'
        call fe_device_free(buffer)
        call fe_gpu_finalize(ctx)
        stop 1
    end if

    call fe_device_free(buffer)
    call fe_gpu_finalize(ctx)

    print *, 'GPU runtime test passed.'
end program test_gpu_runtime
