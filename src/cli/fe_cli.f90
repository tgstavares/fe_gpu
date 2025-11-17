module fe_cli
    use iso_fortran_env, only: error_unit, int32
    use fe_config, only: fe_runtime_config, describe_config
    use fe_logging, only: log_info
    implicit none
    private

    public :: parse_cli_arguments

contains

    subroutine parse_cli_arguments(cfg)
        type(fe_runtime_config), intent(inout) :: cfg
        integer :: argc, idx, ios
        character(len=512) :: arg
        character(len=512) :: value

        argc = command_argument_count()
        idx = 1

        do while (idx <= argc)
            call get_command_argument(idx, arg)
            select case (trim(arg))
            case ('-h', '--help')
                call print_usage()
                stop 0
            case ('-d', '--data')
                idx = idx + 1
                if (idx > argc) call fail_option('--data requires a file path')
                call get_command_argument(idx, value)
                cfg%data_path = trim(value)
            case ('--fe-tol')
                idx = idx + 1
                if (idx > argc) call fail_option('--fe-tol requires a numeric value')
                call get_command_argument(idx, value)
                read(value, *, iostat=ios) cfg%fe_tolerance
                if (ios /= 0) call fail_option('Invalid numeric value for --fe-tol')
            case ('--fe-max-iters')
                idx = idx + 1
                if (idx > argc) call fail_option('--fe-max-iters requires an integer')
                call get_command_argument(idx, value)
                read(value, *, iostat=ios) cfg%fe_max_iterations
                if (ios /= 0) call fail_option('Invalid integer for --fe-max-iters')
            case ('--cluster-fe')
                idx = idx + 1
                if (idx > argc) call fail_option('--cluster-fe requires a fixed-effect dimension index (1-based)')
                call get_command_argument(idx, value)
                call append_cluster_dimensions(cfg, trim(value))
            case ('--iv-cols')
                idx = idx + 1
                if (idx > argc) call fail_option('--iv-cols requires a regressor index list (1-based)')
                call get_command_argument(idx, value)
                call append_iv_columns(cfg, trim(value))
            case ('--cpu-only')
                cfg%use_gpu = .false.
            case ('--gpu')
                cfg%use_gpu = .true.
            case ('--verbose')
                cfg%verbose = .true.
            case default
                write(error_unit, '("Unrecognized option: ",A)') trim(arg)
                call print_usage()
                stop 1
            end select
            idx = idx + 1
        end do

        if (.not. allocated(cfg%data_path)) cfg%data_path = 'data.bin'
        call sort_cluster_dimensions(cfg)
        call sort_iv_columns(cfg)

        call log_info('Runtime configuration -> ' // describe_config(cfg))
    end subroutine parse_cli_arguments

    subroutine fail_option(message)
        character(len=*), intent(in) :: message
        write(error_unit, '(A)') trim(message)
        call print_usage()
        stop 1
    end subroutine fail_option

    subroutine print_usage()
        write(error_unit, '(A)') 'Usage: fe_gpu [options]'
        write(error_unit, '(A)') '  -d, --data <path>        Path to binary dataset (default: data.bin)'
        write(error_unit, '(A)') '      --fe-tol <float>     Convergence tolerance for FE solver (default: 1e-6)'
        write(error_unit, '(A)') '      --fe-max-iters <int> Maximum FE iterations (default: 500)'
        write(error_unit, '(A)') '      --cluster-fe <list>  Cluster SEs by FE dimensions (comma-separated list, 1-based)'
        write(error_unit, '(A)') '      --iv-cols <list>     Comma-separated regressor indices treated as endogenous'
        write(error_unit, '(A)') '      --cpu-only           Disable GPU acceleration'
        write(error_unit, '(A)') '      --gpu                Force GPU usage when available'
        write(error_unit, '(A)') '      --verbose            Enable verbose logging'
        write(error_unit, '(A)') '  -h, --help               Show this help message'
    end subroutine print_usage

    subroutine append_cluster_dimensions(cfg, raw_value)
        type(fe_runtime_config), intent(inout) :: cfg
        character(len=*), intent(in) :: raw_value
        character(len=:), allocatable :: work
        integer :: start, comma_pos
        character(len=:), allocatable :: token

        work = adjustl(trim(raw_value))
        if (len_trim(work) == 0) call fail_option('--cluster-fe requires at least one dimension index')

        start = 1
        do
            comma_pos = index(work(start:), ',')
            if (comma_pos == 0) then
                token = trim(work(start:))
                call add_cluster_dim(cfg, token)
                exit
            else
                token = trim(work(start:start + comma_pos - 2))
                call add_cluster_dim(cfg, token)
                start = start + comma_pos
                if (start > len(work)) exit
            end if
        end do
    end subroutine append_cluster_dimensions

    subroutine add_cluster_dim(cfg, token)
        type(fe_runtime_config), intent(inout) :: cfg
        character(len=*), intent(in) :: token
        integer :: value, ios
        integer(int32), allocatable :: tmp(:)

        if (len_trim(token) == 0) call fail_option('Empty token in --cluster-fe list')
        read(token, *, iostat=ios) value
        if (ios /= 0 .or. value < 1) call fail_option('Invalid integer in --cluster-fe list')

        if (.not. allocated(cfg%cluster_fe_dims)) then
            allocate(cfg%cluster_fe_dims(1))
            cfg%cluster_fe_dims(1) = value
            return
        end if

        if (size(cfg%cluster_fe_dims) == 0) then
            deallocate(cfg%cluster_fe_dims)
            allocate(cfg%cluster_fe_dims(1))
            cfg%cluster_fe_dims(1) = value
            return
        end if

        if (any(cfg%cluster_fe_dims == value)) return

        allocate(tmp(size(cfg%cluster_fe_dims) + 1))
        if (size(cfg%cluster_fe_dims) > 0) tmp(1:size(cfg%cluster_fe_dims)) = cfg%cluster_fe_dims
        tmp(size(tmp)) = value
        call move_alloc(tmp, cfg%cluster_fe_dims)
    end subroutine add_cluster_dim

    subroutine sort_cluster_dimensions(cfg)
        type(fe_runtime_config), intent(inout) :: cfg
        integer :: i, j
        integer(int32) :: key

        if (.not. allocated(cfg%cluster_fe_dims)) return
        do i = 2, size(cfg%cluster_fe_dims)
            key = cfg%cluster_fe_dims(i)
            j = i - 1
            do while (j >= 1 .and. cfg%cluster_fe_dims(j) > key)
                cfg%cluster_fe_dims(j + 1) = cfg%cluster_fe_dims(j)
                j = j - 1
            end do
            cfg%cluster_fe_dims(j + 1) = key
        end do
    end subroutine sort_cluster_dimensions

    subroutine sort_iv_columns(cfg)
        type(fe_runtime_config), intent(inout) :: cfg
        integer :: i, j
        integer(int32) :: key
        if (.not. allocated(cfg%iv_regressors)) return
        do i = 2, size(cfg%iv_regressors)
            key = cfg%iv_regressors(i)
            j = i - 1
            do while (j >= 1 .and. cfg%iv_regressors(j) > key)
                cfg%iv_regressors(j + 1) = cfg%iv_regressors(j)
                j = j - 1
            end do
            cfg%iv_regressors(j + 1) = key
        end do
    end subroutine sort_iv_columns

    subroutine append_iv_columns(cfg, raw_value)
        type(fe_runtime_config), intent(inout) :: cfg
        character(len=*), intent(in) :: raw_value
        character(len=:), allocatable :: token, work
        integer :: start, comma_pos

        work = adjustl(trim(raw_value))
        if (len_trim(work) == 0) call fail_option('--iv-cols requires at least one column index')
        start = 1
        do
            comma_pos = index(work(start:), ',')
            if (comma_pos == 0) then
                token = trim(work(start:))
                call add_iv_column(cfg, token)
                exit
            else
                token = trim(work(start:start + comma_pos - 2))
                call add_iv_column(cfg, token)
                start = start + comma_pos
                if (start > len(work)) exit
            end if
        end do
    end subroutine append_iv_columns

    subroutine add_iv_column(cfg, token)
        type(fe_runtime_config), intent(inout) :: cfg
        character(len=*), intent(in) :: token
        integer :: value, ios
        integer(int32), allocatable :: tmp(:)

        if (len_trim(token) == 0) call fail_option('Empty token in --iv-cols list')
        read(token, *, iostat=ios) value
        if (ios /= 0 .or. value < 1) call fail_option('Invalid integer in --iv-cols list')

        if (.not. allocated(cfg%iv_regressors)) then
            allocate(cfg%iv_regressors(1))
            cfg%iv_regressors(1) = value
            return
        end if

        if (size(cfg%iv_regressors) == 0) then
            deallocate(cfg%iv_regressors)
            allocate(cfg%iv_regressors(1))
            cfg%iv_regressors(1) = value
            return
        end if

        if (any(cfg%iv_regressors == value)) return

        allocate(tmp(size(cfg%iv_regressors) + 1))
        if (size(cfg%iv_regressors) > 0) tmp(1:size(cfg%iv_regressors)) = cfg%iv_regressors
        tmp(size(tmp)) = value
        call move_alloc(tmp, cfg%iv_regressors)
    end subroutine add_iv_column

end module fe_cli
