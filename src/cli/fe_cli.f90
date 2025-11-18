module fe_cli
    use iso_fortran_env, only: error_unit, int32
    use fe_config, only: fe_runtime_config, describe_config
    use fe_types, only: fe_dataset_header
    use fe_logging, only: log_info, log_warn
    implicit none
    private

    public :: parse_cli_arguments
    public :: apply_config_bindings

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
            case ('--config')
                idx = idx + 1
                if (idx > argc) call fail_option('--config requires a file path')
                call get_command_argument(idx, value)
                call load_config_file(cfg, trim(value))
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
            case ('--iv-z-cols')
                idx = idx + 1
                if (idx > argc) call fail_option('--iv-z-cols requires an instrument index list (1-based)')
                call get_command_argument(idx, value)
                call append_iv_z_columns(cfg, trim(value))
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
        call sort_iv_z_columns(cfg)

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
        write(error_unit, '(A)') '      --config <file>      Load options (data path, formula, etc.) from config file'
        write(error_unit, '(A)') '      --fe-tol <float>     Convergence tolerance for FE solver (default: 1e-6)'
        write(error_unit, '(A)') '      --fe-max-iters <int> Maximum FE iterations (default: 500)'
        write(error_unit, '(A)') '      --cluster-fe <list>  Cluster SEs by FE dimensions (comma-separated list, 1-based)'
        write(error_unit, '(A)') '      --iv-cols <list>     Comma-separated regressor indices treated as endogenous'
        write(error_unit, '(A)') '      --iv-z-cols <list>   Comma-separated instrument column indices (default: all)'
        write(error_unit, '(A)') '      --formula \"y ~ x1 x2 (x2 ~ z1 z2), fe(worker firm) cluster(worker)\"'
        write(error_unit, '(A)') '                             Estimate regression with named columns (config only)'
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
        if (allocated(cfg%cluster_name_targets)) deallocate(cfg%cluster_name_targets)

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

    subroutine sort_iv_z_columns(cfg)
        type(fe_runtime_config), intent(inout) :: cfg
        integer :: i, j
        integer(int32) :: key
        if (.not. allocated(cfg%iv_instrument_cols)) return
        do i = 2, size(cfg%iv_instrument_cols)
            key = cfg%iv_instrument_cols(i)
            j = i - 1
            do while (j >= 1 .and. cfg%iv_instrument_cols(j) > key)
                cfg%iv_instrument_cols(j + 1) = cfg%iv_instrument_cols(j)
                j = j - 1
            end do
            cfg%iv_instrument_cols(j + 1) = key
        end do
    end subroutine sort_iv_z_columns

    subroutine append_iv_z_columns(cfg, raw_value)
        type(fe_runtime_config), intent(inout) :: cfg
        character(len=*), intent(in) :: raw_value
        character(len=:), allocatable :: token, work
        integer :: start, comma_pos

        if (allocated(cfg%iv_instrument_names)) deallocate(cfg%iv_instrument_names)
        work = adjustl(trim(raw_value))
        if (len_trim(work) == 0) call fail_option('--iv-z-cols requires at least one column index')
        start = 1
        do
            comma_pos = index(work(start:), ',')
            if (comma_pos == 0) then
                token = trim(work(start:))
                call add_iv_z_column(cfg, token)
                exit
            else
                token = trim(work(start:start + comma_pos - 2))
                call add_iv_z_column(cfg, token)
                start = start + comma_pos
                if (start > len(work)) exit
            end if
        end do
    end subroutine append_iv_z_columns

    subroutine add_iv_z_column(cfg, token)
        type(fe_runtime_config), intent(inout) :: cfg
        character(len=*), intent(in) :: token
        integer :: value, ios
        integer(int32), allocatable :: tmp(:)

        if (len_trim(token) == 0) call fail_option('Empty token in --iv-z-cols list')
        read(token, *, iostat=ios) value
        if (ios /= 0 .or. value < 1) call fail_option('Invalid integer in --iv-z-cols list')

        if (.not. allocated(cfg%iv_instrument_cols)) then
            allocate(cfg%iv_instrument_cols(1))
            cfg%iv_instrument_cols(1) = value
            return
        end if

        if (size(cfg%iv_instrument_cols) == 0) then
            deallocate(cfg%iv_instrument_cols)
            allocate(cfg%iv_instrument_cols(1))
            cfg%iv_instrument_cols(1) = value
            return
        end if

        if (any(cfg%iv_instrument_cols == value)) return

        allocate(tmp(size(cfg%iv_instrument_cols) + 1))
        if (size(cfg%iv_instrument_cols) > 0) tmp(1:size(cfg%iv_instrument_cols)) = cfg%iv_instrument_cols
        tmp(size(tmp)) = value
        call move_alloc(tmp, cfg%iv_instrument_cols)
    end subroutine add_iv_z_column

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
        if (allocated(cfg%iv_regressor_names)) deallocate(cfg%iv_regressor_names)

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

    subroutine load_config_file(cfg, file_path)
        type(fe_runtime_config), intent(inout) :: cfg
        character(len=*), intent(in) :: file_path
        integer :: unit, ios, pos
        character(len=1024) :: line
        character(len=:), allocatable :: key, value
        logical :: exists

        inquire(file=file_path, exist=exists)
        if (.not. exists) then
            write(error_unit, '("Config file not found: ",A)') trim(file_path)
            stop 1
        end if

        open(newunit=unit, file=file_path, status='old', action='read', iostat=ios)
        if (ios /= 0) then
            write(error_unit, '("Unable to open config file: ",A)') trim(file_path)
            stop 1
        end if

        do
            read(unit, '(A)', iostat=ios) line
            if (ios /= 0) exit
            call strip_inline_comment(line)
            line = adjustl(trim(line))
            if (len_trim(line) == 0) cycle
            pos = index(line, '=')
            if (pos <= 0) cycle
            key = to_lower(adjustl(trim(line(:pos - 1))))
            value = trim(strip_quotes(adjustl(trim(line(pos + 1:)))))
            select case (trim(key))
            case ('data')
                cfg%data_path = trim(value)
            case ('fe_tol')
                read(value, *, iostat=ios) cfg%fe_tolerance
                if (ios /= 0) call fail_option('Invalid fe_tol value in config file')
            case ('fe_max_iters')
                read(value, *, iostat=ios) cfg%fe_max_iterations
                if (ios /= 0) call fail_option('Invalid fe_max_iters value in config file')
            case ('use_gpu')
                cfg%use_gpu = parse_logical(value)
            case ('verbose')
                cfg%verbose = parse_logical(value)
            case ('formula')
                call parse_formula_string(cfg, value)
            case default
                call log_warn('Ignoring unrecognized config key: ' // trim(key))
            end select
        end do

        close(unit)
    end subroutine load_config_file

    subroutine strip_inline_comment(text)
        character(len=*), intent(inout) :: text
        integer :: pos
        pos = index(text, '#')
        if (pos > 0) text = text(:pos - 1)
    end subroutine strip_inline_comment

    logical function parse_logical(raw) result(val)
        character(len=*), intent(in) :: raw
        character(len=:), allocatable :: lower
        lower = to_lower(trim(raw))
        select case (lower)
        case ('1', 'true', '.true.')
            val = .true.
        case default
            val = .false.
        end select
    end function parse_logical

    subroutine parse_formula_string(cfg, raw_formula)
        type(fe_runtime_config), intent(inout) :: cfg
        character(len=*), intent(in) :: raw_formula
        character(len=:), allocatable :: work, main_part, tail, rhs
        integer :: comma_pos, tilde_pos

        work = trim(raw_formula)
        if (len_trim(work) == 0) return
        call clear_name_array(cfg%iv_regressor_names)
        call clear_name_array(cfg%iv_instrument_names)
        call clear_name_array(cfg%cluster_name_targets)
        call clear_name_array(cfg%fe_name_targets)
        call clear_name_array(cfg%regressor_name_targets)

        comma_pos = index(work, ',')
        if (comma_pos > 0) then
            main_part = trim(work(:comma_pos - 1))
            tail = trim(work(comma_pos + 1:))
        else
            main_part = work
            tail = ''
        end if

        tilde_pos = index(main_part, '~')
        if (tilde_pos <= 0) then
            call log_warn('Formula is missing "~" separator; ignoring entry.')
            return
        end if

        call set_scalar_name(cfg%depvar_name, trim(main_part(:tilde_pos - 1)))
        rhs = trim(main_part(tilde_pos + 1:))
        call parse_rhs_terms(cfg, rhs)

        if (len_trim(tail) > 0) then
            call parse_formula_options(cfg, tail)
        end if
        cfg%formula_spec = work
    end subroutine parse_formula_string

    subroutine set_scalar_name(target, source)
        character(len=:), allocatable, intent(inout) :: target
        character(len=*), intent(in) :: source
        if (allocated(target)) deallocate(target)
        if (len_trim(source) == 0) return
        allocate(character(len=len_trim(source)) :: target)
        target = trim(source)
    end subroutine set_scalar_name

    subroutine parse_rhs_terms(cfg, rhs)
        type(fe_runtime_config), intent(inout) :: cfg
        character(len=*), intent(in) :: rhs
        integer :: i, n, start, depth
        character(len=:), allocatable :: block, token

        n = len_trim(rhs)
        i = 1
        do while (i <= n)
            if (rhs(i:i) == ' ') then
                i = i + 1
                cycle
            end if
            if (rhs(i:i) == '(') then
                start = i + 1
                depth = 1
                do while (i <= n .and. depth > 0)
                    i = i + 1
                    if (i > n) exit
                    if (rhs(i:i) == '(') then
                        depth = depth + 1
                    else if (rhs(i:i) == ')') then
                        depth = depth - 1
                    end if
                end do
                if (depth /= 0) then
                    call log_warn('Unmatched parenthesis in formula; ignoring trailing text.')
                    exit
                end if
                block = rhs(start:i - 1)
                call parse_iv_block(cfg, block)
                i = i + 1
            else
                start = i
                do while (i <= n .and. rhs(i:i) /= ' ' .and. rhs(i:i) /= '(')
                    i = i + 1
                end do
                token = rhs(start:i - 1)
                call append_name_unique(cfg%regressor_name_targets, trim(token))
            end if
        end do
    end subroutine parse_rhs_terms

    subroutine parse_iv_block(cfg, text)
        type(fe_runtime_config), intent(inout) :: cfg
        character(len=*), intent(in) :: text
        character(len=:), allocatable :: block, lhs, rhs
        integer :: tilde_pos
        character(len=:), allocatable :: tokens(:)
        integer :: i

        block = trim(text)
        tilde_pos = index(block, '~')
        if (tilde_pos <= 0) then
            call log_warn('Invalid IV specification (missing "~") in formula.')
            return
        end if
        lhs = trim(block(:tilde_pos - 1))
        rhs = trim(block(tilde_pos + 1:))

        call split_whitespace(lhs, tokens)
        do i = 1, size(tokens)
            call append_name(cfg%iv_regressor_names, tokens(i))
        end do
        if (allocated(tokens)) deallocate(tokens)

        call split_whitespace(rhs, tokens)
        do i = 1, size(tokens)
            call append_name(cfg%iv_instrument_names, tokens(i))
        end do
        if (allocated(tokens)) deallocate(tokens)
    end subroutine parse_iv_block

    subroutine parse_formula_options(cfg, tail)
        type(fe_runtime_config), intent(inout) :: cfg
        character(len=*), intent(in) :: tail
        integer :: pos
        character(len=:), allocatable :: lowered, block

        lowered = to_lower(tail)

        pos = index(lowered, 'fe(')
        if (pos > 0) then
            call extract_parenthesized(tail, pos + 2, block)
            call store_name_list(cfg%fe_name_targets, block)
        end if

        pos = index(lowered, 'cluster(')
        if (pos > 0) then
            call extract_parenthesized(tail, pos + 7, block)
            call store_name_list(cfg%cluster_name_targets, block)
        end if
    end subroutine parse_formula_options

    subroutine extract_parenthesized(text, start_pos, block)
        character(len=*), intent(in) :: text
        integer, intent(in) :: start_pos
        character(len=:), allocatable, intent(out) :: block
        integer :: depth, i, n

        depth = 1
        n = len(text)
        do i = start_pos + 1, n
            if (text(i:i) == '(') then
                depth = depth + 1
            else if (text(i:i) == ')') then
                depth = depth - 1
                if (depth == 0) exit
            end if
        end do
        if (depth /= 0) then
            block = ''
        else
            block = text(start_pos + 1:i - 1)
        end if
    end subroutine extract_parenthesized

    subroutine store_name_list(target, text)
        character(len=:), allocatable, intent(inout) :: target(:)
        character(len=*), intent(in) :: text
        character(len=:), allocatable :: tokens(:)
        integer :: i

        call clear_name_array(target)
        call split_whitespace(text, tokens)
        do i = 1, size(tokens)
            call append_name(target, tokens(i))
        end do
        if (allocated(tokens)) deallocate(tokens)
    end subroutine store_name_list

    subroutine split_whitespace(text, tokens)
        character(len=*), intent(in) :: text
        character(len=:), allocatable, intent(out) :: tokens(:)
        integer :: i, n, start, count, max_len, idx

        n = len_trim(text)
        count = 0
        max_len = 0
        i = 1
        do while (i <= n)
            if (text(i:i) == ' ') then
                i = i + 1
                cycle
            end if
            start = i
            do while (i <= n .and. text(i:i) /= ' ')
                i = i + 1
            end do
            count = count + 1
            max_len = max(max_len, i - start)
        end do
        if (count == 0) then
            allocate(character(len=1) :: tokens(0))
            return
        end if

        allocate(character(len=max_len) :: tokens(count))
        tokens = ''
        i = 1
        idx = 0
        do while (i <= n)
            if (text(i:i) == ' ') then
                i = i + 1
                cycle
            end if
            start = i
            do while (i <= n .and. text(i:i) /= ' ')
                i = i + 1
            end do
            idx = idx + 1
            call assign_string(tokens(idx), text(start:i - 1))
        end do
    end subroutine split_whitespace

    subroutine append_name(list, value)
        character(len=:), allocatable, intent(inout) :: list(:)
        character(len=*), intent(in) :: value
        character(len=:), allocatable :: tmp(:)
        integer :: n_old, new_len, i

        if (len_trim(value) == 0) return

        if (.not. allocated(list)) then
            allocate(character(len=len_trim(value)) :: list(1))
            list(1) = trim(value)
            return
        end if

        n_old = size(list)
        new_len = max(len(list(1)), len_trim(value))
        allocate(character(len=new_len) :: tmp(n_old + 1))
        tmp = ''
        do i = 1, n_old
            call assign_string(tmp(i), list(i))
        end do
        call assign_string(tmp(n_old + 1), trim(value))
        call move_alloc(tmp, list)
    end subroutine append_name

    subroutine append_name_unique(list, value)
        character(len=:), allocatable, intent(inout) :: list(:)
        character(len=*), intent(in) :: value
        if (len_trim(value) == 0) return
        if (contains_name(list, value)) return
        call append_name(list, value)
    end subroutine append_name_unique

    logical function contains_name(list, value) result(found)
        character(len=:), allocatable, intent(in) :: list(:)
        character(len=*), intent(in) :: value
        integer :: i
        found = .false.
        if (.not. allocated(list)) return
        do i = 1, size(list)
            if (len_trim(list(i)) == 0) cycle
            if (to_lower(trim(list(i))) == to_lower(trim(value))) then
                found = .true.
                return
            end if
        end do
    end function contains_name

    subroutine clear_name_array(list)
        character(len=:), allocatable, intent(inout) :: list(:)
        if (allocated(list)) deallocate(list)
    end subroutine clear_name_array

    subroutine assign_string(dest, source)
        character(len=*), intent(inout) :: dest
        character(len=*), intent(in) :: source
        integer :: copy_len
        dest = ''
        copy_len = min(len(dest), len_trim(source))
        if (copy_len > 0) dest(1:copy_len) = source(1:copy_len)
    end subroutine assign_string

    function to_lower(text) result(out)
        character(len=*), intent(in) :: text
        character(len=len(text)) :: out
        integer :: i
        do i = 1, len(text)
            select case (text(i:i))
            case ('A':'Z')
                out(i:i) = achar(iachar(text(i:i)) + 32)
            case default
                out(i:i) = text(i:i)
            end select
        end do
    end function to_lower

    function strip_quotes(text) result(out)
        character(len=*), intent(in) :: text
        character(len=len(text)) :: out
        integer :: n
        out = trim(text)
        n = len_trim(out)
        if (n >= 2) then
            if ((out(1:1) == '"' .and. out(n:n) == '"') .or. (out(1:1) == '''' .and. out(n:n) == '''')) then
                out = out(2:n - 1)
            end if
        end if
    end function strip_quotes

    subroutine apply_config_bindings(cfg, header)
        type(fe_runtime_config), intent(inout) :: cfg
        type(fe_dataset_header), intent(in) :: header
        integer(int32), allocatable :: mapped(:)

        if ((.not. allocated(cfg%cluster_fe_dims)) .or. size(cfg%cluster_fe_dims) == 0) then
            if (allocated(cfg%cluster_name_targets) .and. allocated(header%fe_names)) then
                call map_name_list(cfg%cluster_name_targets, header%fe_names, mapped)
                if (allocated(mapped)) then
                    if (allocated(cfg%cluster_fe_dims)) deallocate(cfg%cluster_fe_dims)
                    allocate(cfg%cluster_fe_dims(size(mapped)))
                    cfg%cluster_fe_dims = mapped
                end if
            end if
        end if

        if (allocated(cfg%fe_name_targets) .and. size(cfg%fe_name_targets) > 0 .and. &
            allocated(header%fe_names)) then
            call map_name_list(cfg%fe_name_targets, header%fe_names, mapped)
            if (allocated(mapped)) then
                if (allocated(cfg%fe_selection)) deallocate(cfg%fe_selection)
                allocate(cfg%fe_selection(size(mapped)))
                cfg%fe_selection = mapped
            end if
        end if

        if (allocated(cfg%regressor_name_targets) .and. size(cfg%regressor_name_targets) > 0 .and. &
            allocated(header%regressor_names)) then
            call map_name_list(cfg%regressor_name_targets, header%regressor_names, mapped)
            if (allocated(mapped)) then
                if (allocated(cfg%regressor_selection)) deallocate(cfg%regressor_selection)
                allocate(cfg%regressor_selection(size(mapped)))
                cfg%regressor_selection = mapped
            end if
        end if

        if (((.not. allocated(cfg%iv_regressors)) .or. size(cfg%iv_regressors) == 0) .and. &
            allocated(cfg%iv_regressor_names) .and. allocated(header%regressor_names)) then
            call map_name_list(cfg%iv_regressor_names, header%regressor_names, mapped)
            if (allocated(mapped)) then
                if (allocated(cfg%iv_regressors)) deallocate(cfg%iv_regressors)
                allocate(cfg%iv_regressors(size(mapped)))
                cfg%iv_regressors = mapped
                call sort_iv_columns(cfg)
            end if
        end if
        call remap_iv_with_selection(cfg)

        if (((.not. allocated(cfg%iv_instrument_cols)) .or. size(cfg%iv_instrument_cols) == 0) .and. &
            allocated(cfg%iv_instrument_names) .and. allocated(header%instrument_names)) then
            call map_name_list(cfg%iv_instrument_names, header%instrument_names, mapped)
            if (allocated(mapped)) then
                if (allocated(cfg%iv_instrument_cols)) deallocate(cfg%iv_instrument_cols)
                allocate(cfg%iv_instrument_cols(size(mapped)))
                cfg%iv_instrument_cols = mapped
                call sort_iv_z_columns(cfg)
            end if
        end if

        call sort_cluster_dimensions(cfg)
    end subroutine apply_config_bindings

    subroutine remap_iv_with_selection(cfg)
        type(fe_runtime_config), intent(inout) :: cfg
        integer(int32), allocatable :: remapped(:)
        integer :: i, pos, kept
        if (.not. allocated(cfg%regressor_selection)) return
        if (.not. allocated(cfg%iv_regressors)) return
        if (size(cfg%regressor_selection) == 0 .or. size(cfg%iv_regressors) == 0) return

        allocate(remapped(size(cfg%iv_regressors)))
        kept = 0
        do i = 1, size(cfg%iv_regressors)
            pos = find_position(cfg%regressor_selection, cfg%iv_regressors(i))
            if (pos > 0) then
                kept = kept + 1
                remapped(kept) = int(pos, int32)
            else
                call log_warn('Dropping IV regressor index not present in selected regressors.')
            end if
        end do
        if (kept == 0) then
            deallocate(cfg%iv_regressors)
            allocate(cfg%iv_regressors(0))
        else
            if (kept < size(remapped)) then
                cfg%iv_regressors = remapped(1:kept)
            else
                cfg%iv_regressors = remapped
            end if
        end if
        deallocate(remapped)
    end subroutine remap_iv_with_selection

    integer function find_position(selection, value) result(pos)
        integer(int32), intent(in) :: selection(:)
        integer(int32), intent(in) :: value
        integer :: i
        pos = 0
        do i = 1, size(selection)
            if (selection(i) == value) then
                pos = i
                return
            end if
        end do
    end function find_position

    subroutine map_name_list(requested, available, indices)
        character(len=*), intent(in) :: requested(:)
        character(len=*), intent(in), optional :: available(:)
        integer(int32), allocatable, intent(out) :: indices(:)
        integer :: i, found, count
        logical, allocatable :: mark(:)
        character(len=:), allocatable :: trimmed(:)

        if (.not. present(available)) then
            call log_warn('Dataset metadata missing; cannot resolve names.')
            return
        end if

        allocate(mark(size(requested)))
        mark = .false.
        count = 0
        do i = 1, size(requested)
            found = find_name_index(trim(requested(i)), available)
            if (found > 0) then
                mark(i) = .true.
                count = count + 1
            else
                call log_warn('Unable to match name "' // trim(requested(i)) // '" in dataset metadata.')
            end if
        end do
        if (count == 0) then
            deallocate(mark)
            return
        end if

        allocate(indices(count))
        count = 0
        do i = 1, size(requested)
            if (.not. mark(i)) cycle
            found = find_name_index(trim(requested(i)), available)
            if (found > 0) then
                count = count + 1
                indices(count) = int(found, int32)
            end if
        end do
        deallocate(mark)
    end subroutine map_name_list

    integer function find_name_index(name, candidates) result(idx)
        character(len=*), intent(in) :: name
        character(len=*), intent(in) :: candidates(:)
        integer :: i
        character(len=:), allocatable :: lhs, rhs
        idx = 0
        lhs = to_lower(trim(name))
        do i = 1, size(candidates)
            rhs = to_lower(trim(candidates(i)))
            if (lhs == rhs) then
                idx = i
                return
            end if
        end do
    end function find_name_index

end module fe_cli
