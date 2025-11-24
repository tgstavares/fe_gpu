module fe_cli
    use iso_fortran_env, only: error_unit, int32
    use fe_config, only: fe_runtime_config, describe_config, fe_formula_term, fe_formula_interaction
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
            case ('--cpu-threads')
                idx = idx + 1
                if (idx > argc) call fail_option('--cpu-threads requires an integer')
                call get_command_argument(idx, value)
                read(value, *, iostat=ios) cfg%cpu_threads
                if (ios /= 0 .or. cfg%cpu_threads < 0) call fail_option('Invalid integer for --cpu-threads')
            case ('--fast')
                cfg%fast_mode = .true.
            case ('--demean-cg')
                cfg%demean_cg = .true.
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
        write(error_unit, '(A)') '      --cpu-threads <int>  CPU threads for CPU fallbacks (default: OpenMP default)'
        write(error_unit, '(A)') '      --fast               Enable faster GPU clustering path (may change clustered SEs)'
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
            case ('cpu_threads')
                read(value, *, iostat=ios) cfg%cpu_threads
                if (ios /= 0 .or. cfg%cpu_threads < 0) call fail_option('Invalid cpu_threads value in config file')
            case ('use_gpu')
                cfg%use_gpu = parse_logical(value)
            case ('fast')
                cfg%fast_mode = parse_logical(value)
            case ('verbose')
                cfg%verbose = parse_logical(value)
            case ('formula')
                call parse_formula_string(cfg, value)
            case ('fe_names')
                call store_name_list(cfg%fe_override_names, normalize_list_text(value))
            case default
                call log_warn('Ignoring unrecognized config key: ' // trim(key))
            end select
        end do

        close(unit)
    end subroutine load_config_file

    subroutine strip_inline_comment(text)
        character(len=*), intent(inout) :: text
        integer :: i, n
        logical :: in_single, in_double
        character(len=1) :: ch, prev
        logical :: treat_as_comment
        integer :: first_nonspace

        n = len_trim(text)
        first_nonspace = 0
        do i = 1, len(text)
            if (text(i:i) /= ' ' .and. text(i:i) /= char(9)) then
                first_nonspace = i
                exit
            end if
        end do
        if (first_nonspace == 1 .and. n > 0) then
            if (text(1:1) == '#') then
                text = ''
                return
            end if
        end if

        in_single = .false.
        in_double = .false.
        do i = 1, len(text)
            ch = text(i:i)
            if (ch == '"' .and. .not. in_single) then
                in_double = .not. in_double
            else if (ch == '''' .and. .not. in_double) then
                in_single = .not. in_single
            else if ((ch == '#') .and. .not. in_single .and. .not. in_double) then
                treat_as_comment = .false.
                if (i == 1) then
                    treat_as_comment = .true.
                else
                    prev = text(i - 1:i - 1)
                    if (prev == ' ' .or. prev == char(9)) treat_as_comment = .true.
                end if
                if (treat_as_comment) then
                    if (i > 1) then
                        text = text(:i - 1)
                    else
                        text = ''
                    end if
                    exit
                end if
            end if
        end do
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
        if (allocated(cfg%formula_terms)) deallocate(cfg%formula_terms)
        allocate(cfg%formula_terms(0))
        if (allocated(cfg%formula_interactions)) deallocate(cfg%formula_interactions)
        allocate(cfg%formula_interactions(0))
        cfg%use_formula_design = .false.

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
        if (size(cfg%formula_terms) > 0 .or. size(cfg%formula_interactions) > 0) then
            cfg%use_formula_design = .true.
            cfg%formula_spec = work
        else
            cfg%formula_spec = ''
        end if
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
                token = trim(rhs(start:i - 1))
                if (.not. token_has_factor_notation(token)) call append_name_unique(cfg%regressor_name_targets, token)
                call process_formula_token(cfg, token)
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

    subroutine process_formula_token(cfg, token)
        type(fe_runtime_config), intent(inout) :: cfg
        character(len=*), intent(in) :: token
        if (len_trim(token) == 0) return
        if (index(token, '&&') > 0) then
            call handle_double_hash(cfg, trim(token), '&&')
        else if (index(token, '##') > 0) then
            call handle_double_hash(cfg, trim(token), '##')
        else if (index(token, '#') > 0) then
            call handle_interaction_token(cfg, trim(token))
        else
            call add_main_factor_token(cfg, trim(token))
        end if
    end subroutine process_formula_token

    subroutine handle_double_hash(cfg, token, delim)
        type(fe_runtime_config), intent(inout) :: cfg
        character(len=*), intent(in) :: token
        character(len=*), intent(in) :: delim
        character(len=:), allocatable :: parts(:)
        type(fe_formula_term), allocatable :: subset(:)
        integer :: n, mask, subset_size, idx, j

        call split_token_by(token, delim, parts)
        n = size(parts)
        if (n <= 0) return
        do mask = 1, ishft(1, n) - 1
            subset_size = popcnt(mask)
            allocate(subset(subset_size))
            idx = 0
        do j = 1, n
            if (iand(mask, ishft(1, j - 1)) /= 0) then
                idx = idx + 1
                subset(idx) = parse_factor_from_token(trim(parts(j)))
                if (subset(idx)%is_categorical) cfg%formula_has_categorical = .true.
            end if
        end do
            if (subset_size == 1) then
                call add_formula_main_term(cfg, subset(1))
            else
                call add_formula_interaction(cfg, subset)
            end if
            deallocate(subset)
        end do
    end subroutine handle_double_hash

    subroutine handle_interaction_token(cfg, token)
        type(fe_runtime_config), intent(inout) :: cfg
        character(len=*), intent(in) :: token
        character(len=:), allocatable :: parts(:)
        type(fe_formula_term), allocatable :: factors(:)
        integer :: n, j

        call split_token_by(token, '#', parts)
        n = size(parts)
        if (n <= 1) then
            if (n == 1) call add_main_factor_token(cfg, trim(parts(1)))
            return
        end if
        allocate(factors(n))
        do j = 1, n
            factors(j) = parse_factor_from_token(trim(parts(j)))
            if (factors(j)%is_categorical) cfg%formula_has_categorical = .true.
        end do
        call add_formula_interaction(cfg, factors)
        deallocate(factors)
    end subroutine handle_interaction_token

    subroutine add_main_factor_token(cfg, token)
        type(fe_runtime_config), intent(inout) :: cfg
        character(len=*), intent(in) :: token
        type(fe_formula_term) :: term
        term = parse_factor_from_token(trim(token))
        call add_formula_main_term(cfg, term)
    end subroutine add_main_factor_token

    function parse_factor_from_token(text) result(term)
        character(len=*), intent(in) :: text
        type(fe_formula_term) :: term
        character(len=:), allocatable :: lowered
        lowered = trim(text)
        if (len(lowered) >= 2 .and. lowered(1:2) == 'i.') then
            term%is_categorical = .true.
            term%name = trim(lowered(3:))
        else
            term%is_categorical = .false.
            term%name = lowered
        end if
    end function parse_factor_from_token

    subroutine add_formula_main_term(cfg, term)
        type(fe_runtime_config), intent(inout) :: cfg
        type(fe_formula_term), intent(in) :: term
        type(fe_formula_term), allocatable :: tmp(:)
        integer :: i, old_size
        if (len_trim(term%name) == 0) return
        if (term%is_categorical) cfg%formula_has_categorical = .true.
        if (.not. allocated(cfg%formula_terms)) then
            allocate(cfg%formula_terms(0))
        end if
        do i = 1, size(cfg%formula_terms)
            if (terms_equal(cfg%formula_terms(i), term)) return
        end do
        old_size = size(cfg%formula_terms)
        allocate(tmp(old_size + 1))
        if (old_size > 0) tmp(1:old_size) = cfg%formula_terms
        tmp(old_size + 1) = term
        call move_alloc(tmp, cfg%formula_terms)
    end subroutine add_formula_main_term

    subroutine add_formula_interaction(cfg, factors)
        type(fe_runtime_config), intent(inout) :: cfg
        type(fe_formula_term), intent(in) :: factors(:)
        type(fe_formula_interaction), allocatable :: tmp(:)
        integer :: i, old_size
        if (size(factors) <= 1) return
        if (.not. allocated(cfg%formula_interactions)) then
            allocate(cfg%formula_interactions(0))
        end if
        do i = 1, size(cfg%formula_interactions)
            if (interactions_equal(cfg%formula_interactions(i), factors)) return
        end do
        old_size = size(cfg%formula_interactions)
        allocate(tmp(old_size + 1))
        if (old_size > 0) tmp(1:old_size) = cfg%formula_interactions
        allocate(tmp(old_size + 1)%factors(size(factors)))
        tmp(old_size + 1)%factors = factors
        call move_alloc(tmp, cfg%formula_interactions)
    end subroutine add_formula_interaction

    logical function terms_equal(a, b) result(equal)
        type(fe_formula_term), intent(in) :: a, b
        equal = (a%is_categorical .eqv. b%is_categorical) .and. &
            (to_lower(trim(a%name)) == to_lower(trim(b%name)))
    end function terms_equal

    logical function interactions_equal(interaction, factors) result(equal)
        type(fe_formula_interaction), intent(in) :: interaction
        type(fe_formula_term), intent(in) :: factors(:)
        integer :: i

        equal = .false.
        if (.not. allocated(interaction%factors)) return
        if (size(interaction%factors) /= size(factors)) return
        do i = 1, size(factors)
            if (.not. terms_equal(interaction%factors(i), factors(i))) return
        end do
        equal = .true.
    end function interactions_equal

    subroutine split_token_by(text, sep, tokens)
        character(len=*), intent(in) :: text
        character(len=*), intent(in) :: sep
        character(len=:), allocatable, intent(out) :: tokens(:)
        character(len=:), allocatable :: temp(:)
        integer :: start, pos, len_sep, n

        len_sep = len(sep)
        n = len_trim(text)
        start = 1
        allocate(character(len=1) :: temp(0))
        do while (start <= n)
            pos = index(text(start:), sep)
            if (pos == 0) then
                call append_name(temp, trim(text(start:)))
                exit
            else
                call append_name(temp, trim(text(start:start + pos - 2)))
                start = start + pos - 1 + len_sep
                if (start > n) exit
            end if
        end do
        call move_alloc(temp, tokens)
    end subroutine split_token_by

    logical function token_has_factor_notation(token) result(flag)
        character(len=*), intent(in) :: token
        character(len=:), allocatable :: work
        work = trim(token)
        flag = .false.
        if (len(work) >= 2) then
            if (work(1:2) == 'i.') then
                flag = .true.
                return
            end if
        end if
        if (index(work, '#') > 0) flag = .true.
    end function token_has_factor_notation

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

    function normalize_list_text(text) result(out)
        character(len=*), intent(in) :: text
        character(len=len(text)) :: out
        integer :: i

        out = text
        do i = 1, len(out)
            if (out(i:i) == ',') out(i:i) = ' '
        end do
    end function normalize_list_text

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

    subroutine copy_name_array(source, target)
        character(len=*), intent(in) :: source(:)
        character(len=:), allocatable, intent(inout) :: target(:)
        character(len=:), allocatable :: tmp(:)
        integer :: max_len, i

        if (size(source) == 0) then
            if (allocated(target)) deallocate(target)
            allocate(character(len=1) :: target(0))
            return
        end if

        max_len = 0
        do i = 1, size(source)
            max_len = max(max_len, len_trim(source(i)))
        end do
        if (max_len <= 0) max_len = 1
        allocate(character(len=max_len) :: tmp(size(source)))
        tmp = ''
        do i = 1, size(source)
            call assign_string(tmp(i), trim(source(i)))
        end do
        if (allocated(target)) deallocate(target)
        call move_alloc(tmp, target)
    end subroutine copy_name_array

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
        type(fe_dataset_header), intent(inout) :: header
        integer(int32), allocatable :: mapped(:)

        call apply_fe_name_overrides(cfg, header)

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

        if ((.not. cfg%use_formula_design) .and. allocated(cfg%regressor_name_targets) .and. &
            size(cfg%regressor_name_targets) > 0 .and. allocated(header%regressor_names)) then
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

    subroutine apply_fe_name_overrides(cfg, header)
        type(fe_runtime_config), intent(in) :: cfg
        type(fe_dataset_header), intent(inout) :: header
        integer :: count
        character(len=32) :: buf_count, buf_dims

        if (.not. allocated(cfg%fe_override_names)) return
        count = size(cfg%fe_override_names)
        if (count == 0) return
        if (header%n_fe <= 0) then
            call log_warn('Ignoring fe_names override because dataset has no fixed-effect dimensions.')
            return
        end if
        if (count /= header%n_fe) then
            write(buf_count, '(I0)') count
            write(buf_dims, '(I0)') header%n_fe
            call log_warn('Ignoring fe_names override because it lists ' // trim(buf_count) // &
                ' entries but the dataset contains ' // trim(buf_dims) // ' FE dimensions.')
            return
        end if
        call copy_name_array(cfg%fe_override_names, header%fe_names)
    end subroutine apply_fe_name_overrides

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
