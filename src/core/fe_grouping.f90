module fe_grouping
    use iso_fortran_env, only: int32
    use fe_logging, only: log_error
    implicit none
    private

    public :: compute_fe_group_sizes

contains

    subroutine compute_fe_group_sizes(fe_ids, group_sizes)
        integer(int32), intent(in) :: fe_ids(:, :)
        integer, allocatable, intent(out) :: group_sizes(:)
        integer :: n_dims
        integer :: d, i
        integer(int32) :: max_id

        n_dims = size(fe_ids, 1)
        allocate(group_sizes(n_dims))

        do d = 1, n_dims
            if (size(fe_ids, 2) == 0) then
                max_id = 0
            else
                if (minval(fe_ids(d, :)) < 1_int32) then
                    call fail('FE indices must be positive and contiguous (dimension ' // trim(to_string(d)) // ')')
                end if
                max_id = 0
!$omp parallel do default(shared) private(i) reduction(max:max_id)
                do i = 1, size(fe_ids, 2)
                    max_id = max(max_id, fe_ids(d, i))
                end do
!$omp end parallel do
            end if
            group_sizes(d) = max(0, max_id)
        end do
    end subroutine compute_fe_group_sizes

    subroutine fail(message)
        character(len=*), intent(in) :: message
        call log_error(trim(message))
        error stop 1
    end subroutine fail

    function to_string(value) result(buffer)
        integer, intent(in) :: value
        character(len=16) :: buffer
        write(buffer, '(I0)') value
    end function to_string

end module fe_grouping
