module fe_solver
    use iso_fortran_env, only: int32, real64
    use fe_logging, only: log_info
    implicit none
    private

    public :: chol_solve_and_invert

    interface
        subroutine dpotrf(uplo, n, a, lda, info) bind(C, name="dpotrf_")
            import :: int32, real64
            character(len=1), intent(in) :: uplo
            integer(int32), intent(in) :: n, lda
            real(real64), intent(inout) :: a(lda, *)
            integer(int32), intent(out) :: info
        end subroutine dpotrf

        subroutine dpotrs(uplo, n, nrhs, a, lda, b, ldb, info) bind(C, name="dpotrs_")
            import :: int32, real64
            character(len=1), intent(in) :: uplo
            integer(int32), intent(in) :: n, nrhs, lda, ldb
            real(real64), intent(in) :: a(lda, *)
            real(real64), intent(inout) :: b(ldb, *)
            integer(int32), intent(out) :: info
        end subroutine dpotrs

        subroutine dpotri(uplo, n, a, lda, info) bind(C, name="dpotri_")
            import :: int32, real64
            character(len=1), intent(in) :: uplo
            integer(int32), intent(in) :: n, lda
            real(real64), intent(inout) :: a(lda, *)
            integer(int32), intent(out) :: info
        end subroutine dpotri
    end interface

contains

    subroutine chol_solve_and_invert(Q, b, beta, Q_inv, info)
        real(real64), intent(inout) :: Q(:, :)
        real(real64), intent(inout) :: b(:)
        real(real64), intent(out) :: beta(:)
        real(real64), intent(out) :: Q_inv(:, :)
        integer, intent(out) :: info
        integer :: n
        integer :: info_inv
        integer :: i, j

        n = size(b)
        beta = b

        call dpotrf('U', n, Q, n, info)
        if (info /= 0) return

        call dpotrs('U', n, 1, Q, n, beta, n, info)
        if (info /= 0) return

        call log_info('Solved normal equations with Cholesky.')

        Q_inv = Q
        call dpotri('U', n, Q_inv, n, info_inv)
        if (info_inv /= 0) then
            info = info_inv
            return
        end if

        do j = 1, n
            do i = j + 1, n
                Q_inv(i, j) = Q_inv(j, i)
            end do
        end do
    end subroutine chol_solve_and_invert

end module fe_solver
