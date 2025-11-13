module fe_logging
    use iso_fortran_env, only: output_unit, error_unit
    implicit none
    private

    integer, parameter, public :: LOG_LEVEL_TRACE = 0
    integer, parameter, public :: LOG_LEVEL_DEBUG = 1
    integer, parameter, public :: LOG_LEVEL_INFO  = 2
    integer, parameter, public :: LOG_LEVEL_WARN  = 3
    integer, parameter, public :: LOG_LEVEL_ERROR = 4

    character(len=6), parameter :: level_labels(0:4) = (/ &
        'TRACE:', 'DEBUG:', '      ', 'WARN: ', 'ERROR:' /)

    integer :: current_threshold = LOG_LEVEL_INFO

    public :: set_log_threshold
    public :: log_message
    public :: log_trace, log_debug, log_info, log_warn, log_error
    public :: threshold_enabled

contains

    subroutine set_log_threshold(level)
        integer, intent(in) :: level
        current_threshold = max(LOG_LEVEL_TRACE, min(LOG_LEVEL_ERROR, level))
    end subroutine set_log_threshold

    logical function threshold_enabled(level)
        integer, intent(in) :: level
        threshold_enabled = level >= current_threshold
    end function threshold_enabled

    subroutine log_message(level, message)
        integer, intent(in)        :: level
        character(len=*), intent(in) :: message
        integer :: unit
        integer :: ilevel

        if (.not. threshold_enabled(level)) return

        if (level >= LOG_LEVEL_WARN) then
            unit = error_unit
        else
            unit = output_unit
        end if

        ilevel = max(LOG_LEVEL_TRACE, min(LOG_LEVEL_ERROR, level))
        write(unit, '(" ",A6,"  ",A)') level_labels(ilevel), trim(message)
    end subroutine log_message

    subroutine log_message_normal(level, message)
        integer, intent(in)        :: level
        character(len=*), intent(in) :: message
        integer :: unit
        integer :: ilevel

        if (.not. threshold_enabled(level)) return

        if (level >= LOG_LEVEL_WARN) then
            unit = error_unit
        else
            unit = output_unit
        end if

        ilevel = max(LOG_LEVEL_TRACE, min(LOG_LEVEL_ERROR, level))
        write(unit, '(A)') trim(message)
    end subroutine log_message_normal

    subroutine log_trace(message)
        character(len=*), intent(in) :: message
        call log_message(LOG_LEVEL_TRACE, message)
    end subroutine log_trace

    subroutine log_debug(message)
        character(len=*), intent(in) :: message
        call log_message(LOG_LEVEL_DEBUG, message)
    end subroutine log_debug

    subroutine log_info(message)
        character(len=*), intent(in) :: message
        call log_message_normal(LOG_LEVEL_INFO, message)
    end subroutine log_info

    subroutine log_warn(message)
        character(len=*), intent(in) :: message
        call log_message(LOG_LEVEL_WARN, message)
    end subroutine log_warn

    subroutine log_error(message)
        character(len=*), intent(in) :: message
        call log_message(LOG_LEVEL_ERROR, message)
    end subroutine log_error

end module fe_logging
