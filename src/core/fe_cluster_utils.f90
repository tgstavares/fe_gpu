module fe_cluster_utils
    use iso_c_binding, only: c_int, c_long_long, c_ptr, c_loc
    use iso_fortran_env, only: int32, int64
    implicit none
    private

    public :: build_cluster_ids

    interface
        function c_fe_build_cluster_ids(fe_ids_ptr, ld_fe, n_obs, subset_dims_ptr, subset_sizes_ptr, subset_len, out_ids_ptr, &
                out_n_clusters_ptr) bind(C, name="fe_build_cluster_ids") result(status)
            import :: c_int, c_long_long, c_ptr
            type(c_ptr), value :: fe_ids_ptr
            integer(c_int), value :: ld_fe
            integer(c_long_long), value :: n_obs
            type(c_ptr), value :: subset_dims_ptr
            type(c_ptr), value :: subset_sizes_ptr
            integer(c_int), value :: subset_len
            type(c_ptr), value :: out_ids_ptr
            type(c_ptr), value :: out_n_clusters_ptr
            integer(c_int) :: status
        end function c_fe_build_cluster_ids
    end interface

contains

    subroutine build_cluster_ids(fe_ids, group_sizes, subset_dims, cluster_ids, n_clusters, status)
        integer(int32), intent(in), target :: fe_ids(:, :)
        integer, intent(in) :: group_sizes(:)
        integer, intent(in) :: subset_dims(:)
        integer(int32), intent(inout), target :: cluster_ids(:)
        integer, intent(out) :: n_clusters
        integer, intent(out) :: status
        integer(int32), allocatable, target :: dims32(:)
        integer(int32), allocatable, target :: dim_sizes32(:)
        integer(int32), target :: n_clusters32

        if (size(subset_dims) <= 0) then
            status = -1
            n_clusters = 0
            return
        end if
        if (size(cluster_ids) /= size(fe_ids, 2)) then
            status = -2
            n_clusters = 0
            return
        end if

        allocate(dims32(size(subset_dims)))
        allocate(dim_sizes32(size(subset_dims)))
        dims32 = int(subset_dims, int32)
        dim_sizes32 = int(group_sizes(subset_dims), int32)

        n_clusters32 = 0_int32
        status = c_fe_build_cluster_ids( &
            c_loc(fe_ids(1, 1)), &
            int(size(fe_ids, 1), kind=c_int), &
            int(size(fe_ids, 2), kind=c_long_long), &
            c_loc(dims32(1)), &
            c_loc(dim_sizes32(1)), &
            int(size(dims32), kind=c_int), &
            c_loc(cluster_ids(1)), &
            c_loc(n_clusters32))

        if (status == 0) then
            n_clusters = n_clusters32
        else
            n_clusters = 0
        end if

        deallocate(dims32, dim_sizes32)
    end subroutine build_cluster_ids

end module fe_cluster_utils
