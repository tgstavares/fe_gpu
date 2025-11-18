#!/usr/bin/env bash

# create a variable for a specific directory
DIRPROGS="/home/tgst/Desktop/Project_estimator/Project_estimator"

# Execute a program located in that directory
julia $DIRPROGS/tools/run_fixedeffects_gpu.jl --data nlsw_test.parquet --cluster-fe 1,2,3 --y ln_wage --x hours,ttl_exp,union --fe idcode,occ_code,year --iv-cols 1 --iv-vars tenure --tol 1e-8 --method cuda

# End of script
