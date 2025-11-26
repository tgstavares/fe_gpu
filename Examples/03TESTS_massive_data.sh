#!/usr/bin/env bash

# create a variable for a specific directory
DIRPROGS="/home/tgst/Desktop/Project_estimator/Project_estimator"

# generate dataset
$DIRPROGS/tools/generate_synthetic_data.py --workers 5000000 --firms 1000000 --periods 30 --extra-vars 6 --output data_massive.bin --parquet-output data_massive.parquet
#$DIRPROGS/tools/generate_synthetic_data.py --workers 5000 --firms 1000 --periods 30 --extra-vars 6 --output data_massive.bin --parquet-output data_massive.parquet

# reference
stata-mp -b 03TESTS_stata.do
cat 03TESTS_stata.log
rm 03TESTS_stata.log
julia test_julia1.jl

# regressions
$DIRPROGS/build/src/fe_gpu --data data_massive.bin --fe-tol 1e-8 --formula "wage ~ tenure sick_shock extra_1 extra_2 extra_3 extra_4 extra_5 extra_6, fe(worker firm time) cluster(worker firm time)"
$DIRPROGS/build/src/fe_gpu --data data_massive.bin --fe-tol 1e-8 --formula "wage ~ tenure sick_shock extra_1 extra_2 extra_3 extra_4 extra_5 extra_6, fe(worker firm time) cluster(worker firm time)" --fast

# remove files
rm data_massive*
