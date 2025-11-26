#!/usr/bin/env bash

# create a variable for a specific directory
DIRPROGS="/home/tgst/Desktop/Project_estimator/Project_estimator"

# generate dataset
$DIRPROGS/tools/generate_synthetic_data.py --workers 5000000 --firms 1000000 --periods 30 --extra-vars 6 --output data_massive.bin --parquet-output data_massive.parquet

# regressions
$DIRPROGS/build/src/fe_gpu --data data_massive.bin --fe-tol 1e-8 --formula "wage ~ tenure sick_shock extra_1 extra_2 extra_3 extra_4 extra_5 extra_6, fe(worker firm time) cluster(worker firm time)"
$DIRPROGS/build/src/fe_gpu --data data_massive.bin --fe-tol 1e-8 --formula "wage ~ tenure sick_shock extra_1 extra_2 extra_3 extra_4 extra_5 extra_6, fe(worker firm time) cluster(worker firm time)" --fast

# julia alternative
