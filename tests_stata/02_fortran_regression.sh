#!/usr/bin/env bash

# create a variable for a specific directory
DIRPROGS="/home/tgst/Desktop/Project_estimator/Project_estimator"

# Execute a program located in that directory
$DIRPROGS/build/src/fe_gpu --data nlsw_test.bin --iv-cols 1 --cluster-fe 1,2,3 --verbose --fe-tol 1e-8

# End of script
