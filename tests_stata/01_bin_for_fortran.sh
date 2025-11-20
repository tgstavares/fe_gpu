#!/usr/bin/env bash

# create a variable for a specific directory
DIRPROGS="/home/tgst/Desktop/Project_estimator/Project_estimator"

# Execute a python program located in that directory
$DIRPROGS/tools/dataframe_to_fe_binary.py nlsw_test.parquet \
    --output nlsw_test.bin \
    --y ln_wage \
    --x hours,ttl_exp,union,tenure \
    --fe idcode,occ_code,year \
    --drop-missing --summary --verbose --workers 16 

# End of script
