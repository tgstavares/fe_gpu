#!/usr/bin/env bash

# create a variable for a specific directory
DIRPROGS="/home/tgst/Desktop/Project_estimator/Project_estimator"

# Execute a python program located in that directory
$DIRPROGS/tools/dataframe_to_fe_binary.py temp_nlsw_test.parquet \
    --output temp_nlsw_test.bin \
    --y ln_wage \
    --x hours,ttl_exp,union,tenure,wks_ue,gg2,gg3,gg4,gg5,gg6,gg7,gg8,gg9,gg10,gg11,gg12 \
    --fe idcode,occ_code,year \
    --drop-missing --summary --verbose --workers 16 

# $DIRPROGS/tools/dataframe_to_fe_binary.py nlsw_test.parquet \
#     --output nlsw_test.bin \
#     --y ln_wage \
#     --x hours,ttl_exp,union,tenure,wks_ue,ind_code,msp \
#     --fe idcode,occ_code,year \
#     --drop-missing --summary --verbose --workers 16 

