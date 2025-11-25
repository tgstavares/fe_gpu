#!/usr/bin/env bash

# create a variable for a specific directory
DIRPROGS="/home/tgst/Desktop/Project_estimator/Project_estimator"

# Execute a python program located in that directory

# $DIRPROGS/tools/dataframe_to_fe_binary.py /srv/projetos/P008/_LIXO/nlswork_116M.parquet \
#     --output /srv/projetos/P008/02_testing_fe_estimators/nlswork_116M.bin \
#     --y ln_wage \
#     --x age,agesq,union \
#     --fe id,occupation,year \
#     --drop-missing --summary --verbose --workers 8 

$DIRPROGS/tools/dataframe_to_fe_binary.py /srv/projetos/P008/_LIXO/nlswork_5M.parquet \
    --output /srv/projetos/P008/02_testing_fe_estimators/nlswork_5M.bin \
    --y ln_wage \
    --x age,agesq,union \
    --fe id,occupation,year \
    --drop-missing --summary --verbose --workers 8 

