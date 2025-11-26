#!/usr/bin/env bash

# create a variable for a specific directory
DIRPROGS="/home/tgst/Desktop/Project_estimator/Project_estimator"

$DIRPROGS/tools/dataframe_to_fe_binary.py /srv/projetos/P008/_LIXO/nlswork_5M.parquet \
    --output nlswork_5M.bin \
    --y ln_wage \
    --x age,agesq,union \
    --fe id,occupation,year \
    --drop-missing --summary --verbose --workers 8 

./input_cwd.sh
stata-mp -b export_fe.do
rm export_fe.log

../build/src/fe_gpu --config config_stata.cfg --save-fe fes --cpu-threads 8

head fes_fe1.csv
head fe_stata_id.csv

head fes_fe3.csv
head fe_stata_year.csv

rm fes_fe*
rm fe_stata_*
rm nlswork_5M.bin
