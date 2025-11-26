#!/usr/bin/env bash

# create a variable for a specific directory
DIRPROGS="/home/tgst/Desktop/Project_estimator/Project_estimator"

# datasets to bin
$DIRPROGS/tools/dataframe_to_fe_binary.py /srv/projetos/P008/_LIXO/nlswork_116M.parquet \
    --output nlswork_116M.bin \
    --y ln_wage \
    --x age,agesq,union \
    --fe id,occupation,year \
    --drop-missing --summary --verbose --workers 8 

$DIRPROGS/tools/dataframe_to_fe_binary.py /srv/projetos/P008/_LIXO/nlswork_5M.parquet \
    --output nlswork_5M.bin \
    --y ln_wage \
    --x age,agesq,union \
    --fe id,occupation,year \
    --drop-missing --summary --verbose --workers 8 

# Stata regressions as reference
./input_cwd.sh
stata-mp -b 02TESTS_miguel_datasets.do
rm 02TESTS_miguel_datasets.log
cat REGS_02_stata.txt
cat REGS_03_stata.txt
rm REGS_*

# Estimations
$DIRPROGS/build/src/fe_gpu --data nlswork_5M.bin --fe-tol 1e-8 --formula "ln_wage ~ age agesq union, fe(id occupation year)"
$DIRPROGS/build/src/fe_gpu --data nlswork_5M.bin --fe-tol 1e-8 --formula "ln_wage ~ age agesq union, fe(id occupation year) cluster(id occupation year)"

$DIRPROGS/build/src/fe_gpu --data nlswork_116M.bin --fe-tol 1e-8 --formula "ln_wage ~ age agesq union, fe(id occupation year)"
$DIRPROGS/build/src/fe_gpu --data nlswork_116M.bin --fe-tol 1e-8 --formula "ln_wage ~ age agesq union, fe(id occupation year) cluster(id occupation year)"
$DIRPROGS/build/src/fe_gpu --data nlswork_116M.bin --fe-tol 1e-8 --formula "ln_wage ~ age agesq union, fe(id occupation year) cluster(id occupation year)" --fast

# Remove stuff
rm nlswork_*
