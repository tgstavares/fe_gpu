#!/usr/bin/env bash

# create a variable for a specific directory
DIRPROGS="/home/tgst/Desktop/Project_estimator/Project_estimator"

# Stata to create parquet dataset and use reghdfe as reference
./input_cwd.sh
stata-mp -b 01TESTS_stata_dataset_small.do
rm 01TESTS_stata_dataset_small.log
cat REGS_01_stata.txt
rm REGS_01_stata.txt

# Convert parquet to bin
$DIRPROGS/tools/dataframe_to_fe_binary.py nlsw_test.parquet \
    --output nlsw_test.bin \
    --y ln_wage \
    --x hours,ttl_exp,union,tenure,wks_ue,msp,ind_code \
    --iv wks_work,ind_code \
    --fe idcode,occ_code,year \
    --drop-missing --summary --verbose --workers 16 

# Estimations
$DIRPROGS/build/src/fe_gpu --data nlsw_test.bin --fe-tol 1e-8 --formula "ln_wage ~ hours ttl_exp union tenure wks_ue, fe(idcode occ_code year)"
$DIRPROGS/build/src/fe_gpu --data nlsw_test.bin --fe-tol 1e-8 --formula "ln_wage ~ hours ttl_exp union tenure wks_ue, fe(idcode occ_code year) vce(cluster idcode occ_code year)"
$DIRPROGS/build/src/fe_gpu --data nlsw_test.bin --fe-tol 1e-8 --formula "ln_wage ~ hours ttl_exp union tenure wks_ue i.1.ind_code&&i.2.msp tenure&wks_ue, fe(idcode occ_code year) cluster(idcode occ_code year)"
$DIRPROGS/build/src/fe_gpu --data nlsw_test.bin --fe-tol 1e-8 --formula "ln_wage ~ hours ttl_exp union tenure wks_ue (hours ~ wks_work), fe(idcode occ_code year) cluster(idcode occ_code)"
$DIRPROGS/build/src/fe_gpu --data nlsw_test.bin --fe-tol 1e-8 --formula "ln_wage ~ hours ttl_exp union tenure wks_ue i.1.ind_code&&i.2.msp (hours ~ wks_work), fe(idcode occ_code year) cluster(idcode occ_code)"
$DIRPROGS/build/src/fe_gpu --data nlsw_test.bin --fe-tol 1e-8 --formula "ln_wage ~ hours i.2.msp ttl_exp union tenure wks_ue (hours i.2.msp ~ wks_work i.1.ind_code), fe(idcode occ_code year) cluster(idcode occ_code)"

# Remove datafiles
rm nlsw_test.*
