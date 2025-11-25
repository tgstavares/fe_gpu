#!/usr/bin/env bash

# create a variable for a specific directory
DIRPROGS="/home/tgst/Desktop/Project_estimator/Project_estimator"

# Execute a python program located in that directory
$DIRPROGS/build/src/fe_gpu --data nlsw_test.bin --fe-tol 1e-8 --formula "ln_wage ~ hours ttl_exp union tenure wks_ue, fe(idcode occ_code year)" --verbose

