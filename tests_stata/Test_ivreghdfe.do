clear all
local HOME "/home/tgst/Desktop/Project_estimator/Project_estimator/tests_stata"
cd `HOME'
use nlsw_test.dta
reghdfe ln_wage hours wks_work ttl_exp union c.ttl_exp#c.union i.ind_code##i.msp, absorb(idcode occ_code year) vce(cluster idcode occ_code year)
