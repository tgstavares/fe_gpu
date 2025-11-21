clear all
local HOME "/home/tgst/Desktop/Project_estimator/Project_estimator/tests_stata"
cd `HOME'
use nlsw_test.dta
capture set rmsg on
reghdfejl ln_wage hours wks_work ttl_exp union tenure, absorb(idcode occ_code year)
capture set rmsg off
