clear all
local HOME "/home/tgst/Desktop/Project_estimator/Project_estimator/tests_stata"
cd `HOME'
use nlsw_test.dta
reghdfe ln_wage hours tenure ttl_exp union, absorb(idcode occ_code year) vce(cluster idcode occ_code year)
