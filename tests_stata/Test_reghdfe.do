clear all
local HOME "/home/tgst/Desktop/Project_estimator/Project_estimator/tests_stata"
cd `HOME'
use nlsw_test.dta

tab ind_code, gen(gg)
pq save using temp_nlsw_test.parquet, replace
reghdfe ln_wage gg2 gg3 gg4 gg5, absorb(idcode occ_code year) vce(cluster idcode occ_code year)
reghdfe ln_wage gg2 gg3 gg4 gg5 gg6, absorb(idcode occ_code year) vce(cluster idcode occ_code year)
