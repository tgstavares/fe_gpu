clear all
local HOME "/home/tgst/Desktop/Project_estimator/Project_estimator/tests_stata"
cd `HOME'
use nlsw_test.dta

reghdfe ln_wage hours tenure ttl_exp union, absorb(idcode occ_code year) vce(cluster idcode)
matrix list e(V)

reghdfe ln_wage hours tenure ttl_exp union, absorb(idcode occ_code year) vce(cluster occ_code)
matrix list e(V)

reghdfe ln_wage hours tenure ttl_exp union, absorb(idcode occ_code year) vce(cluster year)
matrix list e(V)

reghdfe ln_wage hours tenure ttl_exp union, absorb(idcode occ_code year) vce(cluster idcode occ_code)
matrix list e(V)

reghdfe ln_wage hours tenure ttl_exp union, absorb(idcode occ_code year) vce(cluster idcode year)
matrix list e(V)

reghdfe ln_wage hours tenure ttl_exp union, absorb(idcode occ_code year) vce(cluster occ_code year)
matrix list e(V)

reghdfe ln_wage hours tenure ttl_exp union, absorb(idcode occ_code year) vce(cluster idcode occ_code year)
matrix list e(V)
