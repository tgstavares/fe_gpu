clear all
local HOME "/home/tgst/Desktop/Project_estimator/Project_estimator/Examples"
cd `HOME'
use nlsw_test.dta

reghdfe ln_wage hours ttl_exp union tenure wks_ue i.ind_code##i.msp c.tenure#c.wks_ue, absorb(idcode occ_code year) vce(cluster idcode occ_code year)
