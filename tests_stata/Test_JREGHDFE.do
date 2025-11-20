clear all
local HOME "/home/tgst/Desktop/Project_estimator/Project_estimator/tests_stata"
cd `HOME'


**** Data
webuse nlswork, clear
keep ln_wage ttl_exp union idcode year occ_code hours tenure wks_work wks_ue ind_code msp
save nlsw_test, replace

local i=12
foreach nn of numlist 1/`i' {
	di "Iter: `nn'"
	append using nlsw_test
	count
	compress
	save nlsw_test, replace
}
compress
order idcode occ_code year ln_wage hours wks_work ttl_exp union tenure wks_ue ind_code msp
pq save using nlsw_test.parquet, replace

**** Tests // *reg ln_wage i.year ttl_exp union, vce(cluster year)
capture set rmsg on
capture quietly log using "REGS_01_stata.txt", text replace

*reghdfe   ln_wage  hours                           ttl_exp union, absorb(idcode occ_code year) vce(cluster idcode occ_code year)
*ivreghdfe ln_wage (hours          = tenure)        ttl_exp union, absorb(idcode occ_code year) vce(cluster idcode occ_code year)
*ivreghdfe ln_wage (hours wks_work = tenure wks_ue) ttl_exp union, absorb(idcode occ_code year) vce(cluster idcode occ_code year)
*reghdfe ln_wage hours wks_work ttl_exp union c.ttl_exp#c.union i.ind_code##i.msp, absorb(idcode occ_code year) vce(cluster idcode occ_code year)

capture quietly log close
capture set rmsg off
