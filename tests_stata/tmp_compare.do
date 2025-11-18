clear all
use nlsw_test.dta
ivreghdfe ln_wage (hours = tenure) ttl_exp union, absorb(idcode occ_code year) vce(cluster idcode occ_code year)
display "clusters: `e(clustvar)'"
display "N_clust: `e(N_clust)'"
ereturn list
