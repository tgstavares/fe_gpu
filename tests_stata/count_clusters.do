clear all
local HOME "/home/tgst/Desktop/Project_estimator/Project_estimator/tests_stata"
cd `HOME'
use nlsw_test.dta

tempname fh
file open `fh' using cluster_counts.txt, write replace

quietly describe
scalar __tmp = r(N)
file write `fh' "N_obs = " %10.0f __tmp _n

preserve
contract idcode
count
scalar __tmp = r(N)
file write `fh' "unique idcode = " %10.0f __tmp _n
restore

preserve
contract occ_code
count
scalar __tmp = r(N)
file write `fh' "unique occ_code = " %10.0f __tmp _n
restore

preserve
contract year
count
scalar __tmp = r(N)
file write `fh' "unique year = " %10.0f __tmp _n
restore

preserve
contract idcode occ_code
count
scalar __tmp = r(N)
file write `fh' "unique idcode#occ_code = " %10.0f __tmp _n
restore

preserve
contract idcode year
count
scalar __tmp = r(N)
file write `fh' "unique idcode#year = " %10.0f __tmp _n
restore

preserve
contract occ_code year
count
scalar __tmp = r(N)
file write `fh' "unique occ_code#year = " %10.0f __tmp _n
restore

preserve
contract idcode occ_code year
count
scalar __tmp = r(N)
file write `fh' "unique idcode#occ_code#year = " %10.0f __tmp _n
restore

file close `fh'
