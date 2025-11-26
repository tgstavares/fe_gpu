clear all
local HOME "/home/tgst/Desktop/Project_estimator/Project_estimator/Examples"
cd `HOME'

pq use using "/srv/projetos/P008/_LIXO/nlswork_5M.parquet", clear
set rmsg on
reghdfe ln_wage age agesq union, absorb(fe_id = id  fe_occupation = occupation  fe_year = year)
set rmsg off

local vars id occupation year
foreach v in `vars'{
	preserve
	keep if e(sample)
	keep `v' fe_`v'
	gcollapse (mean) fe_`v', by(`v')
	sort `v'
	*save temp_fe_`v', replace
	export delimited using "fe_stata_`v'.csv", replace
	restore
}

