clear all
local HOME "/home/tgst/Desktop/Project_estimator/Project_estimator/Examples"
cd `HOME'

capture set rmsg on

pq use using "/srv/projetos/P008/_LIXO/nlswork_5M.parquet", clear
capture quietly log using "REGS_02_stata.txt", text replace
reghdfe ln_wage age agesq union, absorb(id occupation year)
reghdfe ln_wage age agesq union, absorb(id occupation year) vce(cluster id occupation year)
capture quietly log close

capture quietly log using "REGS_03_stata.txt", text replace
pq use using "/srv/projetos/P008/_LIXO/nlswork_116M.parquet", clear
reghdfejl ln_wage age agesq union, absorb(id occupation year)
reghdfejl ln_wage age agesq union, absorb(id occupation year) vce(cluster id occupation year)
capture quietly log close

capture set rmsg off
