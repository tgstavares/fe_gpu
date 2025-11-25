clear all
local HOME "/home/tgst/Desktop/Project_estimator/Project_estimator/Examples"
cd `HOME'

capture set rmsg on

pq use using "../_LIXO/nlswork_5M.parquet", clear
*pq use using "../_LIXO/nlswork_116M.parquet", clear

reghdfe ln_wage age agesq union, absorb(id occupation year)
*reghdfejl ln_wage age agesq union, absorb(id occupation year)

capture set rmsg off
