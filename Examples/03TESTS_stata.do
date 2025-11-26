clear all

pq use using "/home/tgst/Desktop/Project_estimator/Project_estimator/Examples/data_massive.parquet", clear

capture set rmsg on
reghdfejl wage tenure sick_shock extra_1 extra_2 extra_3 extra_4 extra_5 extra_6, absorb(worker firm time) vce(cluster worker firm time)
capture set rmsg off
