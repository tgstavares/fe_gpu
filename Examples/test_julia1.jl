start_time = time()

using DataFrames
using FixedEffectModels
using StatsModels
using CUDA
using Parquet2

ds = Parquet2.Dataset("data_massive.parquet")
df = DataFrame(ds)

tt2 = time()
result = reg(df, @formula(wage ~ tenure + sick_shock + extra_1 + extra_2 + extra_3 + extra_4 + extra_5 + extra6 + fe(worker) + fe(firm) + fe(time)), Vcov.cluster(:worker,:firm,:time); method=:CUDA, tol=1e-8)
println("")
show(stdout, MIME("text/plain"), result)
println("")

texec = time() - tt2
total_seconds = time() - start_time
println("Total regression time: $(round(texec        , digits=3)) seconds")
println("Total execution time : $(round(total_seconds, digits=3)) seconds")
