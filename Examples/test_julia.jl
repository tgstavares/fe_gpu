start_time = time()

using DataFrames
using FixedEffectModels
using StatsModels
using CUDA
using Parquet2

ds = Parquet2.Dataset("nlsw_test.parquet")
df = DataFrame(ds)

result = reg(df, @formula(ln_wage ~ (hours ~ tenure) + ttl_exp + union + fe(idcode) + fe(occ_code) + fe(year)), Vcov.cluster(:idcode,:occ_code,:year); method=:CUDA, tol=1e-8)
println("")
show(stdout, MIME("text/plain"), result)
println("")

total_seconds = time() - start_time
println("Total execution time: $(round(total_seconds, digits=3)) seconds")
