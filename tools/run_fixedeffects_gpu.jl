#!/usr/bin/env julia
# Run a FixedEffectModels.jl regression on the synthetic Parquet dataset using CUDA acceleration.

using Printf
using DataFrames
using FixedEffectModels
using StatsModels
using CUDA

try
    FixedEffectModels.info
catch
    FixedEffectModels.eval(:(info(args...; kwargs...) = nothing))
end

try
    Base.info
catch
    Base.@eval function info(args...; kwargs...) nothing end
end
const PARQUET_BACKEND = let
    try
        @eval using Parquet2
        :parquet2
    catch
        try
            @eval using Parquet
            :parquet
        catch err
            error("Install Parquet2.jl or Parquet.jl to load Parquet datasets: ", err)
        end
    end
end

const TARGET_COLUMN = :wage
const FE_COLUMNS = [:worker, :firm, :time]

struct CLIOptions
    data_path::String
    cluster_dims::Vector{Int}
    tol::Float64
    method::Symbol
    verbose::Bool
end

function usage_and_exit()
    println("Usage: run_fixedeffects_gpu.jl --data FILE [--cluster-fe 1,2] [--tol 1e-6] [--method cuda] [--verbose]")
    exit(1)
end

function parse_cli()::CLIOptions
    data_path = nothing
    cluster_dims = Int[]
    tol = 1.0e-6
    method_sym = :cuda
    verbose = false

    i = 1
    while i <= length(ARGS)
        arg = ARGS[i]
        if arg == "--data"
            i += 1
            i > length(ARGS) && usage_and_exit()
            data_path = ARGS[i]
        elseif arg == "--cluster-fe"
            i += 1
            i > length(ARGS) && usage_and_exit()
            tokens = split(ARGS[i], ',')
            for tok in tokens
                tok = strip(tok)
                isempty(tok) && continue
                push!(cluster_dims, parse(Int, tok))
            end
        elseif arg == "--tol"
            i += 1
            i > length(ARGS) && usage_and_exit()
            tol = parse(Float64, ARGS[i])
        elseif arg == "--method"
            i += 1
            i > length(ARGS) && usage_and_exit()
    method_sym = Symbol(lowercase(strip(ARGS[i])))
        elseif arg == "--verbose"
            verbose = true
        else
            @error("Unknown argument $arg")
            usage_and_exit()
        end
        i += 1
    end

    data_path === nothing && usage_and_exit()
    return CLIOptions(data_path, cluster_dims, tol, normalize_method(method_sym), verbose)
end

function normalize_method(sym::Symbol)
    lowercase_sym = Symbol(lowercase(String(sym)))
    if lowercase_sym == :cuda || lowercase_sym == :gpu
        return :gpu
    end
    return lowercase_sym
end

function cluster_symbols(dim_ids::Vector{Int})
    if isempty(dim_ids)
        return Symbol[]
    end
    max_dim = length(FE_COLUMNS)
    out = Symbol[]
    for d in dim_ids
        if d < 1 || d > max_dim
            error("Cluster FE dimension $d is out of range (1-$max_dim)")
        end
        push!(out, FE_COLUMNS[d])
    end
    unique!(out)
    return out
end

function build_formula(regressors::Vector{Symbol}, fe_cols::Vector{Symbol})
    rhs_terms = vcat(["1"], string.(regressors), ["fe($(string(fe)))" for fe in fe_cols])
    rhs = join(rhs_terms, " + ")
    formula_expr = Meta.parse("@formula(wage ~ $rhs)")
    return eval(formula_expr)
end

function load_parquet_dataset(path::String)
    if PARQUET_BACKEND == :parquet2
        ds = Parquet2.Dataset(path)
        return DataFrame(ds)
    else
        pq = Parquet.File(path)
        return DataFrame(pq)
    end
end

function main()
    opts = parse_cli()

    if opts.method == :gpu && !CUDA.has_cuda()
        @warn "CUDA not available; falling back to CPU execution"
        opts = CLIOptions(opts.data_path, opts.cluster_dims, opts.tol, :cpu, opts.verbose)
    end

    if opts.verbose
        @info "Loading Parquet dataset" path=opts.data_path backend=PARQUET_BACKEND
    end
    df = load_parquet_dataset(opts.data_path)

    names(df) |> isempty && error("Parquet file contains no columns.")
    for col in (TARGET_COLUMN, FE_COLUMNS...)
        col in propertynames(df) || error("Column $col missing in dataset.")
    end

    colnames = Symbol.(names(df))
    regressors = [name for name in colnames if name != TARGET_COLUMN && !(name in FE_COLUMNS)]
    isempty(regressors) && error("No regressors found besides target and FE columns.")
    sort!(regressors)

    fe_syms = FE_COLUMNS
    cluster_syms = cluster_symbols(opts.cluster_dims)
    formula = build_formula(regressors, fe_syms)

    vcov_est = isempty(cluster_syms) ? Vcov.robust() : Vcov.cluster(cluster_syms...)
    if opts.verbose
        @info "Running FixedEffectModels.reg" method=opts.method tol=opts.tol clusters=cluster_syms
    end

    if opts.method == :gpu
        CUDA.allowscalar(false)
    end
    start_ns = time_ns()
    result = reg(df, formula, vcov_est; method=opts.method, tol=opts.tol)
    elapsed = (time_ns() - start_ns) / 1e9

    println("FixedEffectModels.jl regression complete")
    @printf("Method: %s | tol=%g | runtime=%.3f s\n", String(opts.method), opts.tol, elapsed)
    if !isempty(cluster_syms)
        println("Clustered SEs on: $(cluster_syms)")
    end
    println("")
    show(stdout, MIME("text/plain"), result)
    println("")
end

main()
