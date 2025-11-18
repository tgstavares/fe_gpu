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

const DEFAULT_TARGET_COLUMN = :wage
const DEFAULT_FE_COLUMNS = [:worker, :firm, :time]

struct CLIOptions
    data_path::String
    cluster_dims::Vector{Int}
    tol::Float64
    method::Symbol
    verbose::Bool
    target::Symbol
    regressors::Vector{Symbol}
    fe_columns::Vector{Symbol}
    iv_regressor_indices::Vector{Int}
    iv_var_names::Vector{Symbol}
end

function usage_and_exit()
    println("Usage: run_fixedeffects_gpu.jl --data FILE [--cluster-fe 1,2] [--tol 1e-6] [--method cuda] [--verbose] [--y dep] [--x r1,r2] [--fe fe1,fe2] [--iv-cols 1,2] [--iv-vars z1,z2]")
    exit(1)
end

function parse_symbol_list(value::String)::Vector{Symbol}
    tokens = split(value, ',')
    out = Symbol[]
    for tok in tokens
        stripped = strip(tok)
        isempty(stripped) && continue
        push!(out, Symbol(stripped))
    end
    unique!(out)
    return out
end

function parse_int_list(value::String)::Vector{Int}
    tokens = split(value, ',')
    out = Int[]
    for tok in tokens
        stripped = strip(tok)
        isempty(stripped) && continue
        push!(out, parse(Int, stripped))
    end
    unique!(out)
    return out
end

function parse_cli()::CLIOptions
    data_path = nothing
    cluster_dims = Int[]
    tol = 1.0e-6
    method_sym = :cuda
    verbose = false
    target = DEFAULT_TARGET_COLUMN
    regressors = Symbol[]
    fe_columns = copy(DEFAULT_FE_COLUMNS)
    iv_cols = Int[]
    iv_vars = Symbol[]

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
        elseif arg == "--y"
            i += 1
            i > length(ARGS) && usage_and_exit()
            target = Symbol(strip(ARGS[i]))
        elseif arg == "--x"
            i += 1
            i > length(ARGS) && usage_and_exit()
            regressors = parse_symbol_list(ARGS[i])
        elseif arg == "--fe"
            i += 1
            i > length(ARGS) && usage_and_exit()
            fe_parsed = parse_symbol_list(ARGS[i])
            isempty(fe_parsed) && error("--fe requires at least one column name")
            fe_columns = fe_parsed
        elseif arg == "--iv-cols"
            i += 1
            i > length(ARGS) && usage_and_exit()
            iv_cols = parse_int_list(ARGS[i])
        elseif arg == "--iv-vars"
            i += 1
            i > length(ARGS) && usage_and_exit()
            iv_vars = parse_symbol_list(ARGS[i])
        elseif arg == "--verbose"
            verbose = true
        else
            @error("Unknown argument $arg")
            usage_and_exit()
        end
        i += 1
    end

    data_path === nothing && usage_and_exit()
    return CLIOptions(data_path, cluster_dims, tol, normalize_method(method_sym), verbose, target, regressors, fe_columns,
                      iv_cols, iv_vars)
end

function normalize_method(sym::Symbol)
    s = lowercase(String(sym))
    if s == "cuda" || s == "gpu"
        return :CUDA
    elseif s == "cpu"
        return :cpu
    else
        return Symbol(sym)
    end
end

function cluster_symbols(dim_ids::Vector{Int}, fe_cols::Vector{Symbol})
    if isempty(dim_ids)
        return Symbol[]
    end
    max_dim = length(fe_cols)
    out = Symbol[]
    for d in dim_ids
        if d < 1 || d > max_dim
            error("Cluster FE dimension $d is out of range (1-$max_dim)")
        end
        push!(out, fe_cols[d])
    end
    unique!(out)
    return out
end

function build_formula(target::Symbol, regressors::Vector{Symbol}, fe_cols::Vector{Symbol})
    rhs_terms = vcat(["1"], string.(regressors), ["fe($(string(fe)))" for fe in fe_cols])
    rhs = join(rhs_terms, " + ")
    formula_expr = Meta.parse("@formula($(string(target)) ~ $rhs)")
    return eval(formula_expr)
end

function build_iv_formula(target::Symbol,
                          regressors::Vector{Symbol},
                          fe_cols::Vector{Symbol},
                          iv_regressors::Vector{Symbol},
                          instruments::Vector{Symbol})
    controls = setdiff(regressors, iv_regressors)
    rhs_terms = String[]
    push!(rhs_terms, "1")
    append!(rhs_terms, string.(controls))
    append!(rhs_terms, ["fe($(string(fe)))" for fe in fe_cols])
    instrument_text = join(string.(instruments), " + ")
    for endog in iv_regressors
        push!(rhs_terms, "( $(string(endog)) ~ $instrument_text )")
    end
    rhs = join(rhs_terms, " + ")
    formula_expr = Meta.parse("@formula($(string(target)) ~ $rhs)")
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
        opts = CLIOptions(opts.data_path, opts.cluster_dims, opts.tol, :cpu, opts.verbose, opts.target, opts.regressors,
                          opts.fe_columns, opts.iv_regressor_indices, opts.iv_var_names)
    end

    if opts.verbose
        @info "Loading Parquet dataset" path=opts.data_path backend=PARQUET_BACKEND
    end
    df = load_parquet_dataset(opts.data_path)

    names(df) |> isempty && error("Parquet file contains no columns.")
    colnames = Symbol.(names(df))
    available_cols = Set(colnames)
    for col in (opts.target, opts.fe_columns...)
        col in available_cols || error("Column $col missing in dataset.")
    end

    reg_syms = isempty(opts.regressors) ? [name for name in colnames if name != opts.target && !(name in opts.fe_columns)] :
                                          opts.regressors
    isempty(reg_syms) && error("No regressors specified or found besides target and FE columns.")
    for col in reg_syms
        col in available_cols || error("Regressor $col missing in dataset.")
    end

    fe_syms = opts.fe_columns
    cluster_syms = cluster_symbols(opts.cluster_dims, fe_syms)
    iv_reg_syms = Symbol[]
    if !isempty(opts.iv_regressor_indices)
        for idx in opts.iv_regressor_indices
            (idx < 1 || idx > length(reg_syms)) && error("IV regressor index $idx is out of range (1-$(length(reg_syms)))")
            push!(iv_reg_syms, reg_syms[idx])
        end
        unique!(iv_reg_syms)
    end

    instrument_syms = opts.iv_var_names
    if isempty(instrument_syms)
        instrument_syms = filter(col -> startswith(lowercase(String(col)), "iv"), colnames)
    end
    instrument_syms = unique(instrument_syms)
    for sym in instrument_syms
        sym in available_cols || error("Instrument column $sym missing in dataset.")
    end
    formula =
        if isempty(iv_reg_syms)
            build_formula(opts.target, reg_syms, fe_syms)
        else
            isempty(instrument_syms) && error("No instrument columns available. Use --iv-vars to specify instrument names.")
            build_iv_formula(opts.target, reg_syms, fe_syms, iv_reg_syms, instrument_syms)
        end

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
    if !isempty(iv_reg_syms)
        println("IV regression on regressors: $(iv_reg_syms) using instruments $(instrument_syms)")
    end
    println("")
    show(stdout, MIME("text/plain"), result)
    println("")
end

main()
