# Implementation Notes

## Build Prerequisites
- CMake ≥ 3.24
- Fortran, C, and CUDA compilers (GNU + NVHPC CUDA 12 tested)
- LAPACK/BLAS (detected via `find_package(LAPACK REQUIRED)`)

Configure & build:
```bash
cmake -S . -B build
cmake --build build
ctest --output-on-failure
```

## Runtime Overview
1. **Binary IO (`fe_data_io`)** reads `data.bin` into `fe_host_arrays` using the spec’d header layout.
2. **GPU Upload (`fe_gpu_data`)** allocates device buffers for `y`, `W`, FE IDs, and scratch group stats.
3. **FE Demeaning (`fe_gpu_demean`)** runs alternating projections entirely on GPU, iterating until the max group update < tolerance or `fe_max_iterations`.
4. **Cross Products (`fe_gpu_linalg`)** uses cuBLAS DSYRK/DGEMV to form `Q=W'W` and `b=W'y`.
5. **Solve (`fe_solver`)** copies `Q`, `b` back to host, symmetrizes the upper triangle, and calls LAPACK `dposv`.
6. **Pipeline API (`fe_pipeline`)** exposes `fe_gpu_estimate`, returning `fe_estimation_result` (β vector, FE iterations, solver status). The CLI simply logs these fields.

## Tests
- `test_data_loader` exercises binary IO (float32/float64 coverage).
- `test_gpu_runtime` validates device allocation and memcpy roundtrips.
- `test_gpu_demean` checks that FE means are zeroed after GPU demeaning.
- `test_gpu_regression` builds a synthetic dataset with known β and confirms the full GPU pipeline recovers it.

Run all tests with `ctest --output-on-failure` from the build directory.

## Synthetic Data Generator

Use `tools/generate_synthetic_data.py` to create binary datasets (and optional Parquet mirrors) for experimentation:

```bash
python tools/generate_synthetic_data.py \
    --workers 5000 --firms 400 --periods 12 \
    --intercept 1.0 --betas 0.5 -0.3 \
    --extra-vars 2 --extra-coeffs 0.2 -0.1 \
    --noise-std 0.4 --sick-std 0.7 --extra-std 0.35 \
    --seed 42 --output custom_data.bin \
    --parquet-output custom_data.parquet

# or load the same options from a file
python tools/generate_synthetic_data.py @configs/example_synthetic.txt

./build/src/fe_gpu --data custom_data.bin --verbose
```

Flags let you control cohort sizes, β coefficients, extra regressors, noise level, and precision. The binary output follows the `data.bin` layout consumed by `fe_gpu`, whereas the Parquet file (if requested) contains the same observation-level columns (`worker`, `firm`, `time`, regressors, and `wage`) for external tooling.

## Julia comparison runner

`tools/run_fixedeffects_gpu.jl` benchmarks the Julia/FixedEffectModels.jl implementation on the Parquet dataset:

```bash
julia tools/run_fixedeffects_gpu.jl \
    --data synthetic_example.parquet \
    --cluster-fe 1,2 \
    --tol 1e-6 \
    --method cuda \
    --verbose
```

The script auto-builds the formula (all regressors plus `fe(worker) + fe(firm) + fe(time)`), enables CUDA execution when available, clusters the covariance on the requested FE dimensions, and reports the total runtime alongside the coefficient table. Use the same `--cluster-fe` and tolerance settings as `fe_gpu` to make apples-to-apples comparisons.
