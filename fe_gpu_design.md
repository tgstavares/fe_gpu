# GPU-Accelerated High-Dimensional Fixed-Effects Regression (Design Spec)

## 1. Context and Goals

We want to implement a **high-dimensional fixed-effects (HDFE) linear regression estimator** using **Fortran + CUDA** (or C/CUDA wrappers where needed), targeting very large datasets (tens of millions of observations or more) and high-dimensional fixed effects.

The estimator is conceptually similar to what `FixedEffectModels.jl` + `FixedEffects.jl` do, but implemented as a standalone HPC code aimed at:

- **Maximizing speedup from GPUs**, especially for:
  - High-dimensional fixed-effect demeaning;
  - Dense linear algebra on the “short” set of regressors.
- **Minimizing data movement** between:
  - Disk ↔ CPU RAM;
  - CPU RAM ↔ GPU memory.
- **Using a very simple on-disk binary format** (no Parquet decoding inside Fortran) for fast IO.

The primary use case is OLS with high-dimensional FE and cluster-robust standard errors, where **only the coefficients on `W` matter** (fixed effects are nuisance parameters).

---

## 2. Model and Notation

We consider models of the form:

$$
y = W \beta + F \gamma + \varepsilon
$$

- $y$: $N \times 1$ dependent variable.
- $W$: $N \times K$ matrix of “regular” regressors (continuous or dummies), **dense**.
- $F$: high-dimensional fixed-effects structure, possibly with several FE dimensions:
  - e.g. worker FE, firm FE, time FE, worker×year, etc.
  - Stored as **integer group indices** rather than a literal dummy matrix.
- $\beta$: $K \times 1$ coefficients of interest.
- $\gamma$: FE coefficients (usually not needed explicitly).
- We may also want:
  - Weights $w_i$,
  - Cluster IDs $c_i$ for cluster-robust VCOV.

The core idea is to obtain **demeaned variables**:

$$
\tilde{y} = M_F y, \quad \tilde{W} = M_F W
$$

where $M_F$ is the within-transformation that subtracts out the fixed effects. Then we solve the “short” regression:

$$
\tilde{y} = \tilde{W} \beta + u
$$

on GPU as much as possible.

---

## 3. Overall Pipeline

We split the computation into:

1. **Offline data preparation (outside Fortran)**  
   - Use Julia or Python to read raw data (Parquet, CSV, etc.).
   - Select the necessary variables and convert them into a simple binary format optimized for Fortran/GPU.
   - This happens **once per dataset** or whenever the dataset changes.

2. **Fortran + CUDA runtime**  
   - Read the binary file(s) into RAM with a small number of large sequential reads.
   - Copy data to GPU.
   - Perform FE demeaning and dense regression on GPU.
   - Bring back only small results (e.g. $K \times K$ matrices, $\beta$, VCOV).

The **offline step** can be implemented later; the current spec focuses on the runtime estimator, but it should be designed assuming this binary format.

---

## 4. Binary On-Disk Data Format

We want a format that Fortran can read with minimal overhead:

- **One main binary file**, `data.bin` (stream, unformatted), with:

### 4.1 Header

Layout (all little-endian):

- `int64` `N` – number of observations.
- `int32` `K` – number of columns in `W`.
- `int32` `n_fe` – number of FE dimensions (e.g. 1 = firm, 2 = worker+firm).
- `int32` `has_cluster` – 0 or 1 (cluster variable present).
- `int32` `has_weights` – 0 or 1 (weights present).
- (Optionally) `int32` `precision_flag` – 0 = `Float64`, 1 = `Float32` on disk/GPU.

We can extend the header later with offsets or metadata if needed.

### 4.2 Data Blocks

Immediately following the header, data blocks in this order:

1. `y` – length `N` (float, precision per `precision_flag`).
2. `W` – `N × K` matrix, **column-major** order (Fortran-friendly):
   - Stored as contiguous `N*K` values: column 1 for all rows, then column 2, etc.
3. FE indices (per FE dimension):
   - For each FE dimension `d = 1..n_fe`:
     - `fe_d` – length `N`, `Int32` group IDs in `1..G_d` (after preprocessing).
   - The mapping levels→IDs and counts `G_d` may be stored separately (text or metadata file) if needed, but the estimator only needs `fe_d` as integer labels.
4. **Optional: cluster IDs**
   - `cluster` – length `N`, `Int32`, if `has_cluster == 1`.
5. **Optional: weights**
   - `w` – length `N`, float, if `has_weights == 1`.

We assume:

- All missing data have been resolved **before** writing this file.
- FE indices and cluster IDs are already integer-coded and contiguous (1..G).

---

## 5. CPU–GPU Division of Labor

### 5.1 What the GPU should do

Ideally on GPU:

1. **Fixed-effects demeaning**  
   - For each FE dimension, repeatedly apply within-transformation $M_{F_d}$ via group-wise means.
   - Combine multiple FE dimensions by iterating until convergence (e.g. alternating projections).
   - Implementation uses group sums, counts, and scatter/gather operations, but the data (y, W) stay on GPU.

2. **Dense linear algebra on $\tilde{W}$, $\tilde{y}$**
   - Compute:
     - $Q = \tilde{W}^\top \tilde{W}$ (`K×K`),
     - $b = \tilde{W}^\top \tilde{y}$ (`K×1`),
     - and optionally residuals $\hat{u} = \tilde{y} - \tilde{W}\hat{\beta}$ (for VCOV).
   - Use cuBLAS (e.g. GEMM/GEMV/SYRK) or equivalent.

3. **Workhorse operations for FE solver**
   - Repeated group-wise reductions and updates:
     - For each FE group, compute sums / averages of `y` and each column of `W` over the group.
     - Subtract group effects back from each observation.

The FE solver *only* needs to produce **demeaned y and W**; we do not need $\gamma$ explicitly.

### 5.2 What the CPU can do

On the CPU side:

- **File IO**: read `data.bin` into host arrays.
- **GPU setup**: initialize CUDA context, allocate device memory, copy data.
- Solve small systems if desired:
  - Solve $Q \beta = b$ on CPU (Lapack) or GPU (cuSOLVER).
  - $K$ is usually small (tens or a few hundreds), so this is cheap either way.
- Cluster-robust variance estimation:
  - Could be done on CPU or GPU depending on design; initial version can be CPU-only.

---

## 6. Algorithmic Sketch

### 6.1 Step 0: IO and host memory

In Fortran (conceptual):

```fortran
! Read header
read(unit) N, K, n_fe, has_cluster, has_weights, precision_flag

allocate(y(N), W(N, K))
allocate(fe_ids(n_fe, N))
if (has_cluster == 1) allocate(cluster(N))
if (has_weights == 1) allocate(weights(N))

! Read y, W, FE IDs, cluster, weights
read(unit) y
read(unit) W
do d = 1, n_fe
    read(unit) fe_ids(d, :)
end do
if (has_cluster == 1) read(unit) cluster
if (has_weights == 1) read(unit) weights
```

Then allocate corresponding device arrays and transfer.

### 6.2 Step 1: Push data to GPU

- Allocate device arrays:
  - `d_y`, `d_W`, `d_fe_ids`, `d_cluster`, `d_weights` as needed.
- Use `cudaMemcpy` (or Fortran bindings) to copy from host.

### 6.3 Step 2: Fixed-effects demeaning on GPU

We want a generic routine:

> Given `d_y` (N) and `d_W` (N×K) and `fe_ids` for `n_fe` dimensions, replace `d_y` and `d_W` with their within-transformed versions (demeaned by all FEs).

High-level algorithm (alternating projection):

1. Initialize `d_y_tilde = d_y`, `d_W_tilde = d_W`.
2. Repeat for a fixed number of iterations or until convergence:
   - For each FE dimension `d = 1..n_fe`:
     1. For each FE group `g` in dimension `d`, compute group means:
        - $\bar{y}_{g}$, $\bar{w}_{g, k}$ for each column `k`.
     2. Subtract these group effects from each observation in that group.
   - Track max change; stop when below tolerance.

Implementation details:

- Represent each FE dimension `d` with:
  - `fe_ids_d(N)` group indices,
  - Precomputed group offsets (CSR-style):
    - `group_start_d(G_d + 1)`, `group_indices_d(N)` if we want reorganized data.
  - This grouping can be done offline and stored, or computed once on CPU and copied to GPU.

- GPU kernels needed:
  - **Kernel A (reduce)**: For each group, accumulate sums for `y` and each column in `W`.
  - **Kernel B (divide)**: Convert sums to means by dividing by group counts.
  - **Kernel C (apply)**: For each observation, subtract its group’s mean.

The FE solver *only* needs to produce **demeaned y and W**; we do not need $\gamma$ explicitly.

### 6.4 Step 3: Dense regression on GPU

Once we have `d_y_tilde` and `d_W_tilde`:

1. Compute $Q = \tilde{W}^\top \tilde{W}$ (`K×K`):
   - Use cuBLAS `gemm` or `syrk`.
2. Compute $b = \tilde{W}^\top \tilde{y}$ (`K×1`):
   - Use cuBLAS `gemv`.
3. Optionally compute residuals:
   - $\hat{y} = \tilde{W} \beta$,
   - `d_resid = d_y_tilde - d_y_hat`.

Then:

- Copy `Q` and `b` back to CPU; solve `Q β = b` with Lapack, **or**
- Keep them on GPU and use cuSOLVER (nice but optional for v1).

### 6.5 Step 4: Cluster-robust VCOV (optional for v1)

For a given cluster ID `cluster(i)`:

- Ideally on GPU:
  - Group residuals and regressors by cluster,
  - Compute meat of the sandwich estimator.

For a first version, it is acceptable to:

1. Copy `β`, and (if needed) `W_tilde`, `y_tilde`, or residuals back to CPU.
2. Compute cluster-robust covariance on CPU with a straightforward algorithm:
   - Standard $(X'X)^{-1} X' \Omega X (X'X)^{-1}$ with clustered `Ω`.

We can refine this later.

---

## 7. Memory and Scaling Considerations

We target cases with:

- `N` up to tens (or hundreds) of millions of observations,
- `K` up to a few hundred regressors.

To handle memory constraints:

1. **Configurable precision**
   - Allow running with `Float32` on GPU for `W`, `y`, residuals, to halve memory.
   - Optionally keep accumulations (e.g. `Q`, `b`) in `Float64`.

2. **Chunking by rows if necessary**
   - If full `N × K` doesn’t fit on GPU:
     - Process data in row chunks `N_chunk`:
       - Demean chunk-by-chunk (requires careful handling of FE spans across chunks; may need grouping pre-processing to keep all members of a FE group in same chunk).
       - Accumulate contributions to `Q` and `b` across chunks.

Initial implementation can assume that **all data fits on GPU**. Chunking can be a second phase.

---

## 8. What the Coding Assistant Should Implement

The main tasks to implement (in rough order):

1. **Fortran module for binary IO**
   - Read `data.bin` as specified.
   - Allocate and fill host arrays: `y`, `W`, `fe_ids`, `cluster`, `weights`.
   - Provide a clean API, e.g. `subroutine load_data(filename, ...)`.

2. **Fortran/CUDA module for GPU setup**
   - Initialize CUDA context.
   - Allocate device arrays and copy data from host.
   - Provide cleanup routines.

3. **GPU FE demeaning module**
   - Data structures to represent FE groupings.
   - Kernels:
     - group-wise reduction (sums/counts),
     - group-wise mean computation,
     - subtraction from observations.
   - Iterative algorithm for multiple FE dimensions (loop until convergence).
   - Operates in-place on `d_y` and `d_W`.

4. **GPU dense regression module**
   - Wrap cuBLAS calls for:
     - $Q = W^\top W$,
     - $b = W^\top y$,
     - possibly residuals.
   - Provide a clean API: `compute_cross_products(d_W, d_y, Q, b)`.

5. **CPU-side solver for β and VCOV (initial version)**
   - Use Lapack to solve `Q β = b`.
   - Implement plain OLS VCOV.
   - (Optional) implement cluster-robust VCOV using CPU arrays.

6. **Top-level driver / CLI**
   - A small Fortran program that:
     - Reads command-line arguments (`data.bin` file, options),
     - Calls IO, GPU setup, FE demeaning, regression,
     - Outputs:
       - `beta`,
       - standard errors,
       - summary stats (N, K, FE dims, etc.).

---

## 9. Extensions and Nice-to-Haves (Later)

- Full GPU-based cluster-robust VCOV.
- Support for instrumental variables (IV) and 2SLS:
  - Running regressions of the form `(y, X, Z)` on FE-demeaned data.
- Mixed precision:
  - `Float32` for data, `Float64` for accumulations.
- Chunking strategies for datasets larger than GPU memory.
- Integration with existing toolchains (e.g. wrappers callable from Stata/Julia).

---

## 10. Summary

This estimator should:

- Treat **fixed effects as a high-dimensional projection problem** implemented as GPU-friendly group operations;
- Keep **y and W resident on GPU** from the moment they’re copied until cross-products are computed;
- Use a **very simple binary file format** to avoid complexity and overhead at runtime;
- Minimize host–device transfers (only transfer large arrays once; small matrices/vectors back).

The code-writing assistant should follow this design and implement the modules in a way that makes it easy to:

- Extend to more features later (IV, more VCOV options),
- Swap in different FE algorithms if needed,
- Integrate with external languages via a stable CLI or C ABI.
