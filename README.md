# fe_gpu

GPU-accelerated fixed-effects regression engine with support for multi-way clustering, categorical expansion, interaction terms, and two-stage least squares (2SLS/IV). The code supports both GPU and CPU fallbacks and can generate design matrices directly from Stata-style formula strings.

## What it does
- Fits linear models with multiple high-dimensional fixed effects on GPU (CPU fallback available).
- Supports clustered standard errors (one-way to three-way) and homoskedastic SEs.
- Handles categorical variables (`i.var`), custom base category selection (`i.<base>.var`), and factorial interactions (`&&`).
- Supports interactions of continuous variables (`x&z`) and mixed terms.
- Supports IV/2SLS with multiple endogenous variables and instruments, including categorical instruments.
- Reads/writes a compact binary format (`.bin`) generated from parquet/CSV via helper tools.

## Build and install
```bash
cmake -S . -B build
cmake --build build -j$(nproc)
```
Options:
- `-DENABLE_CUDA=ON|OFF` (default ON)
- `-DENABLE_TESTS=ON|OFF` (default ON)

Main binary: `build/src/fe_gpu`

## CLI usage (direct binary)
The program estimates linear models of the form  
$`y = X\beta + \epsilon`$, where $`X`$ includes user‑specified regressors, expanded categorical dummies (with a chosen base), interactions, and when requested instrumented columns; fixed effects are partialled out via within‑transformation, and clustered or homoskedastic variance estimators are produced as requested.

For example, the command
```bash
build/src/fe_gpu --data nlsw_test.bin --fe-tol 1e-8 \
  --formula "ln_wage ~ hours tenure wks_ue, fe(idcode occ_code year)"
```
estimates the following statistical model:
$$\ln(\text{wage}_{it}) = \beta_0 + \beta_1\,\text{hours}_{it} + \beta_2\,\text{tenure}_{it} + \alpha_{\text{idcode}(i)} + \alpha_{\text{occ\_code}(j)} + \alpha_{\text{year}(t)} + \varepsilon_{it}.$$
with three sets of additive fixed effects (idcode, occ_code, year) by running a simple OLS regression with fixed effects and standard SEs.


For clustered SEs one can use:
```bash
build/src/fe_gpu --data nlsw_test.bin --fe-tol 1e-8 \
  --formula "ln_wage ~ hours tenure, fe(idcode occ_code year) vce(cluster idcode occ_code year)"
```
This being equivalent to adding the flag `--cluster-fe 1,2,3` to the command.

The program also alows for IV estimation. For example:
```bash
# Single endogenous regressor
build/src/fe_gpu --data nlsw_test.bin --fe-tol 1e-8 \
  --formula "ln_wage ~ hours ttl_exp union tenure wks_ue (hours ~ wks_work), fe(idcode occ_code year) cluster(idcode occ_code)"
```
```bash
# Endogenous categorical and continuous, categorical instruments
build/src/fe_gpu --data nlsw_test.bin --fe-tol 1e-8 \
  --formula "ln_wage ~ hours i.2.msp ttl_exp union tenure wks_ue (hours i.2.msp ~ wks_work i.1.ind_code), fe(idcode occ_code year) cluster(idcode occ_code)"
```

Additionally, categorical base selection and interactions are supported:
- `i.var` expands all levels, dropping the first level by default.
- `i.<base>.var` drops the specified base level (e.g., `i.10.ind_code` omits level 10).
- Factorial interactions: `i.ind_code&&i.msp` (expands all interactions).
- Continuous interactions: `tenure&wks_ue`.

Other useful flags that can be used include:
- `--fast` : faster GPU clustering path (may change clustered SEs slightly).
- `--demean-cg` : use CG-based within-transform.
- `--cpu-threads N` : set CPU threads for fallbacks.
- `--verbose` : show full runtime info.

## Config file usage
You can pass options via a `.cfg`:
```ini
data = /path/to/data.bin
fe_tol = 1e-8
verbose = true
formula = ln_wage ~ hours ttl_exp union tenure wks_ue, fe(idcode occ_code year) vce(cluster idcode occ_code year)
```
Then run:
```bash
build/src/fe_gpu --config config_stata.cfg
```

## Binary data format and tools
- `tools/dataframe_to_fe_binary.py`: convert parquet/CSV into the binary format:
  ```bash
  tools/dataframe_to_fe_binary.py input.parquet \
    --output nlsw_test.bin \
    --y ln_wage \
    --x hours,ttl_exp,union,tenure,wks_ue,msp,ind_code \
    --iv wks_work,ind_code \
    --fe idcode,occ_code,year \
    --drop-missing --summary --verbose --workers 16
  ```
- `tools/generate_synthetic_data.py`: generate large synthetic panel binaries/parquet. Example:
  ```bash
  tools/generate_synthetic_data.py \
    --workers 5000000 --firms 1000000 --periods 30 --extra-vars 6 \
    --output data_massive.bin --parquet-output data_massive.parquet
  ```
  The resulting `data_massive.bin`/`.parquet` can be passed directly to `fe_gpu`, e.g.:
  ```bash
  build/src/fe_gpu --data data_massive.bin --fe-tol 1e-8 \
    --formula "wage ~ tenure sick_shock extra_1 extra_2 extra_3 extra_4 extra_5 extra_6, fe(worker firm time) cluster(worker firm time)"
  ```

## Examples (from `Examples/`)
These scripts reproduce Stata/Julia references and compare outputs and timings.

### 01TESTS_fe_gpu_small.sh (N ≈ 26k)
- Builds `nlsw_test.parquet/bin`, runs Stata (`reghdfe`, `ivreghdfe`) as reference; `fe_gpu` estimates match Stata coefficients and SEs (homoskedastic or clustered) within reported tolerances.
- Formulas exercised:
  1) OLS, no clustering  
     `ln_wage ~ hours ttl_exp union tenure wks_ue, fe(idcode occ_code year)`
  2) OLS, clustered on three FEs  
     `ln_wage ~ hours ttl_exp union tenure wks_ue, fe(idcode occ_code year) vce(cluster idcode occ_code year)`
  3) OLS with categorical interactions + continuous interaction, clustered on three FEs  
     `ln_wage ~ hours ttl_exp union tenure wks_ue i.1.ind_code&&i.2.msp tenure&wks_ue, fe(idcode occ_code year) vce(cluster idcode occ_code year)`
  4) IV (2SLS) with one endogenous regressor, clustered on two FEs  
     `ln_wage ~ hours ttl_exp union tenure wks_ue (hours ~ wks_work), fe(idcode occ_code year) vce(cluster idcode occ_code)`
  5) IV with categorical interactions in instruments, clustered on idcode  
     `ln_wage ~ hours ttl_exp union tenure wks_ue i.ind_code&&i.msp (hours ~ wks_work), fe(idcode occ_code year) vce(cluster idcode)`
  6) IV with both a continuous and a categorical endogenous regressor, clustered on two FEs  
     `ln_wage ~ hours ttl_exp union tenure wks_ue (hours i.msp ~ wks_work i.ind_code), fe(idcode occ_code year) vce(cluster idcode occ_code)`
- Comparisons (vs. Stata logs `01TESTS_fe_gpu_small_output.txt`):

  | # | Model | Avg \\|β_fe_gpu − β_Stata\\| | Avg rel SE diff |
  |---|-------|-------------------------------|------------------|
  |1|OLS (no clustering)|2.97×10⁻⁸|6.43×10⁻⁵|
  |2|OLS (clustered on idcode/occ_code/year)|2.97×10⁻⁸|9.29×10⁻⁶|
  |3|OLS with categorical + continuous interactions (clustered)|5.84×10⁻⁹|6.10×10⁻²|
  |4|IV with hours endogenous (clustered)|1.34×10⁻⁸|1.82×10⁻⁴|
  |5|IV with categorical instruments (clustered)†|1.87×10⁻⁸|2.33×10⁻¹|
  |6|IV with hours and msp endogenous (clustered)|1.98×10⁻⁸|1.58×10⁻⁴|

  †Regression 5 excludes zero (omitted) categories; factor-level SEs differ modestly, coefficients align.


### 02TESTS_miguel_data.sh (N = 5M and 116M)
- The datasets were kindly shared by an applied economist; grateful for the collaboration.
- Converts parquet datasets to bin with FEs (id/occupation/year).
- Stata reference via `reghdfe` / `reghdfejl` on 5M and 116M.
- `fe_gpu` runs:
  - OLS with/without clustering on 5M and 116M
  - `--fast` on 116M
- Timing: 5M run ~4.1s total (demean ~3.5s) in earlier benchmarks; 116M run is heavier but faster than Stata on same hardware.
- Coefficients/SEs align with Stata reference logs `02TESTS_miguel_data_output.txt`.

- Small dataset (5M) comparisons vs. Stata:

  | # | Mode | Avg \|β_fe_gpu − β_Stata\| | Avg rel SE diff |
  |---|-----|---------------------------|------------------|
  |1|Non-clustered|1.08×10⁻⁸|2.48×10⁻⁴|
  |2|Clustered (id, occupation, year)|1.08×10⁻⁸|1.88×10⁻⁵|

- Large dataset (116M) comparisons vs. Stata:

  | # | Mode | Avg \|β_fe_gpu − β_Stata\| | Avg rel SE diff |
  |---|-----|---------------------------|------------------|
  |1|Non-clustered|1.73×10⁻⁸|1.00×10⁻³|
  |2|Clustered (standard)|1.73×10⁻⁸|1.32×10⁻⁵|
  |3|Clustered (`--fast`)|1.73×10⁻⁸|8.72×10⁻²|

- Large dataset execution times (approximate):

  | # | Run | Time (s) |
  |---|-----|---------|
  |1|Stata/Julia non-clustered|≈284.7|
  |2|Stata/Julia clustered|≈262.0|
  |3|fe_gpu non-clustered|≈15.2|
  |4|fe_gpu clustered (standard)|≈26.5|
  |5|fe_gpu clustered (`--fast`)|≈16.0|

### 03TESTS_massive_data.sh (synthetic, very large)
- Generates `data_massive.bin/parquet` with worker/firm/time FEs and extra vars.
- `fe_gpu` runs with/without `--fast` and clustering on three-way FEs.
- Julia GPU reference (`test_julia1.jl`, FixedEffectModels.jl, `method=:CUDA`), plus Stata `reghdfejl` as reference.
- Comparisons on the 150M synthetic dataset (logs in `03TESTS_massive_data_output.txt`):

  - vs. Stata `reghdfejl` (clustered SEs):

    | # | Mode | Avg \|β_fe_gpu − β_Stata\| | Avg rel SE diff |
    |---|------|---------------------------|------------------|
    |1|fe_gpu clustered (standard)|5.0×10⁻⁹|1.20×10⁻⁴|
    |2|fe_gpu clustered (`--fast`)|5.0×10⁻⁹|2.49×10⁻¹|

  - Execution times (approximate):

    | # | Run | Time (s) |
    |---|-----|---------|
    |1|Stata `reghdfejl` clustered|≈505 |
    |2|fe_gpu clustered (standard)|≈57.8 |
    |3|fe_gpu clustered (`--fast`)|≈23.7 |
    |4|FixedEffectModels.jl (GPU) clustered|≈470.7 (regression) / ≈521.5 total |

### Utilities in Examples
- `input_cwd.sh`: patches Stata `.do` files’ `local HOME` to the script’s folder.
- `config_miguel.cfg`, `config_stata.cfg`: ready-to-use configs matching the test scripts.
- Output snapshots (`*_output.txt`) capture prior reference runs for quick comparison.

## Notes on accuracy and performance
- Coefficients match Stata/Julia references across provided examples; clustered SEs align within small numerical tolerances (check the reference logs in `Examples/`).
- `--fast` trades some robustness in clustered SEs for speed (use standard mode for strict matching).
- CG-based demeaning (`--demean-cg`) reduces iterations; GPU offload keeps demeaning and clustering fast.
- CPU fallbacks are threaded; GPU paths log fallbacks when triggered.

## Running the test scripts
From `Examples/`:
```bash
./01TESTS_fe_gpu_small.sh
./02TESTS_miguel_data.sh
./03TESTS_massive_data.sh
```
Make sure paths inside the scripts point to your data locations and that Stata/Julia are available if you want the reference runs.

## License and contributions
Please open issues/PRs for bugs or feature requests. The tests in `tests/` build with `ENABLE_TESTS=ON`; remove or disable if you need a lean build.
