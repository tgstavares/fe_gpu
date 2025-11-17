#!/usr/bin/env python3
"""Convert tabular data into the fe_gpu binary format."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Sequence, Optional

import numpy as np
import pandas as pd

HEADER_FORMAT = "<qiiiii"
PRECISION_FLAGS = {
    "float64": 0,
    "float32": 1,
}

def parse_columns(value: str, *, allow_empty: bool = False) -> List[str]:
    items = [col.strip() for col in value.split(',') if col.strip()]
    if not items and not allow_empty:
        raise argparse.ArgumentTypeError("At least one column must be provided")
    return items

def parse_optional_columns(value: str) -> List[str]:
    if value is None:
        return []
    value = value.strip()
    if not value:
        return []
    return parse_columns(value, allow_empty=True)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a DataFrame stored on disk to the fe_gpu binary format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", type=pathlib.Path, help="Path to input dataset (csv/parquet/pickle)")
    parser.add_argument("--input-format", choices=["auto", "csv", "parquet", "pickle"], default="auto")
    parser.add_argument("--output", type=pathlib.Path, required=True, help="Output binary path")
    parser.add_argument("--y", required=True, help="Column to use as the dependent variable")
    parser.add_argument("--x", required=True, type=parse_columns,
                        help="Comma-separated list of regressors")
    parser.add_argument("--fe", required=True, type=parse_columns,
                        help="Comma-separated list of fixed-effect id columns (order matters)")
    parser.add_argument("--iv", type=parse_optional_columns, default=[],
                        help="Comma-separated list of instrumental variables (optional)")
    parser.add_argument("--cluster", default=None, help="Optional column for cluster-robust SE ids")
    parser.add_argument("--weights", default=None, help="Optional column for observation weights")
    parser.add_argument("--precision", choices=list(PRECISION_FLAGS), default="float64")
    parser.add_argument("--drop-missing", action="store_true", help="Drop rows with missing values in relevant columns")
    parser.add_argument("--summary", action="store_true", help="Print dataset summary as JSON")
    parser.add_argument("--verbose", action="store_true", help="Log progress and timings")
    parser.add_argument("--workers", type=int, default=None, help="Number of CPU workers (default: all available cores)")
    return parser.parse_args()

def determine_format(path: pathlib.Path, requested: str) -> str:
    if requested != "auto":
        return requested
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return "parquet"
    if suffix in {".pkl", ".pickle"}:
        return "pickle"
    return "csv"

def load_dataframe(path: pathlib.Path, *, fmt: str, columns: Sequence[str]) -> pd.DataFrame:
    usecols = list(dict.fromkeys(columns))
    if fmt == "parquet":
        return pd.read_parquet(path, columns=usecols, engine="pyarrow")
    if fmt == "pickle":
        df = pd.read_pickle(path)
        missing = [col for col in usecols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in pickle: {missing}")
        return df[usecols]
    if fmt == "csv":
        return pd.read_csv(path, usecols=usecols)
    raise ValueError(f"Unsupported format: {fmt}")

def relabel_ids(series: pd.Series) -> np.ndarray:
    codes, uniques = pd.factorize(series, sort=False)
    if (codes < 0).any():
        raise ValueError(f"Column {series.name} contains missing values after cleaning")
    return codes.astype(np.int32, copy=False) + 1


def parallel_relabel(df: pd.DataFrame, columns: Sequence[str], workers: int, vlog) -> List[np.ndarray]:
    if len(columns) == 0:
        return []
    if workers <= 1 or len(columns) == 1:
        return [relabel_ids(df[col]) for col in columns]

    results = {col: None for col in columns}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {executor.submit(relabel_ids, df[col]): col for col in columns}
        for future in as_completed(future_map):
            col = future_map[future]
            results[col] = future.result()
    return [results[col] for col in columns]

def write_binary(path: pathlib.Path,
                 y: np.ndarray,
                 X: np.ndarray,
                 fe_ids: np.ndarray,
                 *,
                 instruments: Optional[np.ndarray],
                 cluster: Optional[np.ndarray],
                 weights: Optional[np.ndarray],
                 precision: str) -> None:
    import struct

    n_obs = y.shape[0]
    n_reg = X.shape[1]
    n_instr = 0 if instruments is None else instruments.shape[1]
    n_fe = fe_ids.shape[0]
    has_cluster = 1 if cluster is not None else 0
    has_weights = 1 if weights is not None else 0
    precision_flag = PRECISION_FLAGS[precision]

    float_dtype = np.dtype("<f4") if precision == "float32" else np.dtype("<f8")
    int_dtype = np.dtype("<i4")

    y_arr = np.asarray(y, dtype=float_dtype)
    X_arr = np.asarray(X, dtype=float_dtype, order="F")
    Z_arr = None
    if instruments is not None and n_instr > 0:
        Z_arr = np.asarray(instruments, dtype=float_dtype, order="F")
    fe_arr = np.asarray(fe_ids, dtype=int_dtype)
    cluster_arr = None if cluster is None else np.asarray(cluster, dtype=int_dtype)
    weight_arr = None if weights is None else np.asarray(weights, dtype=float_dtype)

    with path.open("wb") as f:
        f.write(struct.pack(HEADER_FORMAT, n_obs, n_reg, n_instr, n_fe, has_cluster, has_weights))
        f.write(struct.pack("<i", precision_flag))
        y_arr.tofile(f)
        X_arr.T.tofile(f)
        if Z_arr is not None:
            Z_arr.T.tofile(f)
        for d in range(fe_arr.shape[0]):
            fe_arr[d, :].tofile(f)
        if has_cluster:
            cluster_arr.tofile(f)
        if has_weights:
            weight_arr.tofile(f)

def main() -> None:
    args = parse_args()
    t0 = time.perf_counter()

    def vlog(message: str) -> None:
        if args.verbose:
            print(message, file=sys.stderr)

    cpu_total = os.cpu_count() or 1
    workers = args.workers if args.workers and args.workers > 0 else cpu_total
    vlog(f"Detected {cpu_total} logical CPUs; using {workers} worker(s)")

    all_columns: List[str] = [args.y] + args.x + args.fe + args.iv
    if args.cluster:
        all_columns.append(args.cluster)
    if args.weights:
        all_columns.append(args.weights)

    fmt = determine_format(args.input, args.input_format)
    vlog(f"Loading {args.input} (format={fmt})")
    t_load0 = time.perf_counter()
    df = load_dataframe(args.input, fmt=fmt, columns=all_columns)
    t_load1 = time.perf_counter()
    vlog(f"Loaded {len(df):,} rows in {t_load1 - t_load0:.3f} s")

    subset_cols = list(dict.fromkeys(all_columns))
    if args.drop_missing:
        t_clean0 = time.perf_counter()
        missing_mask = df[subset_cols].isna().any(axis=1)
        missing_rows = int(missing_mask.sum())
        if missing_rows:
            df = df.loc[~missing_mask].reset_index(drop=True)
            vlog(f"Dropped {missing_rows} rows containing missing values")
        else:
            vlog("No missing values detected; skipping drop")
        t_clean1 = time.perf_counter()
        vlog(f"Missing-value cleanup took {t_clean1 - t_clean0:.3f} s")
    else:
        if df[subset_cols].isna().any().any():
            raise ValueError("Dataset contains missing values; rerun with --drop-missing")

    df = df.reset_index(drop=True)
    n = len(df)
    if n == 0:
        raise ValueError("No observations remain after processing")

    vlog("Building numeric arrays")
    t_arrays0 = time.perf_counter()
    y = df[args.y].to_numpy(dtype=np.float64, copy=False)
    X = np.asfortranarray(df[args.x].to_numpy(dtype=np.float64, copy=False))
    Z = None
    if args.iv:
        Z = np.asfortranarray(df[args.iv].to_numpy(dtype=np.float64, copy=False))
    start_factor = time.perf_counter()
    fe_arrays = parallel_relabel(df, args.fe, workers, vlog)
    fe_ids = np.stack(fe_arrays, axis=0)
    t_arrays1 = time.perf_counter()
    vlog(f"Relabeled FE columns in {time.perf_counter() - start_factor:.3f} s")
    vlog(f"Finished numeric array build in {t_arrays1 - t_arrays0:.3f} s")

    cluster_arr = None
    if args.cluster:
        cluster_arr = relabel_ids(df[args.cluster])

    weight_arr = None
    if args.weights:
        weight_arr = df[args.weights].to_numpy(dtype=np.float64, copy=False)

    vlog(f"Writing binary output to {args.output}")
    t_write0 = time.perf_counter()
    write_binary(args.output, y, X, fe_ids, instruments=Z, cluster=cluster_arr, weights=weight_arr, precision=args.precision)
    t_write1 = time.perf_counter()
    vlog(f"Wrote binary file in {t_write1 - t_write0:.3f} s")

    if args.summary:
        info = {
            "n_obs": int(n),
            "n_regressors": int(X.shape[1]),
            "n_fe_dims": int(fe_ids.shape[0]),
            "n_instruments": int(0 if Z is None else Z.shape[1]),
            "clustered": args.cluster is not None,
            "weights": args.weights is not None,
            "precision": args.precision,
        }
        print(json.dumps(info, indent=2))

    t1 = time.perf_counter()
    vlog(f"Total elapsed: {t1 - t0:.3f} s")

if __name__ == "__main__":
    main()
