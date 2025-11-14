#!/usr/bin/env python3
"""Convert tabular data into the fe_gpu binary format."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd

HEADER_FORMAT = "<qiiii"
PRECISION_FLAGS = {
    "float64": 0,
    "float32": 1,
}

def parse_columns(value: str, *, allow_empty: bool = False) -> List[str]:
    items = [col.strip() for col in value.split(',') if col.strip()]
    if not items and not allow_empty:
        raise argparse.ArgumentTypeError("At least one column must be provided")
    return items

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
    parser.add_argument("--cluster", default=None, help="Optional column for cluster-robust SE ids")
    parser.add_argument("--weights", default=None, help="Optional column for observation weights")
    parser.add_argument("--precision", choices=list(PRECISION_FLAGS), default="float64")
    parser.add_argument("--drop-missing", action="store_true", help="Drop rows with missing values in relevant columns")
    parser.add_argument("--summary", action="store_true", help="Print dataset summary as JSON")
    parser.add_argument("--verbose", action="store_true", help="Log progress and timings")
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

def write_binary(path: pathlib.Path,
                 y: np.ndarray,
                 X: np.ndarray,
                 fe_ids: np.ndarray,
                 *,
                 cluster: np.ndarray | None,
                 weights: np.ndarray | None,
                 precision: str) -> None:
    import struct

    n_obs = y.shape[0]
    n_reg = X.shape[1]
    n_fe = fe_ids.shape[0]
    has_cluster = 1 if cluster is not None else 0
    has_weights = 1 if weights is not None else 0
    precision_flag = PRECISION_FLAGS[precision]

    dtype = np.float32 if precision == "float32" else np.float64
    y_bytes = np.asarray(y, dtype=dtype).tobytes(order="C")
    W_bytes = np.asarray(X, dtype=dtype, order="F").tobytes(order="F")
    fe_bytes = fe_ids.astype(np.int32, copy=False).tobytes(order="C")
    cluster_bytes = None if cluster is None else cluster.astype(np.int32, copy=False).tobytes(order="C")
    weight_bytes = None if weights is None else np.asarray(weights, dtype=dtype).tobytes(order="C")

    with path.open("wb") as f:
        f.write(struct.pack(HEADER_FORMAT, n_obs, n_reg, n_fe, has_cluster, has_weights))
        f.write(struct.pack("<i", precision_flag))
        f.write(y_bytes)
        f.write(W_bytes)
        f.write(fe_bytes)
        if has_cluster:
            f.write(cluster_bytes)
        if has_weights:
            f.write(weight_bytes)

def main() -> None:
    args = parse_args()
    import time
    t0 = time.perf_counter()

    def vlog(message: str) -> None:
        if args.verbose:
            print(message, file=sys.stderr)

    all_columns: List[str] = [args.y] + args.x + args.fe
    if args.cluster:
        all_columns.append(args.cluster)
    if args.weights:
        all_columns.append(args.weights)

    fmt = determine_format(args.input, args.input_format)
    vlog(f"Loading {args.input} (format={fmt})")
    df = load_dataframe(args.input, fmt=fmt, columns=all_columns)
    vlog(f"Loaded {len(df):,} rows")

    required = set(all_columns)
    subset_cols = list(required)
    if args.drop_missing:
        before = len(df)
        df = df.dropna(subset=subset_cols)
        dropped = before - len(df)
        if dropped:
            vlog(f"Dropped {dropped} rows containing missing values")
    elif df[subset_cols].isna().any().any():
        raise ValueError("Dataset contains missing values; rerun with --drop-missing")

    df = df.reset_index(drop=True)
    n = len(df)
    if n == 0:
        raise ValueError("No observations remain after processing")

    vlog("Building numeric arrays")
    y = df[args.y].to_numpy(dtype=np.float64, copy=False)
    X = np.asfortranarray(df[args.x].to_numpy(dtype=np.float64, copy=False))
    fe_arrays = []
    for col in args.fe:
        fe_arrays.append(relabel_ids(df[col]))
    fe_ids = np.stack(fe_arrays, axis=0)

    cluster_arr = None
    if args.cluster:
        cluster_arr = relabel_ids(df[args.cluster])

    weight_arr = None
    if args.weights:
        weight_arr = df[args.weights].to_numpy(dtype=np.float64, copy=False)

    vlog(f"Writing binary output to {args.output}")
    write_binary(args.output, y, X, fe_ids, cluster=cluster_arr, weights=weight_arr, precision=args.precision)

    if args.summary:
        info = {
            "n_obs": int(n),
            "n_regressors": int(X.shape[1]),
            "n_fe_dims": int(fe_ids.shape[0]),
            "clustered": args.cluster is not None,
            "weights": args.weights is not None,
            "precision": args.precision,
        }
        print(json.dumps(info, indent=2))

    t1 = time.perf_counter()
    vlog(f"Total elapsed: {t1 - t0:.3f} s")

if __name__ == "__main__":
    main()
