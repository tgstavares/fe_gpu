#!/usr/bin/env python3
"""Generate synthetic panel data matching the fe_gpu binary format."""

import argparse
import pathlib
import struct
from typing import Dict, Optional, Sequence

import numpy as np

HEADER_FORMAT = '<qiiiii'
PRECISION_FLAGS = {
    'float64': 0,
    'float32': 1,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic data.bin files for fe_gpu",
        fromfile_prefix_chars='@'
    )
    parser.add_argument('--workers', type=int, default=1000, help='Number of unique workers (default: 1000)')
    parser.add_argument('--firms', type=int, default=200, help='Number of unique firms (default: 200)')
    parser.add_argument('--periods', type=int, default=8, help='Number of time periods (default: 8)')
    parser.add_argument('--intercept', type=float, default=1.0, help='Intercept term b0 (default: 1.0)')
    parser.add_argument('--betas', type=float, nargs=2, default=[0.4, -0.2],
                        metavar=('b1', 'b2'),
                        help='Coefficients for tenure and sick-shock (default: 0.4 -0.2)')
    parser.add_argument('--extra-vars', type=int, default=0,
                        help='Number of additional time-varying regressors (default: 0)')
    parser.add_argument('--extra-coeffs', type=float, nargs='*', default=None,
                        metavar='c', help='Coefficients for extra regressors (provide one per extra-var)')
    parser.add_argument('--extra-std', type=float, default=0.3,
                        help='Std. dev. for additional regressors (default: 0.3)')
    parser.add_argument('--noise-std', type=float, default=0.5, help='Std. dev. of epsilon (default: 0.5)')
    parser.add_argument('--sick-std', type=float, default=0.5, help='Std. dev. of sick-shock regressor (default: 0.5)')
    parser.add_argument('--iv-vars', type=int, default=0,
                        help='Number of synthetic instrumental variables to generate (default: 0)')
    parser.add_argument('--iv-noise-std', type=float, default=0.25,
                        help='Std. dev. of noise added to each instrument (default: 0.25)')
    parser.add_argument('--endog-cols', type=int, nargs='*', default=None,
                        help='1-based indices of regressors treated as endogenous (default: none)')
    parser.add_argument('--precision', choices=PRECISION_FLAGS.keys(), default='float64',
                        help='Floating-point precision for storage (default: float64)')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed (default: 1234)')
    parser.add_argument('-o', '--output', type=pathlib.Path, default=pathlib.Path('synthetic_data.bin'),
                        help='Output binary path (default: synthetic_data.bin)')
    parser.add_argument('--parquet-output', type=pathlib.Path, default=None,
                        help='Optional Parquet output path for Julia/analytics interoperability')
    return parser.parse_args()


def write_string(f, text: str) -> None:
    encoded = text.encode('utf-8')
    f.write(struct.pack('<i', len(encoded)))
    if encoded:
        f.write(encoded)


def write_string_list(f, items: Sequence[str]) -> None:
    f.write(struct.pack('<i', len(items)))
    for item in items:
        write_string(f, item)


def write_metadata_block(f, metadata: Optional[dict]) -> None:
    if not metadata:
        return
    f.write(b'META')
    f.write(struct.pack('<i', 1))
    write_string(f, metadata.get('y', ''))
    write_string_list(f, metadata.get('x', []))
    write_string_list(f, metadata.get('iv', []))
    write_string_list(f, metadata.get('fe', []))
    write_string(f, metadata.get('cluster', ''))
    write_string(f, metadata.get('weights', ''))


def build_panel(args: argparse.Namespace):
    rng = np.random.default_rng(args.seed)

    workers = np.arange(args.workers, dtype=np.int32) + 1
    times = np.arange(args.periods, dtype=np.int32) + 1

    worker_obs = np.repeat(workers, args.periods)
    time_obs = np.tile(times, args.workers)
    n_obs = worker_obs.size

    firm_obs = rng.integers(1, args.firms + 1, size=n_obs, dtype=np.int32)

    tenure = np.zeros(n_obs, dtype=np.float64)
    last_firm = np.full(args.workers, -1, dtype=np.int32)
    tenure_counter = np.zeros(args.workers, dtype=np.int32)
    for idx in range(n_obs):
        w = worker_obs[idx] - 1
        firm = firm_obs[idx]
        if firm == last_firm[w]:
            tenure_counter[w] += 1
        else:
            tenure_counter[w] = 0
            last_firm[w] = firm
        tenure[idx] = tenure_counter[w]

    sick_shock = rng.normal(0.0, args.sick_std, size=n_obs)

    extra_vars = max(0, args.extra_vars)
    extra_coeffs = None
    if extra_vars > 0:
        if args.extra_coeffs is not None and len(args.extra_coeffs) > 0:
            if len(args.extra_coeffs) != extra_vars:
                raise ValueError('--extra-coeffs must match --extra-vars')
            extra_coeffs = np.array(args.extra_coeffs, dtype=np.float64)
        else:
            extra_coeffs = rng.normal(0.0, 0.5, size=extra_vars)
        extras = rng.normal(0.0, args.extra_std, size=(extra_vars, n_obs))
        extras += 0.02 * (time_obs - 1)
    else:
        extras = np.zeros((0, n_obs), dtype=np.float64)
        extra_coeffs = np.zeros(0, dtype=np.float64)
    args.extra_coeffs_actual = extra_coeffs.copy()

    worker_fe = rng.normal(0.0, 0.3, size=args.workers)
    firm_fe = rng.normal(0.0, 0.2, size=args.firms)
    time_fe = rng.normal(0.0, 0.1, size=args.periods)

    eps = rng.normal(0.0, args.noise_std, size=n_obs)
    b1, b2 = args.betas
    wages = args.intercept + b1 * tenure + b2 * sick_shock
    if extra_vars > 0:
        wages += np.dot(extra_coeffs, extras)
    wages += worker_fe[worker_obs - 1] + firm_fe[firm_obs - 1] + time_fe[time_obs - 1] + eps

    if extra_vars > 0:
        W = np.column_stack((tenure, sick_shock, extras.T)).astype(np.float64)
    else:
        W = np.column_stack((tenure, sick_shock)).astype(np.float64)
    regressor_names = ['tenure', 'sick_shock']
    if extra_vars > 0:
        regressor_names.extend([f'extra_{j + 1}' for j in range(extra_vars)])

    iv_count = max(0, args.iv_vars)
    if iv_count > 0:
        instruments = np.empty((iv_count, n_obs), dtype=np.float64)
        base_cols = W.shape[1]
        if args.endog_cols:
            valid_cols = [max(1, min(base_cols, idx)) - 1 for idx in args.endog_cols]
            if len(valid_cols) == 0:
                valid_cols = list(range(base_cols))
        else:
            valid_cols = list(range(base_cols))
        for j in range(iv_count):
            source_col = valid_cols[j % len(valid_cols)]
            instruments[j, :] = W[:, source_col] + rng.normal(0.0, args.iv_noise_std, size=n_obs)
            instruments[j, :] += rng.normal(0.0, 0.1, size=n_obs)
    else:
        instruments = np.zeros((0, n_obs), dtype=np.float64)
    instrument_names = [f'iv_{j + 1}' for j in range(instruments.shape[0])]

    fe_ids = np.vstack((worker_obs, firm_obs, time_obs)).astype(np.int32)

    parquet_columns: Dict[str, np.ndarray] = {
        'worker': worker_obs.astype(np.int32, copy=False),
        'firm': firm_obs.astype(np.int32, copy=False),
        'time': time_obs.astype(np.int32, copy=False),
        'tenure': tenure.astype(np.float64, copy=False),
        'sick_shock': sick_shock.astype(np.float64, copy=False),
        'wage': wages.astype(np.float64, copy=False),
    }
    if extras.shape[0] > 0:
        for j in range(extras.shape[0]):
            parquet_columns[f'extra_{j + 1}'] = extras[j].astype(np.float64, copy=False)
    if instruments.shape[0] > 0:
        for j in range(instruments.shape[0]):
            parquet_columns[f'iv_{j + 1}'] = instruments[j].astype(np.float64, copy=False)

    return wages, W, instruments, fe_ids, parquet_columns, regressor_names, instrument_names


def write_binary(path: pathlib.Path, wages: np.ndarray, W: np.ndarray, fe_ids: np.ndarray,
                 precision_flag: int, metadata: Optional[dict] = None,
                 instruments: Optional[np.ndarray] = None):
    n_obs = wages.shape[0]
    n_reg = W.shape[1]
    n_instr = 0 if instruments is None else instruments.shape[0]
    n_fe = fe_ids.shape[0]

    dtype = np.float32 if precision_flag == PRECISION_FLAGS['float32'] else np.float64
    wages_out = wages.astype(dtype)
    W_out = W.astype(dtype, order='F')
    if instruments is not None and n_instr > 0:
        Z_out = instruments.astype(dtype, order='F')
    else:
        Z_out = None

    with path.open('wb') as f:
        f.write(struct.pack('<q', n_obs))
        f.write(struct.pack('<i', n_reg))
        f.write(struct.pack('<i', n_instr))
        f.write(struct.pack('<i', n_fe))
        f.write(struct.pack('<i', 0))  # has_cluster
        f.write(struct.pack('<i', 0))  # has_weights
        f.write(struct.pack('<i', precision_flag))
        f.write(wages_out.tobytes(order='C'))
        f.write(W_out.tobytes(order='F'))
        if Z_out is not None:
            f.write(Z_out.tobytes(order='F'))
        for d in range(n_fe):
            f.write(fe_ids[d].astype('<i4', copy=False).tobytes(order='C'))
        write_metadata_block(f, metadata)


def write_parquet(path: pathlib.Path, columns: Dict[str, np.ndarray]):
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SystemExit('pyarrow is required to write Parquet output. Install via `pip install pyarrow`.') from exc

    arrays = {name: pa.array(col) for name, col in columns.items()}
    table = pa.table(arrays)
    pq.write_table(table, path)


def main():
    args = parse_args()
    wages, W, Z, fe_ids, parquet_columns, reg_names, instrument_names = build_panel(args)
    metadata = {
        'y': 'wage',
        'x': reg_names,
        'iv': instrument_names,
        'fe': ['worker', 'firm', 'time'],
        'cluster': '',
        'weights': '',
    }
    write_binary(
        args.output,
        wages,
        W,
        fe_ids,
        PRECISION_FLAGS[args.precision],
        metadata=metadata,
        instruments=Z if Z.size > 0 else None,
    )
    bytes_len = args.output.stat().st_size
    print(f"Wrote {args.output} with {W.shape[0]} observations, {W.shape[1]} regressors, "
          f"{Z.shape[0]} instruments, {fe_ids.shape[0]} FE dims ({bytes_len} bytes)")
    if args.extra_vars > 0:
        coeffs = args.extra_coeffs_actual
        print(f"Extra coefficients used: {coeffs.tolist()}")
    if args.endog_cols:
        endog_str = ','.join(str(idx) for idx in args.endog_cols)
        print(f"Endogenous regressors (1-based indices): {endog_str}")
        print(f"Suggested fe_gpu flag: --iv-cols {endog_str}")
    if args.parquet_output is not None:
        write_parquet(args.parquet_output, parquet_columns)
        print(f"Wrote {args.parquet_output} (Parquet) mirroring the binary dataset")


if __name__ == '__main__':
    main()
