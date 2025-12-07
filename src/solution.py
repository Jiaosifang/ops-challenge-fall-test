import numpy as np
import polars as pl
from numba import njit, prange


@njit(fastmath=True, nogil=True)
def _rolling_beta_stateful(symbol_idx: np.ndarray, low: np.ndarray, close: np.ndarray, window: int) -> np.ndarray:
    n = low.shape[0]

    n_symbols = np.max(symbol_idx) + 1

    buffer_low = np.empty((n_symbols, window), dtype=np.float64)
    buffer_close = np.empty((n_symbols, window), dtype=np.float64)
    counts = np.zeros(n_symbols, dtype=np.int64)
    positions = np.zeros(n_symbols, dtype=np.int64)
    sum_low = np.zeros(n_symbols, dtype=np.float64)
    sum_close = np.zeros(n_symbols, dtype=np.float64)
    sum_low_sq = np.zeros(n_symbols, dtype=np.float64)
    sum_low_close = np.zeros(n_symbols, dtype=np.float64)

    out = np.empty(n, dtype=np.float64)

    for i in range(n):
        sid = symbol_idx[i]
        pos = positions[sid]
        count = counts[sid]

        if count == window:
            old_low = buffer_low[sid, pos]
            old_close = buffer_close[sid, pos]
            sum_low[sid] -= old_low
            sum_close[sid] -= old_close
            sum_low_sq[sid] -= old_low * old_low
            sum_low_close[sid] -= old_low * old_close
        else:
            count += 1
            counts[sid] = count

        cur_low = low[i]
        cur_close = close[i]
        buffer_low[sid, pos] = cur_low
        buffer_close[sid, pos] = cur_close

        sum_low[sid] += cur_low
        sum_close[sid] += cur_close
        sum_low_sq[sid] += cur_low * cur_low
        sum_low_close[sid] += cur_low * cur_close

        pos += 1
        if pos == window:
            pos = 0
        positions[sid] = pos

        if count < 2:
            out[i] = np.nan
            continue

        inv_count = 1.0 / count
        cov = (sum_low_close[sid] - (sum_low[sid] * sum_close[sid]) * inv_count) / (count - 1.0)
        var_low = (sum_low_sq[sid] - (sum_low[sid] * sum_low[sid]) * inv_count) / (count - 1.0)

        if var_low < 1e-6 + 1e-10:
            out[i] = 0.0
        else:
            out[i] = cov / var_low

    return out


def ops_rolling_regbeta(input_path: str, window: int = 20) -> np.ndarray:
    df = (
        pl.scan_parquet(input_path)
        .with_columns(
            [
                pl.col("symbol").cast(pl.Categorical),
                pl.col("Low").cast(pl.Float64),
                pl.col("Close").cast(pl.Float64),
            ]
        )
        .select(["symbol", "Low", "Close"])
        .collect()
    )

    low = df["Low"].to_numpy().astype(np.float64, copy=False)
    close = df["Close"].to_numpy().astype(np.float64, copy=False)
    symbol_codes = df["symbol"].to_physical().to_numpy().astype(np.int64, copy=False)

    beta = _rolling_beta_stateful(symbol_codes, low, close, window)

    return beta.reshape(-1, 1)

