"""Data loading, normalization and correlation helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "load_matrix",
    "normalize_data",
    "pearson_correlation",
]


def load_matrix(source, sep: str = ",", has_header=None, has_index=None) -> np.ndarray:
    """Load a 2-D numeric matrix from a file path or array-like.

    Rows are the elements to be clustered (samples); columns are features.
    Accepts CSV / TSV / whitespace-delimited text.  A first textual row or
    column is auto-detected and dropped unless explicitly overridden.

    Parameters
    ----------
    source:
        File path (str) or an array-like already in memory.
    sep:
        Field separator for text files.
    has_header, has_index:
        Force header / index handling.  ``None`` means auto-detect.
    """
    if not isinstance(source, str):
        arr = np.asarray(source, dtype=float)
        if arr.ndim != 2:
            raise ValueError("Data must be a 2-D matrix (n_samples, n_features)")
        return arr

    # Detect header by trying a numeric read first.
    read_kwargs = {"sep": sep, "engine": "python"}
    header = 0 if has_header else None
    index_col = 0 if has_index else None

    if has_header is None or has_index is None:
        header, index_col = _sniff_layout(source, sep)

    df = pd.read_csv(source, header=header, index_col=index_col, **read_kwargs)
    # Keep only numeric columns; drop any that survived as text labels.
    numeric = df.apply(pd.to_numeric, errors="coerce")
    if numeric.isna().all(axis=0).any():
        numeric = numeric.dropna(axis=1, how="all")
    arr = numeric.to_numpy(dtype=float)
    if np.isnan(arr).any():
        raise ValueError(
            f"{source!r} contains non-numeric or missing values after parsing"
        )
    if arr.ndim != 2:
        raise ValueError("Loaded data is not a 2-D matrix")
    return arr


def _sniff_layout(path: str, sep: str):
    """Best-effort detection of header row / index column for a delimited file."""
    with open(path, "r") as fh:
        first = fh.readline().strip()
        second = fh.readline().strip()
    if not first:
        return None, None

    def _row_is_numeric(line: str) -> bool:
        if not line:
            return True
        tokens = line.split(sep) if sep in line else line.split()
        for tok in tokens:
            try:
                float(tok)
            except ValueError:
                return False
        return True

    header = None if _row_is_numeric(first) else 0
    # Index column: first token of the (numeric) data row is non-numeric.
    data_line = second if header == 0 else first
    index_col = None
    if data_line:
        tokens = data_line.split(sep) if sep in data_line else data_line.split()
        if tokens:
            try:
                float(tokens[0])
            except ValueError:
                index_col = 0
    return header, index_col


def normalize_data(data: np.ndarray, by_sample: bool = False) -> np.ndarray:
    """Standardize to zero mean and unit variance.

    ``by_sample=True`` normalizes each row (sample) independently -- the regime
    required by the correlation-based log-likelihood.  ``by_sample=False``
    normalizes each feature (column).  Zero-variance vectors are left centred
    (the denominator is clamped to 1 to avoid division by zero).
    """
    data = np.asarray(data, dtype=float)
    if data.ndim != 2:
        raise ValueError("Data must be a 2-D array (n_samples, n_features)")

    axis = 1 if by_sample else 0
    mean = np.mean(data, axis=axis, keepdims=True)
    std = np.std(data, axis=axis, keepdims=True)
    std = np.where(std == 0, 1.0, std)
    return (data - mean) / std


def pearson_correlation(data: np.ndarray, rowvar: bool = True) -> np.ndarray:
    """Pearson correlation matrix.

    With ``rowvar=True`` (default) correlations are between rows (samples),
    yielding an (n_samples, n_samples) matrix -- the input the
    correlation-based log-likelihood expects.  Any NaNs that arise from
    constant vectors are replaced by zeros (off-diagonal) / ones (diagonal).
    """
    data = np.asarray(data, dtype=float)
    cor = np.corrcoef(data, rowvar=rowvar)
    cor = np.atleast_2d(cor)
    if np.isnan(cor).any():
        cor = np.nan_to_num(cor, nan=0.0)
        np.fill_diagonal(cor, 1.0)
    return cor
