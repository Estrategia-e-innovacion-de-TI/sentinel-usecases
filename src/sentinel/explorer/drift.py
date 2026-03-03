"""Statistical drift detection utilities."""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def detect_drift(
    df: pd.DataFrame,
    column: str,
    window: int,
    method: str = "ks",
) -> List[Dict]:
    """Detect statistical drift in a column using a sliding window.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame (must be sorted by time).
    column : str
        Column to check for drift.
    window : int
        Size of each comparison window.
    method : str
        Detection method. Currently only ``"ks"`` (Kolmogorov-Smirnov) is supported.

    Returns
    -------
    list of dict
        Each dict contains ``start_idx``, ``end_idx``, ``statistic``, ``p_value``,
        and ``drifted`` (True if p_value < 0.05).
    """
    results: List[Dict] = []
    series = df[column].dropna().values

    if len(series) < 2 * window:
        return results

    reference = series[:window]

    for i in range(window, len(series) - window + 1, window):
        test = series[i : i + window]
        stat, p = ks_2samp(reference, test)
        results.append({
            "start_idx": i,
            "end_idx": i + window,
            "statistic": round(float(stat), 4),
            "p_value": round(float(p), 6),
            "drifted": p < 0.05,
        })

    return results
