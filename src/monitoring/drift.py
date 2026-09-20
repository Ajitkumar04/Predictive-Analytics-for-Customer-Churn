from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def _prepare_numeric(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return values


def _shared_bins(reference: pd.Series, current: pd.Series, bins: int = 5) -> np.ndarray:
    ref = _prepare_numeric(reference)
    cur = _prepare_numeric(current)

    if ref.empty or cur.empty:
        return np.array([0.0, 1.0])

    combined = pd.concat([ref, cur], ignore_index=True)
    if combined.nunique() == 1:
        value = float(combined.iloc[0])
        return np.array([value - 1e-6, value + 1e-6])

    safe_bins = max(2, min(int(bins), len(combined)))
    start = float(combined.min())
    end = float(combined.max())
    if np.isclose(start, end):
        return np.array([start - 1e-6, end + 1e-6])

    return np.linspace(start, end, safe_bins + 1)


def compute_psi(reference: pd.Series, current: pd.Series, bins: int = 5) -> float:
    """Compute the Population Stability Index (PSI) using shared bins."""
    ref = _prepare_numeric(reference)
    cur = _prepare_numeric(current)

    if ref.empty or cur.empty:
        return 0.0

    if ref.nunique() == 1 and cur.nunique() == 1:
        if float(ref.iloc[0]) == float(cur.iloc[0]):
            return 0.0

    edges = _shared_bins(ref, cur, bins=bins)
    ref_hist, _ = np.histogram(ref.to_numpy(dtype=float), bins=edges)
    cur_hist, _ = np.histogram(cur.to_numpy(dtype=float), bins=edges)

    ref_total = ref_hist.sum()
    cur_total = cur_hist.sum()
    if ref_total == 0 or cur_total == 0:
        return 0.0

    ref_pct = ref_hist / ref_total
    cur_pct = cur_hist / cur_total

    valid = (ref_pct > 0) | (cur_pct > 0)
    ref_pct = ref_pct[valid]
    cur_pct = cur_pct[valid]

    epsilon = 1e-6
    ref_pct = np.clip(ref_pct, epsilon, None)
    cur_pct = np.clip(cur_pct, epsilon, None)

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


def detect_data_drift(reference: pd.DataFrame, current: pd.DataFrame, columns: Iterable[str]) -> dict[str, dict[str, float | str]]:
    """Return a simple drift report using PSI-based checks."""
    drift_report: dict[str, dict[str, float | str]] = {}

    for column in columns:
        ref_series = reference[column]
        cur_series = current[column]
        psi_value = compute_psi(ref_series, cur_series)

        if psi_value > 0.25:
            status = "drift_detected"
        elif psi_value > 0.1:
            status = "warning"
        else:
            status = "stable"

        drift_report[column] = {
            "psi": round(float(psi_value), 6),
            "status": status,
        }

    return drift_report
