"""Shape-state variables from confirmed extremum windows.

Confirmed extrema come from the same online CPD -> Direction -> Excep -> Status -> MinMax 
process implemented in ``cpd.market_state_vector.MarketStateVector``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from cpd.market_state_vector import MarketStateVector


@dataclass(frozen=True)
class ConfirmedExtremum:
    """One online-confirmed high or low event."""

    kind: str
    idx: int
    timestamp: pd.Timestamp
    value: float
    confirmed_idx: int
    confirmed_timestamp: pd.Timestamp


def _safe_div(num: float, den: float) -> float:
    if not np.isfinite(num) or not np.isfinite(den) or abs(den) < 1e-12:
        return np.nan
    return float(num / den)


def _normalise_ohlcv(data: pd.DataFrame) -> pd.DataFrame:
    """Return a datetime-indexed frame with Open/High/Low/Close/Volume if present."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")

    df = data.copy()
    rename = {
        "date": "Date",
        "datetime": "Date",
        "timestamp": "Date",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
        "amount": "Amount",
        "symbol": "symbol",
        "name": "name",
    }
    df = df.rename(columns={c: rename.get(c, c) for c in df.columns})

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")
    elif not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("data must have a Date/date column or a DatetimeIndex")

    required = {"Open", "High", "Low", "Close"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"data is missing OHLC columns: {sorted(missing)}")

    keep = [c for c in ["symbol", "name", "Open", "High", "Low", "Close", "Volume", "Amount"] if c in df.columns]
    df = df[keep].sort_index()
    for col in ["Open", "High", "Low", "Close", "Volume", "Amount"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    return df


def _append_extremum(extrema: List[ConfirmedExtremum], ext: ConfirmedExtremum) -> bool:
    """Append an alternating extremum sequence, replacing same-kind duplicates."""
    if not extrema:
        extrema.append(ext)
        return True

    last = extrema[-1]
    if last.idx == ext.idx and last.kind == ext.kind:
        return False

    if last.kind != ext.kind:
        extrema.append(ext)
        return True

    if ext.kind == "high" and ext.value >= last.value:
        extrema[-1] = ext
        return True
    if ext.kind == "low" and ext.value <= last.value:
        extrema[-1] = ext
        return True
    return False


def _build_msv_confirmed_extrema(
    ohlc: pd.DataFrame,
    *,
    detector_method: str,
    detector_q: float,
    detector_kwargs: Optional[Dict[str, Any]],
    cpd_confirm_lag: int,
    reset_structure_on_cpd: bool,
) -> tuple[pd.DataFrame, Dict[int, List[ConfirmedExtremum]]]:
    """Use the original MarketStateVector online process to obtain extrema."""
    builder = MarketStateVector(
        detector_method=detector_method,
        detector_q=detector_q,
        detector_kwargs=detector_kwargs,
        cpd_confirm_lag=cpd_confirm_lag,
        reset_structure_on_cpd=reset_structure_on_cpd,
    )
    state = builder.build_market_state_vector(data=ohlc[["Open", "High", "Low", "Close"]])
    state = state.reset_index().rename(columns={"timestamp": "date"})

    dates = pd.DatetimeIndex(ohlc.index)
    by_confirm: Dict[int, List[ConfirmedExtremum]] = {}
    confirmed_rows = state[state["new_extremum_confirmed"].fillna(False)].copy()
    for row in confirmed_rows.itertuples(index=False):
        kind = getattr(row, "confirmed_extremum_kind")
        idx_value = getattr(row, "confirmed_extremum_idx")
        value = getattr(row, "confirmed_extremum_value")
        confirmed_at = pd.Timestamp(getattr(row, "date"))
        if kind not in {"high", "low"} or pd.isna(idx_value) or pd.isna(value):
            continue

        idx = int(idx_value)
        if idx < 0 or idx >= len(dates):
            continue
        confirm_idx = int(ohlc.index.get_loc(confirmed_at))
        ext = ConfirmedExtremum(
            kind=str(kind),
            idx=idx,
            timestamp=dates[idx],
            value=float(value),
            confirmed_idx=confirm_idx,
            confirmed_timestamp=confirmed_at,
        )
        by_confirm.setdefault(confirm_idx, []).append(ext)

    keep = [
        "date",
        "direction",
        "excep",
        "status",
        "current_trend",
        "current_trend_code",
        "current_phase",
        "trend_candidate",
        "new_extremum_confirmed",
        "confirmed_extremum_kind",
        "confirmed_extremum_idx",
        "confirmed_extremum_value",
        "cpd_confirm_event",
        "time_since_last_cpd",
        "time_since_last_cpd_confirm",
    ]
    diagnostics = state[[col for col in keep if col in state.columns]].copy()
    return diagnostics, by_confirm


def _window_features(
    extrema: Sequence[ConfirmedExtremum],
    close_t: float,
    current_idx: int,
) -> Dict[str, float]:
    """Compute variables for the latest four-extremum structural window."""
    if len(extrema) < 4:
        return {}

    window = list(extrema[-4:])
    highs = [e for e in window if e.kind == "high"]
    lows = [e for e in window if e.kind == "low"]
    if len(highs) != 2 or len(lows) != 2:
        return {}

    h1, h2 = sorted(highs, key=lambda e: e.idx)
    l1, l2 = sorted(lows, key=lambda e: e.idx)
    e0, e3 = window[0], window[-1]

    high_gap = h2.idx - h1.idx
    low_gap = l2.idx - l1.idx
    window_bars = e3.idx - e0.idx
    bars_since_window_end = current_idx - e3.idx
    leg_gaps = np.diff([e.idx for e in extrema])
    positive_leg_gaps = leg_gaps[leg_gaps > 0]
    median_leg_gap = float(np.median(positive_leg_gaps[-20:])) if len(positive_leg_gaps) else np.nan
    upper = max(h1.value, h2.value)
    lower = min(l1.value, l2.value)
    range_ = upper - lower
    width1 = h1.value - l1.value
    width2 = h2.value - l2.value
    dh = h2.value - h1.value
    dl = l2.value - l1.value
    upper_line_t = h1.value + _safe_div(h2.value - h1.value, high_gap) * (current_idx - h1.idx)
    lower_line_t = l1.value + _safe_div(l2.value - l1.value, low_gap) * (current_idx - l1.idx)
    upper_break = float(close_t > upper_line_t) if np.isfinite(upper_line_t) else np.nan
    lower_break = float(close_t < lower_line_t) if np.isfinite(lower_line_t) else np.nan

    return {
        "starts_high": 1.0 if e0.kind == "high" else 0.0,
        "high1": h1.value,
        "high2": h2.value,
        "low1": l1.value,
        "low2": l2.value,
        "high_change": dh,
        "low_change": dl,
        "expansion": dh - dl if np.isfinite(dh) and np.isfinite(dl) else np.nan,
        "range_pos": _safe_div(close_t - lower, range_),
        "upper_break": upper_break,
        "lower_break": lower_break,
        "structure_dir": dh + dl if np.isfinite(dh) and np.isfinite(dl) else np.nan,
        "range_pct": range_,
        "last_width_pct": width2,
        "move_age": _safe_div(bars_since_window_end, median_leg_gap),
        "high_speed": _safe_div(abs(dh), high_gap),
        "low_speed": _safe_div(abs(dl), low_gap),
        "extrema_freq": _safe_div(4.0, window_bars),
    }


def _nan_prefixed_features(prefix: str, feature_names: Iterable[str]) -> Dict[str, float]:
    return {f"{prefix}_{name}": np.nan for name in feature_names}


BASE_FEATURE_NAMES = [
    "starts_high",
    "high1",
    "high2",
    "low1",
    "low2",
    "high_change",
    "low_change",
    "expansion",
    "range_pos",
    "upper_break",
    "lower_break",
    "structure_dir",
    "range_pct",
    "last_width_pct",
    "move_age",
    "high_speed",
    "low_speed",
    "extrema_freq",
]


def build_shape_state_variables(
    data: pd.DataFrame,
    stock_id: Optional[str] = None,
    *,
    log_price: bool = False,
    n_lags: int = 0,
    decay: Optional[float] = None,
    detector_method: str = "cusum",
    detector_q: float = 1.0,
    detector_kwargs: Optional[Dict[str, Any]] = None,
    cpd_confirm_lag: int = 0,
    reset_structure_on_cpd: bool = False,
) -> pd.DataFrame:
    """Return one shape-state row per bar for a single stock.

    Parameters
    ----------
    data:
        Single-stock OHLCV data.  Lowercase AKShare columns are accepted.
    stock_id:
        Optional stock id.  If omitted, the first ``symbol`` value is used when
        available.
    n_lags:
        Number of previous structural windows to append as ``lag1_*`` etc.
        The default is 0 so the first-pass analysis uses only the current
        window ``Z_k``.
    decay:
        Optional geometric weight applied to lagged window variables.  For
        example ``decay=0.8`` stores ``0.8 * lag1_high_change``.
    detector_method, detector_q, detector_kwargs, cpd_confirm_lag, reset_structure_on_cpd:
        Passed to the original ``MarketStateVector`` online method.
    """
    ohlc = _normalise_ohlcv(data)
    if ohlc.empty:
        raise ValueError("cannot build variables from empty data")

    if log_price:
        price_cols = ["Open", "High", "Low", "Close"]
        valid = (ohlc[price_cols] > 0).all(axis=1)
        ohlc = ohlc.loc[valid].copy()
        if ohlc.empty:
            raise ValueError("cannot log-transform data without positive OHLC values")
        ohlc.loc[:, price_cols] = np.log(ohlc[price_cols].astype(float))

    if stock_id is None:
        stock_id = str(ohlc["symbol"].iloc[0]).zfill(6) if "symbol" in ohlc.columns else ""

    diagnostics, confirmed_by_bar = _build_msv_confirmed_extrema(
        ohlc=ohlc,
        detector_method=detector_method,
        detector_q=float(detector_q),
        detector_kwargs=detector_kwargs,
        cpd_confirm_lag=int(cpd_confirm_lag),
        reset_structure_on_cpd=bool(reset_structure_on_cpd),
    )

    closes = ohlc["Close"].to_numpy(float)
    rows: List[Dict[str, Any]] = []
    extrema: List[ConfirmedExtremum] = []
    window_history: List[Dict[str, float]] = []

    for i, ts in enumerate(ohlc.index):
        new_extrema = confirmed_by_bar.get(i, [])
        for ext in new_extrema:
            _append_extremum(extrema, ext)
            features = _window_features(extrema, closes[i], i)
            if features:
                window_history.append(features)

        current = _window_features(extrema, closes[i], i)
        has_window = bool(current)
        row: Dict[str, Any] = {
            "stock_id": stock_id,
            "date": pd.Timestamp(ts),
            "open": float(ohlc["Open"].iloc[i]),
            "high": float(ohlc["High"].iloc[i]),
            "low": float(ohlc["Low"].iloc[i]),
            "close": float(ohlc["Close"].iloc[i]),
            "num_confirmed_extrema": int(len(extrema)),
            "has_structural_window": bool(has_window),
        }
        if "Volume" in ohlc.columns:
            row["volume"] = float(ohlc["Volume"].iloc[i])
        if "Amount" in ohlc.columns:
            row["amount"] = float(ohlc["Amount"].iloc[i])

        row.update({name: current.get(name, np.nan) for name in BASE_FEATURE_NAMES})

        for lag in range(1, int(n_lags) + 1):
            prefix = f"lag{lag}"
            source_idx = len(window_history) - 1 - lag
            if source_idx < 0:
                row.update(_nan_prefixed_features(prefix, BASE_FEATURE_NAMES))
                continue
            lag_values = dict(window_history[source_idx])
            if decay is not None:
                weight = float(decay) ** lag
                for key, value in lag_values.items():
                    if np.isfinite(value):
                        lag_values[key] = value * weight
            row.update({f"{prefix}_{key}": value for key, value in lag_values.items()})

        rows.append(row)

    out = pd.DataFrame(rows)
    out["date"] = pd.to_datetime(out["date"])
    if diagnostics is not None and not diagnostics.empty:
        diagnostics["date"] = pd.to_datetime(diagnostics["date"])
        out = out.merge(diagnostics, on="date", how="left")
    return out


def build_shape_state_variables_from_csv(path: str | pd.PathLike[str], **kwargs: Any) -> pd.DataFrame:
    """Convenience wrapper for one downloaded AKShare CSV."""
    data = pd.read_csv(path, dtype={"symbol": str})
    if "symbol" in data.columns and "stock_id" not in kwargs:
        kwargs["stock_id"] = str(data["symbol"].iloc[0]).zfill(6)
    return build_shape_state_variables(data, **kwargs)


__all__ = [
    "ConfirmedExtremum",
    "BASE_FEATURE_NAMES",
    "build_shape_state_variables",
    "build_shape_state_variables_from_csv",
]
