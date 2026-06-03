"""Online market state vector construction based on CPD + Automatic 1-2-3 logic.

This module keeps the existing change-point detector unchanged. It uses CPD 
only to obtain an online regime direction, then follows an Automatic
One-Two-Three style state machine:

    Direction -> Excep -> Status -> MinMax process -> Trend structure -> State vector

The implementation is strictly online.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .brk_detector import Brk

# ---------------------------------------------------------------------------
# Data loading / validation
# ---------------------------------------------------------------------------


def _ensure_ohlc_frame(data: pd.DataFrame) -> pd.DataFrame:
    """Return a clean OHLC DataFrame with a DatetimeIndex.

    Input must contain Open, High, Low, and Close columns.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("data index must be a DatetimeIndex")

    rename_map: Dict[Any, str] = {}
    for col in data.columns:
        key = str(col).strip().lower()
        if key in {"open", "high", "low", "close"}:
            rename_map[col] = key.capitalize()

    out = data.rename(columns=rename_map).copy()

    required = {"Open", "High", "Low", "Close"}
    if not required.issubset(out.columns):
        missing = ", ".join(sorted(required - set(out.columns)))
        raise ValueError(f"data must contain Open, High, Low, and Close columns; missing: {missing}")

    out = out[["Open", "High", "Low", "Close"]].astype(float)

    out = out.sort_index()
    out = out.loc[~out.index.duplicated(keep="last")]
    return out


def load_data(
    symbol: str = "spx",
    interval: str = "1d",
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """Load OHLC data from a local CSV file.

    Examples
    --------
    >>> df = load_data("spx", interval="1d")
    """
    file_name = f"{str(symbol).strip()}_{str(interval).strip()}.csv"
    project_root = Path(__file__).resolve().parent.parent
    csv_path = project_root / "data" / file_name
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file for {symbol!r} was not found: {csv_path}")

    try:
        df = pd.read_csv(csv_path)
    except pd.errors.ParserError as exc:
        raise ValueError(f"Failed to parse CSV file: {csv_path}") from exc

    if df.empty or "Date" not in df.columns:
        raise ValueError(f"CSV file must contain Date and OHLC columns: {csv_path}")

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")
    if start is not None:
        df = df.loc[pd.to_datetime(start) :]
    if end is not None:
        df = df.loc[: pd.to_datetime(end)]
    ohlc = _ensure_ohlc_frame(df)
    if ohlc.empty:
        raise ValueError(f"No OHLC rows found in CSV file after filtering: {csv_path}")
    return ohlc


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def _segment_slope(values: np.ndarray) -> float:
    """OLS slope of y on time index for one online segment."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=float)
    x_mean = (n - 1) / 2.0
    y_mean = float(np.nanmean(values))
    denom = float(np.sum((x - x_mean) ** 2))
    if denom <= 0:
        return 0.0
    return float(np.sum((x - x_mean) * (values - y_mean)) / denom)


def _sign_with_tol(value: float, tol: float = 1e-12, previous: int = 0) -> int:
    """Return +/-1; if the value is too small, keep the previous nonzero sign."""
    if not np.isfinite(value):
        return previous
    if value > tol:
        return 1
    if value < -tol:
        return -1
    return previous


def _argmax(values: np.ndarray, start: int, end_inclusive: int) -> Optional[int]:
    start = max(0, int(start))
    end = min(len(values), int(end_inclusive) + 1)
    if start >= end:
        return None
    return start + int(np.argmax(values[start:end]))


def _argmin(values: np.ndarray, start: int, end_inclusive: int) -> Optional[int]:
    start = max(0, int(start))
    end = min(len(values), int(end_inclusive) + 1)
    if start >= end:
        return None
    return start + int(np.argmin(values[start:end]))


def _display_price(values: pd.Series | np.ndarray, price_is_log: bool) -> pd.Series | np.ndarray:
    return np.exp(values) if price_is_log else values


def _confirmed_extrema_frame(state_df: pd.DataFrame, price_is_log: bool = False) -> pd.DataFrame:
    required = {"confirmed_extremum_kind", "confirmed_extremum_idx", "confirmed_extremum_value"}
    if not required.issubset(state_df.columns):
        return pd.DataFrame(columns=["timestamp", "kind", "idx", "value"])

    rows: List[Dict[str, Any]] = []
    seen: set[Tuple[int, str]] = set()
    for _, row in state_df[state_df["new_extremum_confirmed"].astype(bool)].iterrows():
        kind = row.get("confirmed_extremum_kind")
        idx = row.get("confirmed_extremum_idx")
        value = row.get("confirmed_extremum_value")
        if pd.isna(kind) or pd.isna(idx) or pd.isna(value):
            continue
        idx_int = int(idx)
        key = (idx_int, str(kind))
        if key in seen or idx_int < 0 or idx_int >= len(state_df):
            continue
        seen.add(key)
        rows.append(
            {
                "timestamp": state_df.index[idx_int],
                "kind": str(kind),
                "idx": idx_int,
                "value": float(np.exp(value) if price_is_log else value),
            }
        )

    if not rows:
        return pd.DataFrame(columns=["timestamp", "kind", "idx", "value"])
    return pd.DataFrame(rows).sort_values("idx").reset_index(drop=True)


def _plot_candles(ax, df: pd.DataFrame, price_is_log: bool = False, label: str = "OHLC") -> pd.Series:
    """Draw a lightweight candlestick chart and return displayed Close prices."""
    from matplotlib.dates import date2num
    from matplotlib.patches import Rectangle

    ohlc = df[["Open", "High", "Low", "Close"]].astype(float)
    if price_is_log:
        ohlc = np.exp(ohlc)

    x = date2num(df.index.to_pydatetime())
    if len(x) > 1:
        width = float(np.median(np.diff(x))) * 0.65
    else:
        width = 0.65

    up_color = "#2b9348"
    down_color = "#d00000"
    for xi, open_, high, low, close in zip(x, ohlc["Open"], ohlc["High"], ohlc["Low"], ohlc["Close"]):
        color = up_color if close >= open_ else down_color
        ax.vlines(xi, low, high, color=color, linewidth=0.8, alpha=0.9)
        lower = min(open_, close)
        height = abs(close - open_)
        if height <= 0:
            ax.hlines(close, xi - width / 2, xi + width / 2, color=color, linewidth=1.0)
        else:
            ax.add_patch(
                Rectangle(
                    (xi - width / 2, lower),
                    width,
                    height,
                    facecolor=color,
                    edgecolor=color,
                    linewidth=0.6,
                    alpha=0.75,
                )
            )

    ax.plot([], [], color="#1f2937", linewidth=1.0, label=label)
    return ohlc["Close"]


def plot_price_state_background(
    state_df: pd.DataFrame,
    price_is_log: bool = False,
    start: Optional[pd.Timestamp | str] = None,
    end: Optional[pd.Timestamp | str] = None,
    show: bool = True,
):
    """Plot price plus Direction, Status, CurrentTrend, and Phase state bands."""
    import matplotlib.pyplot as plt

    full_df = state_df.sort_index()
    df = full_df
    if start is not None:
        df = df.loc[pd.to_datetime(start) :]
    if end is not None:
        df = df.loc[: pd.to_datetime(end)]
    if df.empty:
        raise ValueError("No rows available for the requested plot window.")

    trend_colors = {
        "uptrend": "#b7e4c7",
        "downtrend": "#ffc9c9",
        "no_trend": "#e9ecef",
        "uptrend_question": "#fff3bf",
        "downtrend_question": "#ffd8a8",
        "trend_question": "#fff3bf",
    }
    direction_colors = {-1: "#ffc9c9", 0: "#e9ecef", 1: "#b7e4c7"}
    status_colors = {-1: "#91a7ff", 0: "#e9ecef", 1: "#ffd43b"}
    phase_colors = {
        "up_movement": "#2b9348",
        "up_correction": "#f08c00",
        "down_movement": "#d00000",
        "down_correction": "#1971c2",
        "up_phase_unknown": "#adb5bd",
        "down_phase_unknown": "#adb5bd",
        "no_phase": "#e9ecef",
    }

    fig, axes = plt.subplots(
        5,
        1,
        figsize=(15, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [5, 0.45, 0.45, 0.55, 0.55]},
    )
    ax, direction_ax, status_ax, trend_ax, phase_ax = axes

    trend_group = df["current_trend"].ne(df["current_trend"].shift()).cumsum()
    for _, segment in df.groupby(trend_group):
        trend = segment["current_trend"].iloc[0]
        ax.axvspan(
            segment.index[0],
            segment.index[-1],
            color=trend_colors.get(trend, "#f1f3f5"),
            alpha=0.35,
            linewidth=0,
        )

    price = _plot_candles(ax, df, price_is_log=price_is_log, label="OHLC")

    if "cpd_event" in df.columns:
        cpd = df[df["cpd_event"].astype(bool)]
        ax.scatter(cpd.index, price.loc[cpd.index], marker="x", s=42, color="#7b2cbf", label="CPD")

    status_switch = df["status"].ne(df["status"].shift()).fillna(False)
    if len(status_switch) > 0:
        status_switch.iloc[0] = False
    switch_df = df[status_switch]
    ax.scatter(
        switch_df.index,
        price.loc[switch_df.index],
        marker="o",
        s=26,
        facecolors="none",
        edgecolors="#f08c00",
        label="Status switch",
    )

    extrema = _confirmed_extrema_frame(full_df, price_is_log=price_is_log)
    extrema = extrema[
        (extrema["timestamp"] >= df.index[0]) & (extrema["timestamp"] <= df.index[-1])
    ] if not extrema.empty else extrema
    if not extrema.empty:
        highs = extrema[extrema["kind"] == "high"]
        lows = extrema[extrema["kind"] == "low"]
        ax.scatter(highs["timestamp"], highs["value"], marker="v", s=55, color="#d00000", label="Confirmed high")
        ax.scatter(lows["timestamp"], lows["value"], marker="^", s=55, color="#2b9348", label="Confirmed low")

    def _draw_band(band_ax, column: str, colors: Dict[Any, str], label: str) -> None:
        if column not in df.columns:
            band_ax.set_visible(False)
            return
        groups = df[column].ne(df[column].shift()).cumsum()
        for _, segment in df.groupby(groups):
            value = segment[column].iloc[0]
            color_key = int(value) if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(value) else value
            band_ax.axvspan(
                segment.index[0],
                segment.index[-1],
                color=colors.get(color_key, "#dee2e6"),
                alpha=0.85,
                linewidth=0,
            )
            if len(segment) >= max(2, len(df) // 20):
                midpoint = segment.index[len(segment) // 2]
                band_ax.text(midpoint, 0.5, str(value), ha="center", va="center", fontsize=8)
        band_ax.set_ylim(0, 1)
        band_ax.set_yticks([])
        band_ax.set_ylabel(label)
        band_ax.grid(False)

    _draw_band(direction_ax, "direction", direction_colors, "Direction")
    _draw_band(status_ax, "status", status_colors, "Status")
    _draw_band(trend_ax, "current_trend", trend_colors, "Trend")
    _draw_band(phase_ax, "current_phase", phase_colors, "Phase")

    ax.set_title("Price with Direction, Status, CurrentTrend, Phase, and Events")
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    if show:
        plt.show()
    return fig, axes


def plot_confirmed_high_low(
    state_df: pd.DataFrame,
    price_is_log: bool = False,
    start: Optional[pd.Timestamp | str] = None,
    end: Optional[pd.Timestamp | str] = None,
    show: bool = True,
):
    """Plot price with confirmed high/low points and their zigzag connection."""
    import matplotlib.pyplot as plt

    full_df = state_df.sort_index()
    df = full_df
    if start is not None:
        df = df.loc[pd.to_datetime(start) :]
    if end is not None:
        df = df.loc[: pd.to_datetime(end)]
    if df.empty:
        raise ValueError("No rows available for the requested plot window.")

    extrema = _confirmed_extrema_frame(full_df, price_is_log=price_is_log)
    extrema = extrema[
        (extrema["timestamp"] >= df.index[0]) & (extrema["timestamp"] <= df.index[-1])
    ] if not extrema.empty else extrema

    fig, ax = plt.subplots(figsize=(15, 7))
    price = _plot_candles(ax, df, price_is_log=price_is_log, label="OHLC")

    if not extrema.empty:
        highs = extrema[extrema["kind"] == "high"]
        lows = extrema[extrema["kind"] == "low"]
        ax.plot(extrema["timestamp"], extrema["value"], color="#495057", linewidth=1.0, label="Confirmed zigzag")
        ax.scatter(highs["timestamp"], highs["value"], marker="v", s=70, color="#d00000", label="Confirmed high")
        ax.scatter(lows["timestamp"], lows["value"], marker="^", s=70, color="#2b9348", label="Confirmed low")

    ax.set_title("Price with Confirmed High/Low MinMax Process")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax


def plot_position_series(
    state_df: pd.DataFrame,
    start: Optional[pd.Timestamp | str] = None,
    end: Optional[pd.Timestamp | str] = None,
    show: bool = True,
):
    """Plot Position with structural threshold lines and trend markers."""
    import matplotlib.pyplot as plt

    df = state_df.sort_index()
    if start is not None:
        df = df.loc[pd.to_datetime(start) :]
    if end is not None:
        df = df.loc[: pd.to_datetime(end)]
    if df.empty:
        raise ValueError("No rows available for the requested plot window.")
    if "position" not in df.columns:
        raise ValueError("state_df must include a position column.")

    trend_colors = {
        "uptrend": "#2b9348",
        "downtrend": "#d00000",
        "no_trend": "#868e96",
        "uptrend_question": "#f08c00",
        "downtrend_question": "#e67700",
        "trend_question": "#f08c00",
    }

    fig, (ax, trend_ax) = plt.subplots(
        2,
        1,
        figsize=(15, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [4, 0.7]},
    )

    ax.plot(df.index, df["position"], color="#1f2937", linewidth=1.1, label="Position")
    for level in [0.0, 0.2, 0.5, 0.8, 1.0]:
        ax.axhline(level, color="#868e96", linewidth=0.8, linestyle="--", alpha=0.65)
        ax.text(df.index[0], level, f" {level:g}", va="bottom", ha="left", fontsize=8, color="#495057")

    ax.set_title("Position in Confirmed High/Low Range")
    ax.set_ylabel("Position")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    trend_group = df["current_trend"].ne(df["current_trend"].shift()).cumsum()
    for _, segment in df.groupby(trend_group):
        trend = segment["current_trend"].iloc[0]
        trend_ax.axvspan(
            segment.index[0],
            segment.index[-1],
            color=trend_colors.get(trend, "#dee2e6"),
            alpha=0.75,
            linewidth=0,
        )
        midpoint = segment.index[len(segment) // 2]
        trend_ax.text(midpoint, 0.5, trend, ha="center", va="center", fontsize=8)

    trend_ax.set_ylim(0, 1)
    trend_ax.set_yticks([])
    trend_ax.set_ylabel("Trend")
    trend_ax.grid(False)
    fig.tight_layout()
    if show:
        plt.show()
    return fig, (ax, trend_ax)


# ---------------------------------------------------------------------------
# Structural state
# ---------------------------------------------------------------------------


@dataclass
class Extremum:
    """One confirmed relevant extremum in the online MinMax process."""

    kind: str  # "high" or "low"
    idx: int
    timestamp: pd.Timestamp
    value: float


@dataclass
class _TrendState:
    name: str = "no_trend"
    code: float = 0.0
    candidate: str = "no_candidate"
    point2: float = np.nan
    point3: float = np.nan
    point2_idx: Optional[int] = None
    point3_idx: Optional[int] = None
    candidate_point2: float = np.nan
    candidate_point3: float = np.nan
    candidate_point2_idx: Optional[int] = None
    candidate_point3_idx: Optional[int] = None


@dataclass
class _StructureState:
    last_max_idx: Optional[int] = None
    last_min_idx: Optional[int] = None
    temp_max_idx: Optional[int] = None
    temp_min_idx: Optional[int] = None

    prev_direction: int = 0
    prev_status: int = 0
    excep: int = 1

    extrema: List[Extremum] = field(default_factory=list)
    trend: _TrendState = field(default_factory=_TrendState)
    invalid_up_point3_idx: int = -1
    invalid_down_point3_idx: int = -1


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class MarketStateVector:
    """Build an online market state vector from OHLC data and CPD events."""

    def __init__(
        self,
        detector_method: str = "cusum",
        detector_q: float = 1.0,
        detector_kwargs: Optional[Dict[str, Any]] = None,
        direction_tol: float = 1e-12,
        cpd_confirm_lag: int = 0,
    ) -> None:
        self.detector_method = detector_method
        self.detector_q = detector_q
        self.detector_kwargs = detector_kwargs or {}
        self.direction_tol = direction_tol
        self.cpd_confirm_lag = max(0, int(cpd_confirm_lag))

    # -------------------------- CPD / Direction --------------------------

    def _new_detector(self) -> Brk:
        return Brk(method=self.detector_method, q=self.detector_q, **self.detector_kwargs)

    def _update_cpd_events(
        self,
        detector: Brk,
        close_prefix: pd.Series,
        idx_to_pos: Dict[pd.Timestamp, int],
        i: int,
        confirmed_set: set[int],
        confirmed_positions: List[int],
    ) -> List[int]:
        """Run the existing detector on the current prefix and return new CPD positions."""
        cps = detector.detect(close_prefix)
        new_positions: List[int] = []
        for cp_ts in cps or []:
            ts = pd.Timestamp(cp_ts)
            if ts not in idx_to_pos:
                continue
            pos = idx_to_pos[ts]
            if pos > i or pos in confirmed_set:
                continue
            if i - pos < self.cpd_confirm_lag:
                continue
            confirmed_set.add(pos)
            confirmed_positions.append(pos)
            new_positions.append(pos)
        confirmed_positions.sort()
        return new_positions

    # -------------------------- Exception / Status -----------------------

    def _initialise_minmax_if_needed(
        self,
        i: int,
        direction: int,
        highs: np.ndarray,
        lows: np.ndarray,
        state: _StructureState,
    ) -> None:
        """Initialise last fixed extremum after Direction first becomes nonzero."""
        if direction > 0 and state.last_min_idx is None:
            state.last_min_idx = _argmin(lows, 0, i)
            start = (state.last_min_idx or 0) + 1
            state.temp_max_idx = _argmax(highs, start, i) or i
        elif direction < 0 and state.last_max_idx is None:
            state.last_max_idx = _argmax(highs, 0, i)
            start = (state.last_max_idx or 0) + 1
            state.temp_min_idx = _argmin(lows, start, i) or i

    def _update_excep_and_status(
        self,
        i: int,
        direction: int,
        highs: np.ndarray,
        lows: np.ndarray,
        state: _StructureState,
    ) -> Tuple[int, int]:
        """Update persistent Excep and compute Status.

        This follows the article's idea more closely than a one-bar condition:
        once an exception starts, it remains active until either Direction
        reverses or price recovers beyond the relevant opposite fixed extremum.
        """
        if direction == 0:
            return state.excep, 0

        prev_direction = state.prev_direction
        lastmax = state.last_max_idx
        lastmin = state.last_min_idx

        if state.excep == -1:
            direction_reversed = prev_direction != 0 and prev_direction * direction == -1
            recovered_up = prev_direction == 1 and lastmax is not None and highs[i] >= highs[lastmax]
            recovered_down = prev_direction == -1 and lastmin is not None and lows[i] <= lows[lastmin]
            if direction_reversed or recovered_up or recovered_down:
                state.excep = 1
        else:
            # The article checks exceptional situations only when Direction has
            # not already changed. This avoids double-counting normal reversals.
            same_direction = prev_direction == 0 or prev_direction == direction
            if same_direction:
                breaks_last_low = direction > 0 and lastmin is not None and lows[i] <= lows[lastmin]
                breaks_last_high = direction < 0 and lastmax is not None and highs[i] >= highs[lastmax]
                if breaks_last_low or breaks_last_high:
                    state.excep = -1

        status = int(direction * state.excep)
        return state.excep, status

    # -------------------------- MinMax process ---------------------------

    def _append_extremum(
        self,
        kind: str,
        idx: int,
        value: float,
        timestamp: pd.Timestamp,
        state: _StructureState,
    ) -> None:
        """Append a confirmed extremum while keeping a clean alternating sequence."""
        ext = Extremum(kind=kind, idx=int(idx), value=float(value), timestamp=pd.Timestamp(timestamp))
        if not state.extrema:
            state.extrema.append(ext)
            return

        last = state.extrema[-1]
        if last.kind != kind:
            # Avoid exact duplicate bars of the same kind after noisy switches.
            if last.idx != ext.idx:
                state.extrema.append(ext)
            return

        # If two same-type extrema arrive consecutively, keep the more extreme one.
        if kind == "high" and ext.value >= last.value:
            state.extrema[-1] = ext
        elif kind == "low" and ext.value <= last.value:
            state.extrema[-1] = ext

    def _update_minmax(
        self,
        i: int,
        ts: pd.Timestamp,
        status: int,
        highs: np.ndarray,
        lows: np.ndarray,
        state: _StructureState,
    ) -> bool:
        """Update temporary extrema and fix extrema on Status switches.

        Returns True when a new relevant extremum is confirmed.
        """
        confirmed = False
        prev_status = state.prev_status

        # First nonzero status: start searching in the current direction.
        if prev_status == 0:
            if status == 1:
                if state.temp_max_idx is None:
                    start = (state.last_min_idx or -1) + 1
                    state.temp_max_idx = _argmax(highs, start, i) or i
                if highs[i] >= highs[state.temp_max_idx]:
                    state.temp_max_idx = i
            elif status == -1:
                if state.temp_min_idx is None:
                    start = (state.last_max_idx or -1) + 1
                    state.temp_min_idx = _argmin(lows, start, i) or i
                if lows[i] <= lows[state.temp_min_idx]:
                    state.temp_min_idx = i
            state.prev_status = status
            return False

        # While the previous status was +1, keep moving the temporary maximum.
        if prev_status == 1:
            if state.temp_max_idx is None or highs[i] >= highs[state.temp_max_idx]:
                state.temp_max_idx = i
            if status == -1:
                # Status changed: fix the temporary maximum forever.
                state.last_max_idx = state.temp_max_idx
                self._append_extremum("high", state.last_max_idx, highs[state.last_max_idx], ts, state)
                confirmed = True
                start = state.last_max_idx + 1
                state.temp_min_idx = _argmin(lows, start, i) or i

        # While the previous status was -1, keep moving the temporary minimum.
        if prev_status == -1:
            if state.temp_min_idx is None or lows[i] <= lows[state.temp_min_idx]:
                state.temp_min_idx = i
            if status == 1:
                # Status changed: fix the temporary minimum forever.
                state.last_min_idx = state.temp_min_idx
                self._append_extremum("low", state.last_min_idx, lows[state.last_min_idx], ts, state)
                confirmed = True
                start = state.last_min_idx + 1
                state.temp_max_idx = _argmax(highs, start, i) or i

        # After possible switch, update the new temporary extremum for this bar.
        if status == 1:
            if state.temp_max_idx is None or highs[i] >= highs[state.temp_max_idx]:
                state.temp_max_idx = i
        elif status == -1:
            if state.temp_min_idx is None or lows[i] <= lows[state.temp_min_idx]:
                state.temp_min_idx = i

        state.prev_status = status
        return confirmed

    # -------------------------- Trend structure --------------------------

    @staticmethod
    def _recent_extrema(extrema: List[Extremum], kind: str, n: int = 2) -> List[Extremum]:
        items = [e for e in extrema if e.kind == kind]
        return items[-n:]

    def _update_trend_structure(
        self,
        i: int,
        close_i: float,
        high_i: float,
        low_i: float,
        state: _StructureState,
    ) -> _TrendState:
        """Derive CurrentTrend from 1-2-3 candidates plus point2 breakout.

        A confirmed trend needs two steps:
        1. a structural candidate from confirmed extrema;
        2. a later breakout through point2 while point3 remains unbroken.

        The confirmed trend persists until point3 is broken. A pullback inside
        the trend is represented by current_phase, not by downgrading the trend.
        """
        prev = state.trend

        up_candidate: Optional[Tuple[Extremum, Extremum]] = None
        down_candidate: Optional[Tuple[Extremum, Extremum]] = None
        for a, b, c in zip(state.extrema, state.extrema[1:], state.extrema[2:]):
            if (
                a.kind == "low"
                and b.kind == "high"
                and c.kind == "low"
                and c.value >= a.value
                and c.idx > state.invalid_up_point3_idx
            ):
                # Automatic 1-2-3 up candidate: low1 -> high1(point2) -> higher/equal low2(point3).
                up_candidate = (b, c)
            elif (
                a.kind == "high"
                and b.kind == "low"
                and c.kind == "high"
                and c.value <= a.value
                and c.idx > state.invalid_down_point3_idx
            ):
                # Automatic 1-2-3 down candidate: high1 -> low1(point2) -> lower/equal high2(point3).
                down_candidate = (b, c)

        confirmed_trend_invalidated = False
        if prev.name == "uptrend" and np.isfinite(prev.point3) and low_i <= prev.point3:
            if prev.point3_idx is not None:
                state.invalid_up_point3_idx = max(state.invalid_up_point3_idx, prev.point3_idx)
            prev = _TrendState(name="no_trend", code=0.0)
            confirmed_trend_invalidated = True
        elif prev.name == "downtrend" and np.isfinite(prev.point3) and high_i >= prev.point3:
            if prev.point3_idx is not None:
                state.invalid_down_point3_idx = max(state.invalid_down_point3_idx, prev.point3_idx)
            prev = _TrendState(name="no_trend", code=0.0)
            confirmed_trend_invalidated = True

        candidates: List[_TrendState] = []
        if up_candidate is not None:
            point2, point3 = up_candidate
            stop_alive = i <= point3.idx or low_i > point3.value
            if not stop_alive:
                state.invalid_up_point3_idx = max(state.invalid_up_point3_idx, point3.idx)
            elif high_i >= point2.value and i > point3.idx and not confirmed_trend_invalidated:
                candidates.append(
                    _TrendState(
                        name="uptrend",
                        code=1.0,
                        candidate="uptrend_candidate",
                        point2=point2.value,
                        point3=point3.value,
                        point2_idx=point2.idx,
                        point3_idx=point3.idx,
                        candidate_point2=point2.value,
                        candidate_point3=point3.value,
                        candidate_point2_idx=point2.idx,
                        candidate_point3_idx=point3.idx,
                    )
                )
            else:
                candidates.append(
                    _TrendState(
                        name="uptrend_question",
                        code=0.5,
                        candidate="uptrend_candidate",
                        point2=point2.value,
                        point3=point3.value,
                        point2_idx=point2.idx,
                        point3_idx=point3.idx,
                        candidate_point2=point2.value,
                        candidate_point3=point3.value,
                        candidate_point2_idx=point2.idx,
                        candidate_point3_idx=point3.idx,
                    )
                )

        if down_candidate is not None:
            point2, point3 = down_candidate
            stop_alive = i <= point3.idx or high_i < point3.value
            if not stop_alive:
                state.invalid_down_point3_idx = max(state.invalid_down_point3_idx, point3.idx)
            elif low_i <= point2.value and i > point3.idx and not confirmed_trend_invalidated:
                candidates.append(
                    _TrendState(
                        name="downtrend",
                        code=-1.0,
                        candidate="downtrend_candidate",
                        point2=point2.value,
                        point3=point3.value,
                        point2_idx=point2.idx,
                        point3_idx=point3.idx,
                        candidate_point2=point2.value,
                        candidate_point3=point3.value,
                        candidate_point2_idx=point2.idx,
                        candidate_point3_idx=point3.idx,
                    )
                )
            else:
                candidates.append(
                    _TrendState(
                        name="downtrend_question",
                        code=-0.5,
                        candidate="downtrend_candidate",
                        point2=point2.value,
                        point3=point3.value,
                        point2_idx=point2.idx,
                        point3_idx=point3.idx,
                        candidate_point2=point2.value,
                        candidate_point3=point3.value,
                        candidate_point2_idx=point2.idx,
                        candidate_point3_idx=point3.idx,
                    )
                )

        latest_candidate = max(
            candidates,
            key=lambda t: -1 if t.candidate_point3_idx is None else t.candidate_point3_idx,
            default=None,
        )
        triggered = [t for t in candidates if t.name in {"uptrend", "downtrend"}]
        latest_triggered = max(
            triggered,
            key=lambda t: -1 if t.point3_idx is None else t.point3_idx,
            default=None,
        )

        if (
            prev.name in {"uptrend", "downtrend"}
            and latest_triggered is not None
            and latest_triggered.name == prev.name
            and (
                prev.point3_idx is None
                or (
                    latest_triggered.point3_idx is not None
                    and latest_triggered.point3_idx > prev.point3_idx
                )
            )
        ):
            trend = latest_triggered
        elif prev.name in {"uptrend", "downtrend"}:
            trend = _TrendState(
                name=prev.name,
                code=prev.code,
                candidate=latest_candidate.candidate if latest_candidate is not None else "no_candidate",
                point2=prev.point2,
                point3=prev.point3,
                point2_idx=prev.point2_idx,
                point3_idx=prev.point3_idx,
                candidate_point2=latest_candidate.candidate_point2 if latest_candidate is not None else np.nan,
                candidate_point3=latest_candidate.candidate_point3 if latest_candidate is not None else np.nan,
                candidate_point2_idx=latest_candidate.candidate_point2_idx if latest_candidate is not None else None,
                candidate_point3_idx=latest_candidate.candidate_point3_idx if latest_candidate is not None else None,
            )
        elif latest_triggered is not None:
            trend = latest_triggered
        elif latest_candidate is not None:
            trend = latest_candidate
        else:
            trend = _TrendState(name="no_trend", code=0.0)

        state.trend = trend
        return trend

    @staticmethod
    def _derive_current_phase(trend: _TrendState, status: int, direction: int) -> str:
        """Separate structural trend from the current movement/correction leg."""
        if trend.name in {"uptrend", "uptrend_question"}:
            leg_sign = status if status != 0 else direction
            if leg_sign > 0:
                return "up_movement"
            if leg_sign < 0:
                return "up_correction"
            return "up_phase_unknown"
        if trend.name in {"downtrend", "downtrend_question"}:
            leg_sign = status if status != 0 else direction
            if leg_sign < 0:
                return "down_movement"
            if leg_sign > 0:
                return "down_correction"
            return "down_phase_unknown"
        return "no_phase"

    # -------------------------- Build state vector -----------------------

    def build_market_state_vector(
        self,
        data: Optional[pd.DataFrame] = None,
        symbol: Optional[str] = None,
        interval: str = "1d",
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """Build the online market state vector.

        Parameters
        ----------
        data:
            OHLC DataFrame with a DatetimeIndex and Open/High/Low/Close columns.
            If omitted, ``symbol`` is resolved to a local CSV file.
        symbol:
            CSV path or symbol resolved from ``data/<symbol>_<interval>.csv``.
        """
        if data is None:
            if symbol is None:
                raise ValueError("Either data or symbol must be provided")
            data = load_data(
                symbol=symbol,
                interval=interval,
                start=start,
                end=end,
            )
        ohlc = _ensure_ohlc_frame(data)
        if ohlc.empty:
            raise ValueError("Cannot build market state vector from empty OHLC data")

        timestamps = list(ohlc.index)
        open_arr = ohlc["Open"].to_numpy(float)
        high_arr = ohlc["High"].to_numpy(float)
        low_arr = ohlc["Low"].to_numpy(float)
        close_arr = ohlc["Close"].to_numpy(float)
        close_series = ohlc["Close"].astype(float)

        detector = self._new_detector()
        idx_to_pos = {pd.Timestamp(ts): i for i, ts in enumerate(timestamps)}
        confirmed_cp_set: set[int] = set()
        confirmed_cp_positions: List[int] = []
        last_confirm_bar: Optional[int] = None

        structure = _StructureState()
        rows: List[Dict[str, Any]] = []

        for i, ts in enumerate(timestamps):
            prefix = close_series.iloc[: i + 1]
            new_cps = self._update_cpd_events(
                detector=detector,
                close_prefix=prefix,
                idx_to_pos=idx_to_pos,
                i=i,
                confirmed_set=confirmed_cp_set,
                confirmed_positions=confirmed_cp_positions,
            )
            cpd_event = bool(new_cps)
            if cpd_event:
                last_confirm_bar = i

            regime_start = confirmed_cp_positions[-1] if confirmed_cp_positions else 0
            regime_segment = close_arr[regime_start : i + 1]
            regime_slope = _segment_slope(regime_segment)
            direction = _sign_with_tol(regime_slope, self.direction_tol, previous=structure.prev_direction)

            self._initialise_minmax_if_needed(i, direction, high_arr, low_arr, structure)
            excep, status = self._update_excep_and_status(i, direction, high_arr, low_arr, structure)
            status_switch = i > 0 and status != structure.prev_status
            new_extremum_confirmed = self._update_minmax(i, pd.Timestamp(ts), status, high_arr, low_arr, structure)
            trend = self._update_trend_structure(i, close_arr[i], high_arr[i], low_arr[i], structure)
            current_phase = self._derive_current_phase(trend, status, direction)
            confirmed_extremum = structure.extrema[-1] if new_extremum_confirmed and structure.extrema else None

            last_max = high_arr[structure.last_max_idx] if structure.last_max_idx is not None else np.nan
            last_min = low_arr[structure.last_min_idx] if structure.last_min_idx is not None else np.nan
            temp_max = high_arr[structure.temp_max_idx] if structure.temp_max_idx is not None else np.nan
            temp_min = low_arr[structure.temp_min_idx] if structure.temp_min_idx is not None else np.nan

            denom = last_max - last_min if np.isfinite(last_max) and np.isfinite(last_min) else np.nan
            position = (close_arr[i] - last_min) / denom if np.isfinite(denom) and abs(denom) > 0 else np.nan

            delta_position = np.nan
            if rows and np.isfinite(position) and np.isfinite(rows[-1].get("position", np.nan)):
                delta_position = float(position - rows[-1]["position"])

            distance_to_high = close_arr[i] - last_max if np.isfinite(last_max) else np.nan
            distance_to_low = close_arr[i] - last_min if np.isfinite(last_min) else np.nan
            distance_to_point2 = close_arr[i] - trend.point2 if np.isfinite(trend.point2) else np.nan
            distance_to_point3 = close_arr[i] - trend.point3 if np.isfinite(trend.point3) else np.nan

            if regime_start < i:
                returns = np.diff(close_arr[regime_start : i + 1])
                regime_volatility = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0
            else:
                regime_volatility = 0.0

            time_since_last_cpd = i - last_confirm_bar if last_confirm_bar is not None else i + 1
            trend_age = 0
            if rows and rows[-1]["current_trend"] == trend.name:
                trend_age = int(rows[-1]["trend_age"] + 1)

            structure.prev_direction = direction

            rows.append(
                {
                    "timestamp": ts,
                    "Open": float(open_arr[i]),
                    "High": float(high_arr[i]),
                    "Low": float(low_arr[i]),
                    "Close": float(close_arr[i]),
                    "cpd_event": cpd_event,
                    "direction": int(direction),
                    "excep": int(excep),
                    "status": int(status),
                    "status_switch": bool(status_switch),
                    "new_extremum_confirmed": bool(new_extremum_confirmed),
                    "confirmed_extremum_kind": confirmed_extremum.kind if confirmed_extremum is not None else None,
                    "confirmed_extremum_idx": confirmed_extremum.idx if confirmed_extremum is not None else None,
                    "confirmed_extremum_value": (
                        float(confirmed_extremum.value) if confirmed_extremum is not None else np.nan
                    ),
                    "current_trend": trend.name,
                    "current_trend_code": float(trend.code),
                    "current_phase": current_phase,
                    "trend_candidate": trend.candidate,
                    "trend_age": int(trend_age),
                    "position": float(position) if np.isfinite(position) else np.nan,
                    "delta_position": float(delta_position) if np.isfinite(delta_position) else np.nan,
                    "distance_to_high": float(distance_to_high) if np.isfinite(distance_to_high) else np.nan,
                    "distance_to_low": float(distance_to_low) if np.isfinite(distance_to_low) else np.nan,
                    "distance_to_point2": float(distance_to_point2) if np.isfinite(distance_to_point2) else np.nan,
                    "distance_to_point3": float(distance_to_point3) if np.isfinite(distance_to_point3) else np.nan,
                    "time_since_last_cpd": int(time_since_last_cpd),
                    "regime_slope": float(regime_slope),
                    "regime_volatility": float(regime_volatility),
                    "point2": float(trend.point2) if np.isfinite(trend.point2) else np.nan,
                    "point3": float(trend.point3) if np.isfinite(trend.point3) else np.nan,
                    "point2_idx": trend.point2_idx,
                    "point3_idx": trend.point3_idx,
                    "candidate_point2": (
                        float(trend.candidate_point2) if np.isfinite(trend.candidate_point2) else np.nan
                    ),
                    "candidate_point3": (
                        float(trend.candidate_point3) if np.isfinite(trend.candidate_point3) else np.nan
                    ),
                    "candidate_point2_idx": trend.candidate_point2_idx,
                    "candidate_point3_idx": trend.candidate_point3_idx,
                    "last_max": float(last_max) if np.isfinite(last_max) else np.nan,
                    "last_min": float(last_min) if np.isfinite(last_min) else np.nan,
                    "temp_max": float(temp_max) if np.isfinite(temp_max) else np.nan,
                    "temp_min": float(temp_min) if np.isfinite(temp_min) else np.nan,
                    "num_confirmed_extrema": int(len(structure.extrema)),
                    "regime_start_idx": int(regime_start),
                }
            )

        state_df = pd.DataFrame(rows).set_index("timestamp")
        state_df.index = pd.to_datetime(state_df.index)
        return state_df

    # -------------------------- Evaluation -------------------------------

    def evaluate_state_vector(
        self,
        state_df: pd.DataFrame,
        horizons: Sequence[int] = (5,),
        position_bins: Optional[Sequence[float]] = None,
        price_is_log: bool = False,
    ) -> Dict[str, Any]:
        """Evaluate persistence and future-return separation."""
        if not isinstance(state_df, pd.DataFrame):
            raise TypeError("state_df must be a pandas DataFrame")
        if "Close" not in state_df.columns:
            raise ValueError("state_df must include a Close column")

        df = state_df.sort_index().copy()
        horizons = tuple(int(h) for h in horizons)

        for h in horizons:
            if price_is_log:
                df[f"future_return_{h}"] = df["Close"].shift(-h) - df["Close"]
            else:
                df[f"future_return_{h}"] = df["Close"].shift(-h) / df["Close"] - 1.0

        trend_change = df["current_trend"].ne(df["current_trend"].shift()).fillna(False)
        status_change = df["status"].ne(df["status"].shift()).fillna(False)
        trend_group = trend_change.cumsum()

        duration_table = (
            df.groupby(trend_group)
            .agg(
                current_trend=("current_trend", "first"),
                start=("current_trend", lambda s: s.index[0]),
                end=("current_trend", lambda s: s.index[-1]),
                bars=("current_trend", "size"),
            )
            .reset_index(drop=True)
        )
        duration_summary = duration_table.groupby("current_trend")["bars"].agg(
            ["count", "mean", "median", "min", "max"]
        )

        transition_matrix = pd.crosstab(
            df["current_trend"].shift(),
            df["current_trend"],
            normalize="index",
        ).fillna(0.0)

        if position_bins is None:
            position_bins = [-np.inf, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, np.inf]
        position_bucket = pd.cut(df["position"], bins=position_bins, include_lowest=True)

        position_future_returns: Dict[int, pd.DataFrame] = {}
        trend_future_returns: Dict[int, pd.DataFrame] = {}
        for h in horizons:
            col = f"future_return_{h}"
            position_future_returns[h] = df.groupby(position_bucket, observed=False)[col].agg(
                ["count", "mean", "std", "median", "min", "max"]
            )
            trend_future_returns[h] = df.groupby("current_trend")[col].agg(
                ["count", "mean", "std", "median", "min", "max"]
            )

        return {
            "state_duration_table": duration_table,
            "state_duration_summary": duration_summary,
            "switch_counts": {
                "current_trend_switch_count": int(max(trend_change.sum() - 1, 0)),
                "status_switch_count": int(max(status_change.sum() - 1, 0)),
                "confirmed_extrema_count": int(df.get("new_extremum_confirmed", pd.Series(False, index=df.index)).sum()),
            },
            "transition_matrix": transition_matrix,
            "position_future_returns": position_future_returns,
            "trend_future_returns": trend_future_returns,
        }


# ---------------------------------------------------------------------------
# Functional API
# ---------------------------------------------------------------------------


def build_market_state_vector(
    data: Optional[pd.DataFrame] = None,
    symbol: Optional[str] = None,
    interval: str = "1d",
    start: Optional[str] = None,
    end: Optional[str] = None,
    detector_method: str = "cusum",
    detector_q: float = 1.0,
    detector_kwargs: Optional[Dict[str, Any]] = None,
    direction_tol: float = 1e-12,
    cpd_confirm_lag: int = 0,
) -> pd.DataFrame:
    builder = MarketStateVector(
        detector_method=detector_method,
        detector_q=detector_q,
        detector_kwargs=detector_kwargs,
        direction_tol=direction_tol,
        cpd_confirm_lag=cpd_confirm_lag,
    )
    return builder.build_market_state_vector(
        data=data,
        symbol=symbol,
        interval=interval,
        start=start,
        end=end,
    )


def get_core_state_vector(state_df: pd.DataFrame) -> pd.DataFrame:
    """Return the compact core market state vector from a full state_df."""
    if not isinstance(state_df, pd.DataFrame):
        raise TypeError("state_df must be a pandas DataFrame")

    columns = [
        "current_trend",
        "current_phase",
        "position",
        "regime_slope",
        "regime_volatility",
    ]
    core_df = state_df[columns].copy()

    close = state_df["Close"].replace(0, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        core_df["dist_point3_pct"] = state_df["distance_to_point3"] / close
    core_df["dist_point3_pct"] = core_df["dist_point3_pct"].replace([np.inf, -np.inf], np.nan)

    return core_df[
        [
            "current_trend",
            "current_phase",
            "position",
            "regime_slope",
            "regime_volatility",
            "dist_point3_pct",
        ]
    ]


def evaluate_state_vector(
    state_df: pd.DataFrame,
    horizons: Sequence[int] = (5,),
    position_bins: Optional[Sequence[float]] = None,
    price_is_log: bool = False,
) -> Dict[str, Any]:
    return MarketStateVector().evaluate_state_vector(
        state_df=state_df,
        horizons=horizons,
        position_bins=position_bins,
        price_is_log=price_is_log,
    )


__all__ = [
    "Extremum",
    "MarketStateVector",
    "load_data",
    "build_market_state_vector",
    "get_core_state_vector",
    "evaluate_state_vector",
    "plot_price_state_background",
    "plot_confirmed_high_low",
    "plot_position_series",
]
