from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def validate_series(ds: pd.Series, column_name: Optional[str] = None) -> pd.Series:
    """Validate input data and return a datetime-indexed pandas Series."""
    if isinstance(ds, pd.DataFrame):
        if column_name is None:
            raise ValueError("If ds is a DataFrame, column_name must be provided")
        if column_name not in ds.columns:
            raise ValueError(f"Column '{column_name}' not found in DataFrame")
        ds = ds[column_name]

    if not isinstance(ds, pd.Series):
        raise TypeError("ds must be a pandas Series")
    if not isinstance(ds.index, pd.DatetimeIndex):
        raise ValueError("Series index must be datetime type")

    return ds


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
        confirmed_at = row.get("confirmed_extremum_confirmed_at", pd.NaT)
        rows.append(
            {
                "timestamp": state_df.index[idx_int],
                "confirmed_at": pd.to_datetime(confirmed_at) if not pd.isna(confirmed_at) else pd.NaT,
                "kind": str(kind),
                "idx": idx_int,
                "value": float(np.exp(value) if price_is_log else value),
            }
        )

    if not rows:
        return pd.DataFrame(columns=["timestamp", "confirmed_at", "kind", "idx", "value"])
    return pd.DataFrame(rows).sort_values("idx").reset_index(drop=True)


def _plot_extrema_steps(ax, full_extrema: pd.DataFrame, window_index: pd.DatetimeIndex) -> None:
    """Draw held confirmed high/low levels as MinMax step lines."""
    if full_extrema.empty or len(window_index) == 0:
        return

    start = window_index[0]
    end = window_index[-1]
    styles = {
        "high": {"color": "#d00000", "label": "Confirmed high level"},
        "low": {"color": "#1971c2", "label": "Confirmed low level"},
    }

    for kind, style in styles.items():
        points = full_extrema[full_extrema["kind"] == kind].sort_values("timestamp")
        points = points[points["timestamp"] <= end]
        if points.empty:
            continue

        before = points[points["timestamp"] < start].tail(1)
        inside = points[points["timestamp"] >= start]
        points = pd.concat([before, inside], ignore_index=True)
        if points.empty:
            continue

        x_values = list(pd.to_datetime(points["timestamp"]))
        y_values = list(points["value"].astype(float))
        if x_values[0] < start:
            x_values[0] = start
        if x_values[-1] < end:
            x_values.append(end)
            y_values.append(y_values[-1])

        ax.step(
            x_values,
            y_values,
            where="post",
            linewidth=1.1,
            alpha=0.95,
            color=style["color"],
            label=style["label"],
        )


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

    if "cpd_confirm_event" in df.columns:
        cpd = df[df["cpd_confirm_event"].astype(bool)]
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
    _plot_extrema_steps(ax, extrema, df.index)

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
