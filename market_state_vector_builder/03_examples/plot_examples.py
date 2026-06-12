from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


MANUAL_STATE_COLUMNS = [
    "range_pct",
    "expansion",
    "structure_dir",
    "range_pos",
    "move_age",
    "extrema_freq",
    "speed_imbalance",
    "speed_level",
]

MANUAL_GROUPS = {
    "shape": ["range_pct", "expansion", "structure_dir"],
    "position": ["range_pos"],
    "tempo": ["move_age", "extrema_freq", "speed_imbalance", "speed_level"],
}


def _savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def append_extremum(extrema: list[dict], ext: dict) -> None:
    if not extrema:
        extrema.append(ext)
        return

    last = extrema[-1]
    if last["idx"] == ext["idx"] and last["kind"] == ext["kind"]:
        return

    if last["kind"] != ext["kind"]:
        extrema.append(ext)
        return

    if ext["kind"] == "high" and ext["value"] >= last["value"]:
        extrema[-1] = ext
    elif ext["kind"] == "low" and ext["value"] <= last["value"]:
        extrema[-1] = ext


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot random manual-state K-line examples.")
    parser.add_argument("--final-state-path", default="market_state_vector_builder/outputs/shape_state_analysis_manual/final_state_vector.parquet")
    parser.add_argument("--state-path", default="market_state_vector_builder/outputs/shape_state_analysis_manual/shape_state_variables.parquet")
    parser.add_argument("--symbol-dir", default="data/a_share_1d_akshare/symbols")
    parser.add_argument("--out-dir", default="market_state_vector_builder/outputs/shape_state_analysis_manual/examples")
    parser.add_argument("--n-examples", type=int, default=10)
    parser.add_argument("--lookback", type=int, default=90)
    parser.add_argument("--lookahead", type=int, default=20)
    parser.add_argument("--match-lookback", type=int, default=260)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--state-price-is-log", action="store_true", help="Structural levels in state files are log prices.")
    return parser.parse_args()


def load_ohlcv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    if "date" not in df.columns and "日期" in df.columns:
        df = df.rename(columns={"日期": "date"})
    rename = {
        "开盘": "open",
        "最高": "high",
        "最低": "low",
        "收盘": "close",
        "成交量": "volume",
        "成交额": "amount",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    df["date"] = pd.to_datetime(df["date"])
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values("date").reset_index(drop=True)


def find_level_date(
    ohlc: pd.DataFrame,
    *,
    end_idx: int,
    level: float,
    kind: str,
    search_start: int,
    used: set[int],
    state_price_is_log: bool,
) -> int | None:
    price_col = "high" if kind == "high" else "low"
    window = ohlc.iloc[search_start : end_idx + 1].copy()
    values = pd.to_numeric(window[price_col], errors="coerce").to_numpy(float)
    if not np.isfinite(level) or len(values) == 0:
        return None
    match_level = float(np.exp(level)) if state_price_is_log else float(level)

    scale = max(abs(match_level), 1.0)
    diffs = np.abs(values - match_level) / scale
    order = np.argsort(diffs)
    for pos in order:
        idx = int(window.index[pos])
        if idx not in used and diffs[pos] <= 1e-6:
            return idx

    # Some CPD extrema are based on close-like levels. Fall back to the nearest
    # OHLC match if exact high/low matching is unavailable.
    ohlc_values = window[["open", "high", "low", "close"]].to_numpy(float)
    diffs2 = np.nanmin(np.abs(ohlc_values - match_level) / scale, axis=1)
    order2 = np.argsort(diffs2)
    for pos in order2:
        idx = int(window.index[pos])
        if idx not in used and diffs2[pos] <= 1e-6:
            return idx
    return None


def infer_structure_points(
    row: pd.Series,
    ohlc: pd.DataFrame,
    current_idx: int,
    match_lookback: int,
    state_price_is_log: bool,
) -> list[dict]:
    search_start = max(0, current_idx - match_lookback)
    levels = [
        ("high1", "high", row.get("high1")),
        ("high2", "high", row.get("high2")),
        ("low1", "low", row.get("low1")),
        ("low2", "low", row.get("low2")),
    ]
    used: set[int] = set()
    points = []
    for label, kind, level in levels:
        idx = find_level_date(
            ohlc,
            end_idx=current_idx,
            level=float(level) if pd.notna(level) else np.nan,
            kind=kind,
            search_start=search_start,
            used=used,
            state_price_is_log=state_price_is_log,
        )
        if idx is not None:
            used.add(idx)
        points.append({"label": label, "kind": kind, "idx": idx, "level": level})
    return points


def infer_structure_points_from_state(
    row: pd.Series,
    stock_state: pd.DataFrame,
    ohlc: pd.DataFrame,
    state_price_is_log: bool,
) -> list[dict]:
    row_date = pd.Timestamp(row["date"])
    history = stock_state[pd.to_datetime(stock_state["date"]) <= row_date].copy()
    events = history[history.get("new_extremum_confirmed", False).fillna(False)].copy()
    extrema: list[dict] = []
    for event in events.itertuples(index=False):
        kind = getattr(event, "confirmed_extremum_kind", None)
        idx_value = getattr(event, "confirmed_extremum_idx", np.nan)
        value = getattr(event, "confirmed_extremum_value", np.nan)
        if kind not in {"high", "low"} or pd.isna(idx_value) or pd.isna(value):
            continue
        idx = int(idx_value)
        if idx < 0 or idx >= len(ohlc):
            continue
        append_extremum(extrema, {"kind": str(kind), "idx": idx, "value": float(value)})

    if len(extrema) < 4:
        return []

    window = extrema[-4:]
    highs = sorted([e for e in window if e["kind"] == "high"], key=lambda e: e["idx"])
    lows = sorted([e for e in window if e["kind"] == "low"], key=lambda e: e["idx"])
    if len(highs) != 2 or len(lows) != 2:
        return []

    labeled = [
        ("high1", highs[0]),
        ("high2", highs[1]),
        ("low1", lows[0]),
        ("low2", lows[1]),
    ]
    return [
        {
            "label": label,
            "kind": ext["kind"],
            "idx": ext["idx"],
            "level": ext["value"],
        }
        for label, ext in labeled
    ]


def plot_candles(ax: plt.Axes, df: pd.DataFrame) -> None:
    dates = mdates.date2num(df["date"].to_numpy(dtype="datetime64[ms]"))
    width = 0.65
    for x, (_, row) in zip(dates, df.iterrows()):
        open_, high, low, close = row["open"], row["high"], row["low"], row["close"]
        color = "#d95f59" if close >= open_ else "#4c78a8"
        ax.vlines(x, low, high, color=color, linewidth=0.8, alpha=0.9)
        lower = min(open_, close)
        height = max(abs(close - open_), 1e-8)
        rect = plt.Rectangle((x - width / 2, lower), width, height, facecolor=color, edgecolor=color, alpha=0.75)
        ax.add_patch(rect)
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.grid(True, axis="y", alpha=0.2)


def set_price_axis(ax: plt.Axes, plot_df: pd.DataFrame, points: list[dict], state_price_is_log: bool) -> None:
    values = plot_df[["open", "high", "low", "close"]].to_numpy(float).ravel()
    visible_levels = []
    start_idx = int(plot_df.index.min())
    end_idx = int(plot_df.index.max())
    for point in points:
        if point["idx"] is None or point["idx"] < start_idx or point["idx"] > end_idx:
            continue
        level = float(np.exp(point["level"])) if state_price_is_log else float(point["level"])
        if np.isfinite(level):
            visible_levels.append(level)

    values = np.concatenate([values[np.isfinite(values)], np.array(visible_levels, dtype=float)])
    if len(values) == 0:
        return

    lo = float(np.nanmin(values))
    hi = float(np.nanmax(values))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return
    if abs(hi - lo) < 1e-12:
        pad = max(abs(hi) * 0.02, 0.01)
    else:
        pad = (hi - lo) * 0.06
    ax.set_ylim(lo - pad, hi + pad)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=7, min_n_ticks=4))
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.2f}"))


def plot_manual_state_bars(ax: plt.Axes, row: pd.Series) -> None:
    records: list[tuple[str, str, float, float]] = []
    for group, cols in MANUAL_GROUPS.items():
        for col in cols:
            score_col = f"{col}_score"
            if col in row.index and score_col in row.index and pd.notna(row[col]) and pd.notna(row[score_col]):
                records.append((group, col, float(row[score_col]), float(row[col])))

    if not records:
        ax.axis("off")
        return

    colors = {"shape": "#4c78a8", "position": "#f58518", "tempo": "#54a24b"}
    labels = [f"{name}  raw={raw:.3f}" for _, name, _, raw in records]
    scores = [score for _, _, score, _ in records]
    bar_colors = [colors[group] for group, _, _, _ in records]
    y = np.arange(len(records))

    ax.barh(y, scores, color=bar_colors, alpha=0.88)
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_xlim(-1.05, 1.05)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Quantile score [-1, 1]")
    ax.set_title("Manual State Variables")
    ax.grid(True, axis="x", alpha=0.2)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=colors[group], alpha=0.88, label=group)
        for group in MANUAL_GROUPS
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8, frameon=False)


def plot_example(
    row: pd.Series,
    ohlc: pd.DataFrame,
    stock_state: pd.DataFrame,
    out_path: Path,
    *,
    lookback: int,
    lookahead: int,
    match_lookback: int,
    state_price_is_log: bool,
) -> dict:
    date = pd.Timestamp(row["date"])
    matches = np.flatnonzero(ohlc["date"].to_numpy(dtype="datetime64[ns]") == np.datetime64(date))
    if len(matches) == 0:
        raise ValueError(f"Date {date.date()} not found in OHLC data for {row['stock_id']}")
    current_idx = int(matches[0])
    points = infer_structure_points_from_state(row, stock_state, ohlc, state_price_is_log)
    if not points:
        points = infer_structure_points(row, ohlc, current_idx, match_lookback, state_price_is_log)
    matched_point_indices = [point["idx"] for point in points if point["idx"] is not None]
    if matched_point_indices:
        start = max(0, min(matched_point_indices) - 20)
    else:
        start = max(0, current_idx - lookback)
    end = min(len(ohlc) - 1, current_idx + lookahead)
    plot_df = ohlc.iloc[start : end + 1].copy()

    has_manual_scores = any(f"{col}_score" in row.index and pd.notna(row.get(f"{col}_score")) for col in MANUAL_STATE_COLUMNS)
    if has_manual_scores:
        fig, (ax, bar_ax) = plt.subplots(
            2,
            1,
            figsize=(14, 10),
            gridspec_kw={"height_ratios": [2.2, 1.4]},
        )
    else:
        fig, ax = plt.subplots(figsize=(14, 7))
        bar_ax = None
    plot_candles(ax, plot_df)
    ax.axvline(mdates.date2num(date.to_pydatetime()), color="black", linestyle="--", linewidth=1.2, label="selected date")

    colors = {"high": "#d62728", "low": "#2ca02c"}
    for point in points:
        if point["idx"] is None or point["idx"] < start or point["idx"] > end:
            continue
        pdate = ohlc.loc[point["idx"], "date"]
        x = mdates.date2num(pdate.to_pydatetime())
        y = float(np.exp(point["level"])) if state_price_is_log else float(point["level"])
        marker = "^" if point["kind"] == "high" else "v"
        ax.scatter(x, y, s=95, marker=marker, color=colors[point["kind"]], edgecolor="black", zorder=5)
        offset = 8 if point["kind"] == "high" else -14
        ax.annotate(point["label"], (x, y), xytext=(0, offset), textcoords="offset points", ha="center", fontsize=9)
    set_price_axis(ax, plot_df, points, state_price_is_log)

    info = (
        f"{row['stock_id']}  {date.date()}  close={row['close']:.2f}"
        f"{' (log)' if state_price_is_log else ''}\n"
        f"range_pos={row.get('range_pos', np.nan):.2f}  "
        f"upper_break={row.get('upper_break', np.nan):.3f}  "
        f"lower_break={row.get('lower_break', np.nan):.3f}\n"
        f"starts_high={row.get('starts_high', np.nan):.0f}  "
        f"high1={row.get('high1', np.nan):.2f}, high2={row.get('high2', np.nan):.2f}, "
        f"low1={row.get('low1', np.nan):.2f}, low2={row.get('low2', np.nan):.2f}"
        f"{' (log levels)' if state_price_is_log else ''}"
    )
    ax.text(
        0.01,
        0.98,
        info,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "#999999"},
    )

    missing = [p["label"] for p in points if p["idx"] is None]
    title = "Shape State Example"
    if missing:
        title += f" (unmatched levels: {', '.join(missing)})"
    ax.set_title(title)
    ax.set_ylabel("Price")
    fig.autofmt_xdate()
    if bar_ax is not None:
        plot_manual_state_bars(bar_ax, row)
    _savefig(out_path)

    return {
        "stock_id": row["stock_id"],
        "date": str(date.date()),
        "close": row["close"],
        "output": str(out_path),
        "unmatched_levels": ",".join(missing),
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    final_state = pd.read_parquet(args.final_state_path)
    state = pd.read_parquet(args.state_path)
    keep = ["stock_id", "date", "starts_high", "high1", "high2", "low1", "low2"]
    merged = final_state.merge(
        state[[col for col in keep if col in state.columns]],
        on=["stock_id", "date"],
        how="left",
        suffixes=("", "_raw"),
    )
    if "valid_manual_state" in merged.columns:
        valid_mask = merged["valid_manual_state"].fillna(False)
    else:
        valid_mask = pd.Series(True, index=merged.index)
    candidates = merged[valid_mask].dropna(subset=["high1", "high2", "low1", "low2"])
    sample = candidates.sample(n=min(args.n_examples, len(candidates)), random_state=args.random_state)

    records = []
    for i, (_, row) in enumerate(sample.iterrows(), start=1):
        symbol_path = Path(args.symbol_dir) / f"{row['stock_id']}.csv"
        ohlc = load_ohlcv(symbol_path)
        stock_state = state[state["stock_id"].astype(str).str.zfill(6) == str(row["stock_id"]).zfill(6)].copy()
        out_path = out_dir / f"example_{i:02d}_{row['stock_id']}_{pd.Timestamp(row['date']).strftime('%Y%m%d')}.png"
        records.append(
            plot_example(
                row,
                ohlc,
                stock_state,
                out_path,
                lookback=args.lookback,
                lookahead=args.lookahead,
                match_lookback=args.match_lookback,
                state_price_is_log=args.state_price_is_log,
            )
        )

    pd.DataFrame(records).to_csv(out_dir / "examples_index.csv", index=False, encoding="utf-8-sig")
    (out_dir / "examples_config.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Examples: {len(records)}")
    print(f"Outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
