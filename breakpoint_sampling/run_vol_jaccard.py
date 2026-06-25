from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from breakpoint_sampling.extended_breaking_points import FuncEvtCPD


DEFAULT_METHODS = [
    "swt",
    "cum",
    "cusum_ls",
    "sprt",
    "gma",
    "glr",
    "brandt_glr",
    "e_detector",
    "ssr_cusum",
    "adaptive_cusum",
]


METHOD_KWARGS = {
    "swt": {"window": 20},
    "cusum_ls": {"warmup": 30, "drift": 0.0},
    "sprt": {"warmup": 20, "drift": 0.0},
    "gma": {"warmup": 20, "lam": 0.9},
    "glr": {"window": 60, "init_len": 20, "min_right_len": 5},
    "brandt_glr": {"window": 20, "min_global": 5},
    "e_detector": {"std_window": 30, "warmup": 50},
    "ssr_cusum": {"zeta": 0.25},
    "adaptive_cusum": {"std_window": 30},
}


def load_dif(path: Path, column: str) -> pd.Series:
    df = pd.read_parquet(path)
    if column not in df.columns:
        if len(df.columns) == 1:
            column = str(df.columns[0])
        else:
            raise ValueError(f"Column {column!r} not found. Available columns: {list(df.columns)}")
    series = df[column].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if not isinstance(series.index, pd.DatetimeIndex):
        series.index = pd.to_datetime(series.index)
    return series.diff().dropna()


def make_base_vol(dif: pd.Series, window: int) -> pd.Series:
    base = dif.rolling(window, min_periods=max(8, window // 4)).std()
    fallback = dif.expanding(min_periods=8).std()
    base = base.fillna(fallback).ffill().bfill()
    floor = float(dif.abs().median())
    if not np.isfinite(floor) or floor <= 0:
        floor = float(dif.abs().mean())
    if not np.isfinite(floor) or floor <= 0:
        floor = 1e-8
    base = base.replace([np.inf, -np.inf], np.nan).fillna(floor)
    return base.clip(lower=floor * 1e-3)


def count_events(method: str, dif: pd.Series, base_vol: pd.Series, scale: float, b2b: int) -> tuple[int, list]:
    events = FuncEvtCPD.detect(
        method,
        dif,
        base_vol * scale,
        b2b=b2b,
        **METHOD_KWARGS.get(method, {}),
    )
    return len(events), events


def tune_method(
    method: str,
    dif: pd.Series,
    base_vol: pd.Series,
    *,
    target: int,
    b2b: int,
    max_iter: int,
) -> dict:
    history = []

    def eval_scale(scale: float) -> tuple[int, list]:
        count, events = count_events(method, dif, base_vol, scale, b2b)
        history.append({"scale": float(scale), "count": int(count)})
        return count, events

    low = 1.0
    low_count, low_events = eval_scale(low)
    while low_count < target and low > 1e-12:
        low /= 2.0
        low_count, low_events = eval_scale(low)

    high = low if low_count < target else low * 2.0
    high_count, high_events = eval_scale(high)
    while high_count > target and high < 1e12:
        high *= 2.0
        high_count, high_events = eval_scale(high)

    candidates = [
        (abs(low_count - target), low_count, low, low_events),
        (abs(high_count - target), high_count, high, high_events),
    ]
    if low_count >= target and high_count <= target:
        for _ in range(max_iter):
            mid = float(np.sqrt(low * high))
            mid_count, mid_events = eval_scale(mid)
            candidates.append((abs(mid_count - target), mid_count, mid, mid_events))
            if mid_count > target:
                low = mid
            elif mid_count < target:
                high = mid
            else:
                break

    best_err, best_count, best_scale, best_events = min(candidates, key=lambda x: (x[0], x[2]))
    return {
        "method": method,
        "scale": float(best_scale),
        "count": int(best_count),
        "error": int(best_err),
        "events": best_events,
        "history": history,
    }


def jaccard_matrix(events_by_method: dict[str, Iterable]) -> pd.DataFrame:
    methods = list(events_by_method)
    sets = {name: set(pd.Timestamp(x) for x in events) for name, events in events_by_method.items()}
    mat = pd.DataFrame(np.eye(len(methods)), index=methods, columns=methods, dtype=float)
    for i, a in enumerate(methods):
        for j, b in enumerate(methods):
            if j < i:
                mat.loc[a, b] = mat.loc[b, a]
                continue
            union = sets[a] | sets[b]
            value = 1.0 if not union else len(sets[a] & sets[b]) / len(union)
            mat.loc[a, b] = value
    return mat


def plot_jaccard(mat: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(mat.values, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(mat.columns)), mat.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(mat.index)), mat.index)
    for i in range(len(mat.index)):
        for j in range(len(mat.columns)):
            val = mat.iloc[i, j]
            color = "white" if val < 0.45 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=8)
    ax.set_title("Jaccard similarity of tuned breakpoint events")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune external vol thresholds and plot breakpoint Jaccard matrix.")
    parser.add_argument("--data", default=r"data\TEST.parquet(1).gzip", type=Path)
    parser.add_argument("--column", default="p")
    parser.add_argument("--output-dir", default=Path("breakpoint_sampling/outputs"), type=Path)
    parser.add_argument("--target", default=2000, type=int)
    parser.add_argument("--b2b", default=1, type=int)
    parser.add_argument("--vol-window", default=128, type=int)
    parser.add_argument("--max-iter", default=16, type=int)
    parser.add_argument("--methods", nargs="*", default=DEFAULT_METHODS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    dif = load_dif(args.data, args.column)
    base_vol = make_base_vol(dif, args.vol_window)

    results = []
    events_by_method: dict[str, list] = {}
    histories = {}
    for method in args.methods:
        tuned = tune_method(
            method,
            dif,
            base_vol,
            target=args.target,
            b2b=args.b2b,
            max_iter=args.max_iter,
        )
        results.append(
            {
                "method": method,
                "scale": tuned["scale"],
                "count": tuned["count"],
                "target": args.target,
                "error": tuned["error"],
                "b2b": args.b2b,
            }
        )
        events_by_method[method] = list(tuned["events"])
        histories[method] = tuned["history"]
        print(f"{method:15s} scale={tuned['scale']:.8g} count={tuned['count']}")

    counts = pd.DataFrame(results).sort_values("method")
    counts.to_csv(output_dir / "tuned_counts.csv", index=False)

    event_rows = [
        {"method": method, "timestamp": pd.Timestamp(ts)}
        for method, events in events_by_method.items()
        for ts in events
    ]
    pd.DataFrame(event_rows).to_csv(output_dir / "events_by_method.csv", index=False)

    mat = jaccard_matrix(events_by_method)
    mat.to_csv(output_dir / "jaccard_matrix.csv")
    plot_jaccard(mat, output_dir / "jaccard_matrix.png")

    metadata = {
        "data": str(args.data),
        "column": args.column,
        "n_observations": int(len(dif)),
        "target": int(args.target),
        "b2b": int(args.b2b),
        "vol_window": int(args.vol_window),
        "method_kwargs": METHOD_KWARGS,
        "tuning_history": histories,
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    print(f"wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
