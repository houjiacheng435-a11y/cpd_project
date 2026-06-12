from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from shape_state_variables import build_shape_state_variables_from_csv


STATE_COLUMNS = [
    "range_pct",
    "expansion",
    "structure_dir",
    "range_pos",
    "upper_break",
    "lower_break",
    "move_age",
    "extrema_freq",
    "speed_imbalance",
    "speed_level",
]
BINARY_SCORE_COLUMNS = {"upper_break", "lower_break"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the final manual market-state vector.")
    parser.add_argument("--input-dir", default="data/a_share_1d_akshare/symbols")
    parser.add_argument("--state-path", default=None, help="Optional existing shape_state_variables parquet to reuse.")
    parser.add_argument("--out-dir", default="market_state_vector_builder/outputs/shape_state_analysis_manual")
    parser.add_argument("--max-symbols", type=int, default=80)
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end", default="2026-05-29")
    parser.add_argument("--log-price", action="store_true", help="Use log-transformed OHLC from CPD onward.")
    parser.add_argument("--detector-method", default="cusum")
    parser.add_argument("--detector-q", type=float, default=1.0)
    parser.add_argument("--cpd-confirm-lag", type=int, default=0)
    parser.add_argument("--n-lags", type=int, default=0)
    parser.add_argument("--clip-lower", type=float, default=0.01)
    parser.add_argument("--clip-upper", type=float, default=0.99)
    parser.add_argument("--score-window", type=int, default=1000, help="Rolling history window for realtime score quantiles.")
    parser.add_argument("--score-min-periods", type=int, default=100, help="Minimum prior observations needed for rolling scores.")
    parser.add_argument("--score-lower", type=float, default=0.05)
    parser.add_argument("--score-upper", type=float, default=0.95)
    return parser.parse_args()


def _savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def standardize_by_stock(
    df: pd.DataFrame,
    columns: list[str],
    clip_lower: float,
    clip_upper: float,
) -> pd.DataFrame:
    out = df[["stock_id", "date"] + columns].copy()
    frames = []
    for _, group in out.groupby("stock_id", sort=False):
        group = group.copy()
        x = group[columns].astype(float)
        lo = x.quantile(clip_lower)
        hi = x.quantile(clip_upper)
        x = x.clip(lower=lo, upper=hi, axis=1)
        mean = x.mean()
        std = x.std(ddof=0).replace(0.0, np.nan)
        group.loc[:, columns] = (x - mean) / std
        frames.append(group)
    return pd.concat(frames, ignore_index=True)


def build_state_table(args: argparse.Namespace, out_dir: Path) -> pd.DataFrame:
    if args.state_path:
        table = pd.read_parquet(args.state_path)
        out_path = out_dir / "shape_state_variables.parquet"
        if Path(args.state_path).resolve() != out_path.resolve():
            table.to_parquet(out_path, index=False)
        return table

    files = sorted(Path(args.input_dir).glob("*.csv"))[: args.max_symbols]
    if not files:
        raise FileNotFoundError(f"No CSV files found under {args.input_dir}")

    frames: list[pd.DataFrame] = []
    t0 = time.perf_counter()
    for i, path in enumerate(files, start=1):
        stock_t0 = time.perf_counter()
        state = build_shape_state_variables_from_csv(
            path,
            log_price=args.log_price,
            n_lags=args.n_lags,
            detector_method=args.detector_method,
            detector_q=args.detector_q,
            cpd_confirm_lag=args.cpd_confirm_lag,
        )
        state = state[(state["date"] >= pd.Timestamp(args.start)) & (state["date"] <= pd.Timestamp(args.end))]
        frames.append(state)
        elapsed = time.perf_counter() - t0
        print(
            f"[{i}/{len(files)}] {path.stem}: rows={len(state)} "
            f"stock_elapsed={time.perf_counter() - stock_t0:.1f}s total_elapsed={elapsed / 60:.1f}m",
            flush=True,
        )

    table = pd.concat(frames, ignore_index=True)
    table.to_parquet(out_dir / "shape_state_variables.parquet", index=False)
    return table


def build_manual_state_vector(state: pd.DataFrame) -> pd.DataFrame:
    df = state.copy()
    df["speed_imbalance"] = df["high_speed"] - df["low_speed"]
    df["speed_level"] = df["high_speed"] + df["low_speed"]

    base_columns = [
        "stock_id",
        "date",
        "close",
        "has_structural_window",
        "num_confirmed_extrema",
        "starts_high",
        "high1",
        "high2",
        "low1",
        "low2",
    ]
    keep = [col for col in base_columns if col in df.columns] + STATE_COLUMNS
    final = df[keep].copy()
    final["valid_manual_state"] = final[STATE_COLUMNS].replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
    return final


def add_rolling_quantile_scores(
    final: pd.DataFrame,
    window: int = 1000,
    min_periods: int = 100,
    lower: float = 0.05,
    upper: float = 0.95,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = final.copy()
    rows: list[dict[str, object]] = []
    out["_input_order"] = np.arange(len(out))
    out = out.sort_values(["stock_id", "date", "_input_order"]).reset_index(drop=True)

    for col in STATE_COLUMNS:
        score_col = f"{col}_score"
        out[score_col] = np.nan

        if col in BINARY_SCORE_COLUMNS:
            values = pd.to_numeric(out[col], errors="coerce")
            out[score_col] = np.where(values.notna(), values.clip(0, 1) * 2.0 - 1.0, np.nan)
            rows.append(
                {
                    "variable": col,
                    "score_column": score_col,
                    "method": "binary_fixed",
                    "window": np.nan,
                    "min_periods": np.nan,
                    "lower_quantile": np.nan,
                    "upper_quantile": np.nan,
                }
            )
            continue

        q_low = pd.Series(np.nan, index=out.index, dtype=float)
        q_high = pd.Series(np.nan, index=out.index, dtype=float)
        for _, idx in out.groupby("stock_id", sort=False).groups.items():
            values = pd.to_numeric(out.loc[idx, col], errors="coerce").replace([np.inf, -np.inf], np.nan)
            history = values.shift(1)
            q_low.loc[idx] = history.rolling(window=window, min_periods=min_periods).quantile(lower).to_numpy()
            q_high.loc[idx] = history.rolling(window=window, min_periods=min_periods).quantile(upper).to_numpy()

        values = pd.to_numeric(out[col], errors="coerce")
        denom = q_high - q_low
        valid = np.isfinite(values) & np.isfinite(q_low) & np.isfinite(q_high) & (denom.abs() > 1e-12)
        mapped = pd.Series(np.nan, index=out.index, dtype=float)
        mapped.loc[valid] = 2.0 * (values.loc[valid] - q_low.loc[valid]) / denom.loc[valid] - 1.0
        out[score_col] = mapped.clip(-1.0, 1.0)
        rows.append(
            {
                "variable": col,
                "score_column": score_col,
                "method": "rolling_prior_quantile",
                "window": window,
                "min_periods": min_periods,
                "lower_quantile": lower,
                "upper_quantile": upper,
            }
        )

    out = out.sort_values("_input_order").drop(columns="_input_order").reset_index(drop=True)
    return out, pd.DataFrame(rows)


def plot_correlation(final: pd.DataFrame, out_dir: Path, clip_lower: float, clip_upper: float) -> None:
    valid = final[final["valid_manual_state"]].copy()
    if valid.empty:
        return

    std_df = standardize_by_stock(valid, STATE_COLUMNS, clip_lower, clip_upper)
    corr = std_df[STATE_COLUMNS].corr()
    corr.to_csv(out_dir / "manual_state_correlation.csv", encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(corr.to_numpy(float), cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(corr.columns)))
    ax.set_yticks(np.arange(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
    ax.set_yticklabels(corr.index, fontsize=8)
    ax.set_title("Manual State Variable Correlation")
    fig.colorbar(im, ax=ax, shrink=0.8)
    _savefig(out_dir / "manual_state_correlation_heatmap.png")


def write_readme(out_dir: Path) -> None:
    text = """# Manual Market State Vector

This is the current main workflow. It does not use PCA.

## State Variables

- `range_pct`: structural width in log-price space.
- `expansion`: width change, equal to `high_change - low_change`.
- `structure_dir`: continuous structural direction, equal to `high_change + low_change`.
- `range_pos`: current position in the static structural range.
- `upper_break`: 1 if close is above the dynamic upper line, otherwise 0.
- `lower_break`: 1 if close is below the dynamic lower line, otherwise 0.
- `move_age`: bars since the latest structural-window endpoint, scaled by recent median extremum gap.
- `extrema_freq`: confirmed-extrema frequency inside the structural window.
- `speed_imbalance`: `high_speed - low_speed`.
- `speed_level`: `high_speed + low_speed`.

## Scores

Continuous `*_score` columns use each stock's prior 1000 valid states to estimate
rolling 5%/95% quantiles and map the current value to `[-1, 1]`. The current row
is excluded from its own quantile estimate. `upper_break_score` and
`lower_break_score` use fixed binary mapping: `0 -> -1`, `1 -> 1`.
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args).copy()
    config["method"] = "manual_market_state_vector_without_pca"
    config["state_columns"] = STATE_COLUMNS
    save_json(out_dir / "manual_state_config.json", config)

    state = build_state_table(args, out_dir)
    final = build_manual_state_vector(state)
    final, score_config = add_rolling_quantile_scores(
        final,
        window=args.score_window,
        min_periods=args.score_min_periods,
        lower=args.score_lower,
        upper=args.score_upper,
    )

    final.to_parquet(out_dir / "final_state_vector.parquet", index=False)
    score_config.to_csv(out_dir / "manual_state_score_config.csv", index=False, encoding="utf-8-sig")
    final[STATE_COLUMNS].describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).T.to_csv(
        out_dir / "manual_state_summary.csv",
        encoding="utf-8-sig",
    )
    plot_correlation(final, out_dir, args.clip_lower, args.clip_upper)
    write_readme(out_dir)

    print(f"Rows: {len(final)}")
    print(f"Valid manual states: {int(final['valid_manual_state'].sum())}")
    print(f"Outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
