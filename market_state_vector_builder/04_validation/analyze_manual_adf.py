from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


GROUPS = ["low", "mid", "high"]
DEFAULT_STATE_COLUMNS = (
    "range_pct,expansion,structure_dir,range_pos,upper_break,lower_break,"
    "move_age,extrema_freq,speed_imbalance,speed_level"
)


def _savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate manual state variables with rolling ADF tests.")
    parser.add_argument("--final-state-path", default="market_state_vector_builder/outputs/shape_state_analysis_manual/final_state_vector.parquet")
    parser.add_argument("--out-dir", default="market_state_vector_builder/outputs/shape_state_analysis_manual/adf")
    parser.add_argument("--state-columns", default=DEFAULT_STATE_COLUMNS)
    parser.add_argument("--window", type=int, default=60)
    parser.add_argument("--min-obs", type=int, default=50)
    parser.add_argument("--maxlag", type=int, default=1)
    parser.add_argument("--regression", default="c", choices=["c", "ct", "ctt", "n"])
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--close-is-log", action="store_true", help="Treat final-state close as log(close).")
    parser.add_argument("--save-window-results", action="store_true")
    return parser.parse_args()


def run_rolling_adf(
    final_state: pd.DataFrame,
    state_columns: list[str],
    *,
    window: int,
    min_obs: int,
    maxlag: int,
    regression: str,
    alpha: float,
    close_is_log: bool,
) -> pd.DataFrame:
    records: list[dict] = []
    keep_cols = ["stock_id", "date", "close", *state_columns]
    df = final_state[[col for col in keep_cols if col in final_state.columns]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["stock_id", "date"])

    for stock_id, group in df.groupby("stock_id", sort=False):
        group = group.reset_index(drop=True)
        close = pd.to_numeric(group["close"], errors="coerce").to_numpy(float)
        if close_is_log:
            log_close = np.where(np.isfinite(close), close, np.nan)
        else:
            valid_price = np.isfinite(close) & (close > 0)
            log_close = np.full_like(close, np.nan, dtype=float)
            log_close[valid_price] = np.log(close[valid_price])

        for i in range(window - 1, len(group)):
            y = log_close[i - window + 1 : i + 1]
            y = y[np.isfinite(y)]
            if len(y) < min_obs or float(np.std(y)) <= 1e-12:
                continue

            try:
                stat, pvalue, usedlag, nobs, *_ = adfuller(
                    y,
                    maxlag=maxlag,
                    regression=regression,
                    autolag=None,
                )
            except Exception:
                continue

            row = {
                "stock_id": stock_id,
                "date": group.loc[i, "date"],
                "adf_stat": stat,
                "adf_pvalue": pvalue,
                "adf_reject": float(pvalue < alpha),
                "usedlag": usedlag,
                "nobs": nobs,
            }
            for col in state_columns:
                row[col] = group.loc[i, col]
            records.append(row)

    return pd.DataFrame(records)


def assign_quantile_group(s: pd.Series) -> pd.Series:
    q1 = s.quantile(1.0 / 3.0)
    q2 = s.quantile(2.0 / 3.0)
    out = pd.Series(index=s.index, dtype="object")
    out[s <= q1] = "low"
    out[(s > q1) & (s <= q2)] = "mid"
    out[s > q2] = "high"
    return out


def summarize_by_state_quantile(adf_df: pd.DataFrame, state_columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_records: list[dict] = []
    long_records: list[pd.DataFrame] = []

    for col in state_columns:
        valid = adf_df[np.isfinite(adf_df[col]) & np.isfinite(adf_df["adf_pvalue"])].copy()
        if valid.empty:
            continue
        valid["state_variable"] = col
        valid["state_group"] = assign_quantile_group(valid[col])
        long_records.append(valid[["stock_id", "date", "state_variable", "state_group", "adf_pvalue", "adf_reject"]])

        for group_name in GROUPS:
            part = valid[valid["state_group"] == group_name]
            if part.empty:
                continue
            summary_records.append(
                {
                    "state_variable": col,
                    "state_group": group_name,
                    "n_windows": len(part),
                    "n_stocks": part["stock_id"].nunique(),
                    "adf_reject_rate": part["adf_reject"].mean(),
                    "adf_pvalue_mean": part["adf_pvalue"].mean(),
                    "adf_pvalue_median": part["adf_pvalue"].median(),
                    "adf_pvalue_q25": part["adf_pvalue"].quantile(0.25),
                    "adf_pvalue_q75": part["adf_pvalue"].quantile(0.75),
                }
            )

    summary = pd.DataFrame(summary_records)
    long_df = pd.concat(long_records, ignore_index=True) if long_records else pd.DataFrame()
    return summary, long_df


def plot_reject_rate_bars(summary: pd.DataFrame, path: Path) -> None:
    variables = summary["state_variable"].drop_duplicates().tolist()
    x = np.arange(len(variables))
    width = 0.24

    fig, ax = plt.subplots(figsize=(11, 5))
    for offset, group_name in zip([-width, 0, width], GROUPS):
        values = []
        for variable in variables:
            part = summary[
                (summary["state_variable"] == variable) & (summary["state_group"] == group_name)
            ]["adf_reject_rate"]
            values.append(float(part.iloc[0]) if len(part) else np.nan)
        ax.bar(x + offset, values, width=width, label=group_name)

    ax.set_xticks(x)
    ax.set_xticklabels(variables, rotation=45, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("ADF reject rate")
    ax.set_title("ADF Reject Rate by Manual State Quantile")
    ax.legend(title="state group")
    _savefig(path)


def plot_pvalue_boxplot(long_df: pd.DataFrame, path: Path) -> None:
    variables = long_df["state_variable"].drop_duplicates().tolist()
    data = []
    labels = []
    positions = []
    pos = 1
    colors = {"low": "#4c78a8", "mid": "#f2cf5b", "high": "#d95f59"}

    for variable in variables:
        for group_name in GROUPS:
            values = long_df[
                (long_df["state_variable"] == variable) & (long_df["state_group"] == group_name)
            ]["adf_pvalue"].dropna()
            data.append(values.to_numpy(float))
            labels.append(group_name)
            positions.append(pos)
            pos += 1
        pos += 0.8

    fig, ax = plt.subplots(figsize=(14, 5))
    bp = ax.boxplot(data, positions=positions, widths=0.65, patch_artist=True, showfliers=False)
    for patch, label in zip(bp["boxes"], labels):
        patch.set_facecolor(colors[label])
        patch.set_alpha(0.75)
    for median in bp["medians"]:
        median.set_color("black")

    centers = [np.mean(positions[i * 3 : i * 3 + 3]) for i in range(len(variables))]
    ax.set_xticks(centers)
    ax.set_xticklabels(variables, rotation=45, ha="right")
    ax.axhline(0.05, color="black", linestyle="--", linewidth=1.0, label="p=0.05")
    ax.set_ylim(0, 1)
    ax.set_ylabel("ADF p-value")
    ax.set_title("ADF p-value Distribution by Manual State Quantile")
    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=colors[g], alpha=0.75, label=g) for g in GROUPS]
    ax.legend(handles=legend_handles, title="PC group", loc="upper right")
    _savefig(path)


def save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    state_columns = [col.strip() for col in args.state_columns.split(",") if col.strip()]

    config = vars(args).copy()
    config["state_columns"] = state_columns
    config["series"] = "rolling past-window log(close)"
    save_json(out_dir / "adf_config.json", config)

    final_state = pd.read_parquet(args.final_state_path)
    missing = [col for col in state_columns if col not in final_state.columns]
    if missing:
        raise ValueError(f"Missing state columns in final state: {missing}")

    adf_df = run_rolling_adf(
        final_state,
        state_columns,
        window=args.window,
        min_obs=args.min_obs,
        maxlag=args.maxlag,
        regression=args.regression,
        alpha=args.alpha,
        close_is_log=args.close_is_log,
    )
    summary, long_df = summarize_by_state_quantile(adf_df, state_columns)
    summary.to_csv(out_dir / "adf_by_state_quantile.csv", index=False, encoding="utf-8-sig")
    if args.save_window_results:
        adf_df.to_parquet(out_dir / "adf_window_results.parquet", index=False)

    plot_reject_rate_bars(summary, out_dir / "adf_reject_rate_bars.png")
    plot_pvalue_boxplot(long_df, out_dir / "adf_pvalue_boxplot.png")

    print(f"ADF windows: {len(adf_df)}")
    print(f"Stocks: {adf_df['stock_id'].nunique() if not adf_df.empty else 0}")
    print(f"State variables: {len(state_columns)}")
    print(f"Outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
