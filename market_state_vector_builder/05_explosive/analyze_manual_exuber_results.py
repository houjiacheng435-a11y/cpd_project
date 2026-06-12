from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


GROUPS = ["low", "mid", "high"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize sampled exuber window results for manual state variables.")
    parser.add_argument("--metadata-path", default="market_state_vector_builder/outputs/shape_state_analysis_manual/exuber_sample/input/exuber_window_metadata.csv")
    parser.add_argument("--result-path", default="market_state_vector_builder/outputs/shape_state_analysis_manual/exuber_sample/raw_r_results/exuber_window_results.csv")
    parser.add_argument("--out-dir", default="market_state_vector_builder/outputs/shape_state_analysis_manual/exuber_sample/summary")
    return parser.parse_args()


def _savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def summarize(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    state_rows_by_quantile = results[results["sample_type"] == "state_quantile"].copy()
    state_rows_by_quantile["gsadf_excess"] = state_rows_by_quantile["gsadf_stat"] - state_rows_by_quantile["gsadf_cv95"]
    split = state_rows_by_quantile["sample_name"].str.extract(r"^(.+)_(low|mid|high)$")
    state_rows_by_quantile["state_variable"] = split[0]
    state_rows_by_quantile["state_group"] = split[1]
    quantile_summary = (
        state_rows_by_quantile.groupby(["state_variable", "state_group"], as_index=False)
        .agg(
            n_windows=("window_id", "nunique"),
            n_stocks=("stock_id", "nunique"),
            explosive_rate=("gsadf_reject", "mean"),
            gsadf_stat_mean=("gsadf_stat", "mean"),
            gsadf_stat_median=("gsadf_stat", "median"),
            gsadf_excess_mean=("gsadf_excess", "mean"),
        )
    )

    state_rows = results[results["sample_type"] == "composite_state"].copy()
    state_rows["gsadf_excess"] = state_rows["gsadf_stat"] - state_rows["gsadf_cv95"]
    state_summary = (
        state_rows.groupby("sample_name", as_index=False)
        .agg(
            n_windows=("window_id", "nunique"),
            n_stocks=("stock_id", "nunique"),
            explosive_rate=("gsadf_reject", "mean"),
            gsadf_stat_mean=("gsadf_stat", "mean"),
            gsadf_stat_median=("gsadf_stat", "median"),
            gsadf_excess_mean=("gsadf_excess", "mean"),
        )
        .rename(columns={"sample_name": "state_label"})
        .sort_values("explosive_rate", ascending=False)
    )
    return quantile_summary, state_summary


def plot_quantile_heatmap(summary: pd.DataFrame, path: Path) -> None:
    mat = summary.pivot(index="state_variable", columns="state_group", values="explosive_rate").reindex(columns=GROUPS)
    fig, ax = plt.subplots(figsize=(6, max(4, 0.45 * len(mat))))
    vmax = max(0.01, np.nanmax(mat.to_numpy(float)))
    im = ax.imshow(mat.to_numpy(float), cmap="coolwarm", vmin=0.0, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(mat.columns)))
    ax.set_xticklabels(mat.columns)
    ax.set_yticks(np.arange(len(mat.index)))
    ax.set_yticklabels(mat.index)
    ax.set_title("Sampled Exuber GSADF Reject Rate by State Quantile")
    fig.colorbar(im, ax=ax, shrink=0.85)
    _savefig(path)


def plot_state_bar(summary: pd.DataFrame, path: Path) -> None:
    data = summary.sort_values("explosive_rate", ascending=True)
    fig, ax = plt.subplots(figsize=(9, max(4, 0.42 * len(data))))
    ax.barh(data["state_label"], data["explosive_rate"], color="#d95f59")
    ax.set_xlabel("GSADF reject rate")
    ax.set_title("Sampled Exuber GSADF Reject Rate by Composite State")
    for i, (_, row) in enumerate(data.iterrows()):
        ax.text(row["explosive_rate"], i, f"  n={int(row['n_windows'])}", va="center", fontsize=8)
    _savefig(path)


def plot_score_boxplot(results: pd.DataFrame, path: Path) -> None:
    state_rows = results[results["sample_type"] == "composite_state"].copy()
    states = (
        state_rows.groupby("sample_name")["gsadf_reject"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )
    data = [state_rows[state_rows["sample_name"] == state]["gsadf_stat"].dropna().to_numpy(float) for state in states]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot(data, tick_labels=states, showfliers=False)
    ax.tick_params(axis="x", rotation=45)
    ax.set_ylabel("GSADF statistic")
    ax.set_title("Sampled GSADF Statistic by Composite State")
    _savefig(path)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(args.metadata_path)
    res = pd.read_csv(args.result_path)
    res["gsadf_reject"] = res["gsadf_reject"].astype(bool)
    merged = meta.merge(res, on="window_id", how="inner")
    quantile_summary, state_summary = summarize(merged)
    quantile_summary.to_csv(out_dir / "explosive_by_state_quantile.csv", index=False, encoding="utf-8-sig")
    state_summary.to_csv(out_dir / "explosive_by_state.csv", index=False, encoding="utf-8-sig")

    plot_quantile_heatmap(quantile_summary, fig_dir / "explosive_rate_by_state_quantile_heatmap.png")
    plot_state_bar(state_summary, fig_dir / "explosive_rate_by_state_bar.png")
    plot_score_boxplot(merged, fig_dir / "gsadf_by_state_boxplot.png")

    config = vars(args).copy()
    (out_dir / "exuber_window_summary_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Windows: {merged['window_id'].nunique()}")
    print(f"Rows: {len(merged)}")
    print(f"Outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
