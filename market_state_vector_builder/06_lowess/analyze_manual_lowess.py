from __future__ import annotations

import argparse
import json
import warnings
import sys
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_STATE_COLUMNS = (
    "range_pct,expansion,structure_dir,range_pos,upper_break,lower_break,"
    "move_age,extrema_freq,speed_imbalance,speed_level"
)

BINARY_STATE_COLUMNS = {"upper_break", "lower_break"}
DEFAULT_SCORE_SUFFIX = "_score"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LOWESS analysis for manual market-state variables.")
    parser.add_argument(
        "--final-state-path",
        default="market_state_vector_builder/outputs/shape_state_analysis_manual/final_state_vector.parquet",
    )
    parser.add_argument(
        "--out-dir",
        default="market_state_vector_builder/outputs/shape_state_analysis_manual/06_lowess",
    )
    parser.add_argument("--state-columns", default=DEFAULT_STATE_COLUMNS)
    parser.add_argument("--use-scores", action="store_true", default=True, help="Use *_score columns for LOWESS x-axis.")
    parser.add_argument("--raw-state", action="store_false", dest="use_scores", help="Use raw state columns instead of *_score columns.")
    parser.add_argument("--return-horizons", default="5,10,20")
    parser.add_argument("--vol-horizons", default="5,10,20")
    parser.add_argument("--frac", type=float, default=0.25, help="LOWESS smoothing fraction.")
    parser.add_argument("--it", type=int, default=1, help="LOWESS robust reweighting iterations.")
    parser.add_argument("--max-points", type=int, default=8000, help="Max points used for LOWESS fit per plot.")
    parser.add_argument("--scatter-max-points", type=int, default=8000, help="Max points shown in scatter per plot.")
    parser.add_argument("--target-clip-lower", type=float, default=0.01, help="Lower quantile used to clip target variables.")
    parser.add_argument("--target-clip-upper", type=float, default=0.99, help="Upper quantile used to clip target variables.")
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def _savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_int_list(text: str) -> list[int]:
    values = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(int(item))
    if not values:
        raise ValueError("at least one horizon is required")
    return sorted(set(values))


def add_future_targets(
    final: pd.DataFrame,
    return_horizons: list[int],
    vol_horizons: list[int],
    clip_lower: float,
    clip_upper: float,
) -> pd.DataFrame:
    df = final.copy()
    if "stock_id" not in df.columns or "date" not in df.columns or "close" not in df.columns:
        raise ValueError("final state must contain stock_id, date, and close columns")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["stock_id", "date"]).reset_index(drop=True)

    target_cols: list[str] = []
    for stock_id, idx in df.groupby("stock_id", sort=False).groups.items():
        idx = pd.Index(idx)
        close = pd.to_numeric(df.loc[idx, "close"], errors="coerce").to_numpy(dtype=float)
        n = len(close)
        if n == 0:
            continue

        for horizon in return_horizons:
            values = np.full(n, np.nan, dtype=float)
            if horizon > 0 and n > horizon:
                values[: n - horizon] = close[horizon:] - close[:-horizon]
            col = f"future_return_k{horizon:02d}"
            df.loc[idx, col] = values
            target_cols.append(col)

        daily_ret = np.diff(close)
        for horizon in vol_horizons:
            values = np.full(n, np.nan, dtype=float)
            if horizon > 0 and len(daily_ret) >= horizon:
                for i in range(0, n - horizon):
                    window = daily_ret[i : i + horizon]
                    if np.all(np.isfinite(window)):
                        values[i] = float(np.std(window, ddof=0))
            col = f"future_vol_k{horizon:02d}"
            df.loc[idx, col] = values
            target_cols.append(col)

    for col in sorted(set(target_cols)):
        values = pd.to_numeric(df[col], errors="coerce")
        valid = values.replace([np.inf, -np.inf], np.nan).dropna()
        if valid.empty:
            continue
        lo = valid.quantile(clip_lower)
        hi = valid.quantile(clip_upper)
        df[col] = values.clip(lower=lo, upper=hi)

    return df


def sample_for_lowess(df: pd.DataFrame, x_col: str, y_col: str, *, max_points: int, seed: int) -> pd.DataFrame:
    clean = df[[x_col, y_col]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if clean.empty:
        return clean
    clean = clean.sort_values(x_col).reset_index(drop=True)
    if len(clean) > max_points:
        idx = np.linspace(0, len(clean) - 1, max_points).astype(int)
        clean = clean.iloc[idx].copy()
    return clean


def clip_series_by_quantile(s: pd.Series, lower: float, upper: float) -> pd.Series:
    values = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    valid = values.dropna()
    if valid.empty:
        return values
    lo = valid.quantile(lower)
    hi = valid.quantile(upper)
    return values.clip(lower=lo, upper=hi)


def fit_lowess(x: np.ndarray, y: np.ndarray, frac: float, it: int) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        fitted = lowess(y, x, frac=frac, it=it, return_sorted=True)
    return fitted


def plot_lowess(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    out_path: Path,
    *,
    frac: float,
    it: int,
    max_points: int,
    scatter_max_points: int,
    seed: int,
) -> dict:
    clean = df[[x_col, y_col]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if clean.empty:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.text(0.5, 0.5, "No valid data", ha="center", va="center")
        ax.set_axis_off()
        _savefig(out_path)
        return {"x_column": x_col, "y_column": y_col, "n": 0, "output": str(out_path)}

    clean = clean.sort_values(x_col).reset_index(drop=True)
    fit_df = sample_for_lowess(clean, x_col, y_col, max_points=max_points, seed=seed)
    if fit_df.empty:
        fit_df = clean.copy()

    unique_x = fit_df[x_col].nunique(dropna=True)
    if unique_x <= 8:
        fit_df = fit_df.groupby(x_col, as_index=False)[y_col].mean().sort_values(x_col).reset_index(drop=True)

    x = fit_df[x_col].to_numpy(dtype=float)
    y = fit_df[y_col].to_numpy(dtype=float)
    fitted = None
    if len(fit_df) >= 3 and np.nanstd(x) > 1e-12:
        try:
            fitted = fit_lowess(x, y, frac=frac, it=it)
        except Exception:
            fitted = None

    scatter_df = clean
    if len(scatter_df) > scatter_max_points:
        scatter_df = scatter_df.iloc[np.linspace(0, len(scatter_df) - 1, scatter_max_points).astype(int)].copy()

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    x_scatter = scatter_df[x_col].to_numpy(dtype=float)
    y_scatter = scatter_df[y_col].to_numpy(dtype=float)
    x_range = float(np.nanmax(x_scatter) - np.nanmin(x_scatter)) if len(x_scatter) else 0.0
    max_abs_x = float(np.nanmax(np.abs(x_scatter))) if len(x_scatter) else 0.0
    if max_abs_x <= 1.0 and len(np.unique(x_scatter)) <= 3:
        jitter_scale = max(x_range * 0.02, 0.02)
        rng = np.random.default_rng(seed)
        x_plot = x_scatter + rng.normal(0.0, jitter_scale, size=len(x_scatter))
    else:
        x_plot = x_scatter

    ax.scatter(x_plot, y_scatter, s=8, alpha=0.14, color="#6c757d", edgecolors="none")
    if fitted is not None:
        ax.plot(fitted[:, 0], fitted[:, 1], color="#c0392b", linewidth=2.0)

    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.4)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title)
    ax.grid(True, alpha=0.18)

    if np.isfinite(np.nanmin(x_scatter)) and np.isfinite(np.nanmax(x_scatter)):
        x_pad = max((np.nanmax(x_scatter) - np.nanmin(x_scatter)) * 0.04, 1e-6)
        ax.set_xlim(np.nanmin(x_scatter) - x_pad, np.nanmax(x_scatter) + x_pad)

    _savefig(out_path)

    return {
        "x_column": x_col,
        "y_column": y_col,
        "n": int(len(clean)),
        "fit_n": int(len(fit_df)),
        "output": str(out_path),
        "x_min": float(np.nanmin(clean[x_col])),
        "x_max": float(np.nanmax(clean[x_col])),
        "y_min": float(np.nanmin(clean[y_col])),
        "y_max": float(np.nanmax(clean[y_col])),
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    state_columns = [col.strip() for col in args.state_columns.split(",") if col.strip()]
    return_horizons = parse_int_list(args.return_horizons)
    vol_horizons = parse_int_list(args.vol_horizons)

    config = vars(args).copy()
    config["state_columns"] = state_columns
    config["return_horizons"] = return_horizons
    config["vol_horizons"] = vol_horizons
    save_json(out_dir / "lowess_config.json", config)

    final = pd.read_parquet(args.final_state_path)
    final["date"] = pd.to_datetime(final["date"])

    missing = [col for col in state_columns if col not in final.columns]
    if missing:
        raise ValueError(f"Missing state columns in final state: {missing}")

    final = add_future_targets(final, return_horizons, vol_horizons, float(args.target_clip_lower), float(args.target_clip_upper))
    if "valid_manual_state" in final.columns:
        final = final[final["valid_manual_state"].fillna(False)].copy()

    lowess_state_columns = [col for col in state_columns if col not in BINARY_STATE_COLUMNS]

    target_specs: list[tuple[str, str]] = []
    for horizon in return_horizons:
        target_specs.append((f"return_k{horizon:02d}", f"future_return_k{horizon:02d}"))
    for horizon in vol_horizons:
        target_specs.append((f"vol_k{horizon:02d}", f"future_vol_k{horizon:02d}"))

    index_records: list[dict[str, object]] = []
    for target_name, target_col in target_specs:
        target_dir = out_dir / target_name
        if target_dir.exists():
            shutil.rmtree(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        per_target_records: list[dict[str, object]] = []

        for i, base_x_col in enumerate(lowess_state_columns, start=1):
            x_col = f"{base_x_col}{DEFAULT_SCORE_SUFFIX}" if args.use_scores else base_x_col
            title = f"{x_col} vs {target_col}"
            fig_path = target_dir / f"{i:02d}_{x_col}.png"

            plot_df = final[[x_col, target_col]].copy()
            if not args.use_scores:
                plot_df[x_col] = clip_series_by_quantile(plot_df[x_col], 0.01, 0.99)
            plot_df[target_col] = clip_series_by_quantile(plot_df[target_col], float(args.target_clip_lower), float(args.target_clip_upper))

            record = plot_lowess(
                plot_df,
                x_col,
                target_col,
                title,
                fig_path,
                frac=float(args.frac),
                it=int(args.it),
                max_points=int(args.max_points),
                scatter_max_points=int(args.scatter_max_points),
                seed=int(args.random_state) + i,
            )
            record["variable"] = x_col
            record["target_name"] = target_name
            per_target_records.append(record)
            index_records.append(record | {"variable": x_col, "target_name": target_name})

        pd.DataFrame(per_target_records).to_csv(target_dir / "index.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame(index_records).to_csv(out_dir / "lowess_index.csv", index=False, encoding="utf-8-sig")
    print(f"Targets: {len(target_specs)}")
    print(f"Variables: {len(state_columns)}")
    print(f"Outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
