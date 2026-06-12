from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


GROUPS = ["low", "mid", "high"]
DEFAULT_STATE_COLUMNS = (
    "range_pct,expansion,structure_dir,range_pos,upper_break,lower_break,"
    "move_age,extrema_freq,speed_imbalance,speed_level"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample local windows for R exuber tests by manual state variables.")
    parser.add_argument("--final-state-path", default="market_state_vector_builder/outputs/shape_state_analysis_manual/final_state_vector.parquet")
    parser.add_argument("--out-dir", default="market_state_vector_builder/outputs/shape_state_analysis_manual/exuber_sample/input")
    parser.add_argument("--state-columns", default=DEFAULT_STATE_COLUMNS)
    parser.add_argument("--window", type=int, default=60)
    parser.add_argument("--min-obs", type=int, default=50)
    parser.add_argument("--samples-per-group", type=int, default=120)
    parser.add_argument("--samples-per-state", type=int, default=150)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--close-is-log", action="store_true", help="Treat final-state close as log(close).")
    return parser.parse_args()


def assign_quantile_group(s: pd.Series) -> pd.Series:
    q1 = s.quantile(1.0 / 3.0)
    q2 = s.quantile(2.0 / 3.0)
    out = pd.Series(index=s.index, dtype="object")
    out[s <= q1] = "low"
    out[(s > q1) & (s <= q2)] = "mid"
    out[s > q2] = "high"
    return out


def add_state_groups(df: pd.DataFrame, state_columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in state_columns:
        out[f"{col}_group"] = assign_quantile_group(out[col])
    return out


def label_combo_states(df: pd.DataFrame) -> pd.Series:
    state = pd.Series("other", index=df.index, dtype="object")
    range_pct = df["range_pct_group"]
    expansion = df["expansion_group"]
    structure_dir = df["structure_dir_group"]
    range_pos = df["range_pos_group"]
    upper_break = df["upper_break_group"]
    lower_break = df["lower_break_group"]
    move_age = df["move_age_group"]
    extrema_freq = df["extrema_freq_group"]
    speed_level = df["speed_level_group"]

    state[(range_pct == "high") & (expansion == "high")] = "wide_expanding"
    state[(range_pct == "low") & (expansion == "high")] = "narrow_expanding"
    state[(upper_break == "high") & (range_pos == "high")] = "upper_breakout"
    state[(lower_break == "high") & (range_pos == "low")] = "lower_breakdown"
    state[(extrema_freq == "high") & (speed_level == "high")] = "fast_chop"
    state[(move_age == "high") & (speed_level == "low")] = "slow_extension"
    state[(structure_dir == "high") & (range_pos == "high")] = "up_structure_high_pos"
    state[(structure_dir == "low") & (range_pos == "low")] = "down_structure_low_pos"
    return state


def sample_rows(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if len(df) <= n:
        return df.copy()
    return df.sample(n=n, random_state=seed)


def build_windows(
    df: pd.DataFrame,
    sampled: pd.DataFrame,
    *,
    window: int,
    min_obs: int,
    close_is_log: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values(["stock_id", "date"]).copy()
    grouped = {stock: g.reset_index(drop=True) for stock, g in df.groupby("stock_id", sort=False)}
    records = []
    meta_records = []

    for k, row in enumerate(sampled.itertuples(index=False), start=1):
        stock_id = row.stock_id
        date = pd.Timestamp(row.date)
        group = grouped[stock_id]
        pos_arr = np.flatnonzero(group["date"].to_numpy(dtype="datetime64[ns]") == np.datetime64(date))
        if len(pos_arr) == 0:
            continue
        end_pos = int(pos_arr[0])
        start_pos = end_pos - window + 1
        if start_pos < 0:
            continue
        segment = group.iloc[start_pos : end_pos + 1].copy()
        if close_is_log:
            segment = segment[np.isfinite(segment["close"])]
        else:
            segment = segment[np.isfinite(segment["close"]) & (segment["close"] > 0)]
        if len(segment) < min_obs:
            continue
        window_id = f"w{k:06d}"
        log_close = segment["close"].to_numpy(float) if close_is_log else np.log(segment["close"].to_numpy(float))
        for t, (_, seg_row) in enumerate(segment.iterrows(), start=1):
            records.append(
                {
                    "window_id": window_id,
                    "t": t,
                    "stock_id": stock_id,
                    "date": seg_row["date"].strftime("%Y-%m-%d"),
                    "log_close": log_close[t - 1],
                }
            )
        meta_records.append(row._asdict() | {"window_id": window_id})

    return pd.DataFrame(records), pd.DataFrame(meta_records)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng_seed = int(args.random_state)

    state_columns = [col.strip() for col in args.state_columns.split(",") if col.strip()]

    df = pd.read_parquet(args.final_state_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["valid_manual_state"].fillna(False)].copy()
    missing = [col for col in state_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing state columns: {missing}")
    df = add_state_groups(df, state_columns)
    df["state_label"] = label_combo_states(df)

    samples = []
    for col in state_columns:
        for group_name in GROUPS:
            part = df[df[f"{col}_group"] == group_name]
            sampled = sample_rows(part, args.samples_per_group, rng_seed)
            sampled = sampled.assign(sample_type="state_quantile", sample_name=f"{col}_{group_name}")
            samples.append(sampled)
            rng_seed += 1

    states = [
        "wide_expanding",
        "narrow_expanding",
        "upper_breakout",
        "lower_breakdown",
        "fast_chop",
        "slow_extension",
        "up_structure_high_pos",
        "down_structure_low_pos",
    ]
    for state_name in states:
        part = df[df["state_label"] == state_name]
        sampled = sample_rows(part, args.samples_per_state, rng_seed)
        sampled = sampled.assign(sample_type="composite_state", sample_name=state_name)
        samples.append(sampled)
        rng_seed += 1

    sampled_all = pd.concat(samples, ignore_index=True)
    windows, metadata = build_windows(
        df,
        sampled_all,
        window=args.window,
        min_obs=args.min_obs,
        close_is_log=args.close_is_log,
    )
    windows.to_csv(out_dir / "exuber_windows.csv", index=False, encoding="utf-8-sig")
    metadata.to_csv(out_dir / "exuber_window_metadata.csv", index=False, encoding="utf-8-sig")
    config = vars(args).copy()
    config["state_columns"] = state_columns
    (out_dir / "exuber_window_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Sampled rows: {len(sampled_all)}")
    print(f"Windows exported: {metadata['window_id'].nunique() if not metadata.empty else 0}")
    print(f"Window observations: {len(windows)}")
    print(f"Outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
