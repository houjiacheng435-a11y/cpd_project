"""Run HH/LL same-type change Hankel-DMD comparison experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _parse_int_list(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def _read_log_price_observable(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"event_id", "date", "bar_index", "extreme_type", "observable"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    df = df.sort_values(["date", "bar_index", "event_id"]).reset_index(drop=True)
    df["log_price"] = pd.to_numeric(df["observable"], errors="coerce")
    df = df.dropna(subset=["log_price"]).copy()
    df["extreme_type"] = df["extreme_type"].astype(str).str.lower()
    return df


def build_hh_ll_changes(observable_path: Path) -> dict[str, pd.DataFrame]:
    df = _read_log_price_observable(observable_path)
    highs = df[df["extreme_type"] == "high"].reset_index(drop=True).copy()
    lows = df[df["extreme_type"] == "low"].reset_index(drop=True).copy()
    highs["HH"] = highs["log_price"].diff()
    lows["LL"] = lows["log_price"].diff()
    hh = highs.dropna(subset=["HH"])[["event_id", "date", "bar_index", "log_price", "HH"]].reset_index(drop=True)
    ll = lows.dropna(subset=["LL"])[["event_id", "date", "bar_index", "log_price", "LL"]].reset_index(drop=True)

    length = min(len(hh), len(ll))
    paired = pd.DataFrame(
        {
            "cycle_id": np.arange(length),
            "high_event_id": hh.loc[: length - 1, "event_id"].to_numpy(),
            "high_date": hh.loc[: length - 1, "date"].to_numpy(),
            "low_event_id": ll.loc[: length - 1, "event_id"].to_numpy(),
            "low_date": ll.loc[: length - 1, "date"].to_numpy(),
            "HH": hh.loc[: length - 1, "HH"].to_numpy(float),
            "LL": ll.loc[: length - 1, "LL"].to_numpy(float),
        }
    )
    return {"hh": hh, "ll": ll, "paired": paired}


def _fit_hankel_dmd(data: np.ndarray, *, m: int, rank: int):
    try:
        from pydmd import HankelDMD
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Missing pydmd. Install it with: python -m pip install pydmd") from exc

    model = HankelDMD(svd_rank=rank, exact=False, d=m, reconstruction_method="first")
    model.fit(data)
    return model


def _predict_next(model, feature_count: int) -> np.ndarray:
    high_order_last = model.ho_snapshots[:, -1]
    pred = model.modes @ np.diag(model.eigs) @ np.linalg.pinv(model.modes) @ high_order_last
    return np.real(pred[-feature_count:])


def _metrics(pred: np.ndarray, true: np.ndarray, *, prefix: str = "") -> dict[str, float]:
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    err = pred - true
    abs_err = np.abs(err)
    true_dir = np.sign(true)
    pred_dir = np.sign(pred)
    direction_valid = true_dir != 0
    direction_accuracy = np.nan
    if np.any(direction_valid):
        direction_accuracy = float(np.mean(pred_dir[direction_valid] == true_dir[direction_valid]))
    return {
        f"{prefix}count": int(true.size),
        f"{prefix}error_mean": float(np.mean(err)),
        f"{prefix}error_std": float(np.std(err, ddof=0)),
        f"{prefix}mae": float(np.mean(abs_err)),
        f"{prefix}rmse": float(np.sqrt(np.mean(err**2))),
        f"{prefix}median_absolute_error": float(np.median(abs_err)),
        f"{prefix}direction_accuracy": direction_accuracy,
        f"{prefix}direction_valid_count": int(np.sum(direction_valid)),
    }


def _singular_values(model) -> np.ndarray:
    x_matrix = np.asarray(model.ho_snapshots[:, :-1], dtype=float)
    return np.linalg.svd(x_matrix, compute_uv=False)


def _run_case(
    values: np.ndarray,
    *,
    case_name: str,
    m: int,
    n: int,
    rank: int,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    if values.ndim == 1:
        data = values.reshape(1, -1)
    elif values.ndim == 2:
        data = values.T
    else:
        raise ValueError("values must be a 1D sequence or a 2D feature sequence")

    feature_count = data.shape[0]
    total_steps = data.shape[1]
    train_length = m + n + 1
    if train_length >= total_steps:
        raise ValueError(
            f"train window length {train_length} needs at least 1 out-of-window true point, "
            f"but the sequence only has {total_steps} observations"
        )

    dmd_pred: list[np.ndarray] = []
    zero_pred: list[np.ndarray] = []
    last_pred: list[np.ndarray] = []
    true_values: list[np.ndarray] = []

    for start in range(total_steps - train_length):
        window = data[:, start : start + train_length]
        model = _fit_hankel_dmd(window, m=m, rank=rank)
        dmd_pred.append(_predict_next(model, feature_count))
        zero_pred.append(np.zeros(feature_count))
        last_pred.append(window[:, -1])
        true_values.append(data[:, start + train_length])

    dmd_arr = np.vstack(dmd_pred)
    zero_arr = np.vstack(zero_pred)
    last_arr = np.vstack(last_pred)
    true_arr = np.vstack(true_values)

    tail_model = _fit_hankel_dmd(data[:, -train_length:], m=m, rank=rank)
    s = _singular_values(tail_model)
    energy = s**2
    energy_at_rank = float(np.cumsum(energy)[rank - 1] / np.sum(energy))
    eigs = np.asarray(tail_model.eigs, dtype=complex)

    summary = {
        "case": case_name,
        "m": m,
        "n": n,
        "rank": rank,
        "feature_count": feature_count,
        "train_observation_count": train_length,
        "total_observation_count": total_steps,
        "prediction_count": int(true_arr.shape[0]),
        "singular_value_1": float(s[0]),
        "singular_value_rank": float(s[rank - 1]),
        "energy_at_rank": energy_at_rank,
        "max_lambda_abs": float(np.max(np.abs(eigs))),
        "min_lambda_abs": float(np.min(np.abs(eigs))),
    }
    summary.update(_metrics(dmd_arr, true_arr, prefix="dmd_"))
    summary.update(_metrics(zero_arr, true_arr, prefix="zero_"))
    summary.update(_metrics(last_arr, true_arr, prefix="last_"))
    summary["dmd_rmse_improvement_vs_zero"] = (summary["zero_rmse"] - summary["dmd_rmse"]) / summary["zero_rmse"]
    summary["dmd_rmse_improvement_vs_last"] = (summary["last_rmse"] - summary["dmd_rmse"]) / summary["last_rmse"]
    summary["dmd_mae_improvement_vs_zero"] = (summary["zero_mae"] - summary["dmd_mae"]) / summary["zero_mae"]
    summary["dmd_mae_improvement_vs_last"] = (summary["last_mae"] - summary["dmd_mae"]) / summary["last_mae"]

    leading_rows: list[dict[str, float]] = []
    order = np.argsort(-np.abs(eigs))[: min(5, eigs.size)]
    for lead_id, mode_id in enumerate(order, start=1):
        lam = eigs[mode_id]
        angle = float(np.angle(lam))
        leading_rows.append(
            {
                "case": case_name,
                "m": m,
                "n": n,
                "rank": rank,
                "leading_id": lead_id,
                "mode_id": int(mode_id),
                "lambda_real": float(lam.real),
                "lambda_imag": float(lam.imag),
                "lambda_abs": float(abs(lam)),
                "lambda_angle": angle,
                "period_observations": float(2 * np.pi / abs(angle)) if abs(angle) > 1e-12 else np.inf,
            }
        )
    return summary, leading_rows


def run_experiment(
    *,
    observable_path: Path,
    out_dir: Path,
    m_values: Iterable[int],
    n_values: Iterable[int],
    rank_values: Iterable[int],
) -> dict[str, Path]:
    series = build_hh_ll_changes(observable_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    series["hh"].to_csv(out_dir / "hh_changes.csv", index=False, encoding="utf-8-sig")
    series["ll"].to_csv(out_dir / "ll_changes.csv", index=False, encoding="utf-8-sig")
    series["paired"].to_csv(out_dir / "hh_ll_paired_changes.csv", index=False, encoding="utf-8-sig")

    cases = {
        "HH_only": series["hh"]["HH"].to_numpy(float),
        "LL_only": series["ll"]["LL"].to_numpy(float),
        "HH_LL_2D": series["paired"][["HH", "LL"]].to_numpy(float),
    }

    summary_rows: list[dict[str, float]] = []
    leading_rows: list[dict[str, float]] = []
    failure_rows: list[dict[str, str]] = []
    for case_name, values in cases.items():
        for m in m_values:
            for n in n_values:
                for rank in rank_values:
                    try:
                        summary, leading = _run_case(values, case_name=case_name, m=m, n=n, rank=rank)
                        summary_rows.append(summary)
                        leading_rows.extend(leading)
                    except Exception as exc:
                        failure_rows.append({"case": case_name, "m": m, "n": n, "rank": rank, "error": str(exc)})

    summary_df = pd.DataFrame(summary_rows).sort_values(["case", "dmd_rmse"]).reset_index(drop=True)
    leading_df = pd.DataFrame(leading_rows)
    paths = {
        "summary": out_dir / "hh_ll_hankel_summary.csv",
        "leading_eigenvalues": out_dir / "hh_ll_leading_eigenvalues.csv",
        "config": out_dir / "config.json",
    }
    summary_df.to_csv(paths["summary"], index=False, encoding="utf-8-sig")
    leading_df.to_csv(paths["leading_eigenvalues"], index=False, encoding="utf-8-sig")
    if failure_rows:
        paths["failures"] = out_dir / "failures.csv"
        pd.DataFrame(failure_rows).to_csv(paths["failures"], index=False, encoding="utf-8-sig")

    paths["config"].write_text(
        json.dumps(
            {
                "observable_path": str(observable_path),
                "definition": {
                    "HH": "log(high_k) - log(high_{k-1})",
                    "LL": "log(low_k) - log(low_{k-1})",
                    "HH_LL_2D": "[HH_k, LL_k] aligned by same ordinal index",
                },
                "m_values": list(m_values),
                "n_values": list(n_values),
                "rank_values": list(rank_values),
                "baselines": {
                    "zero": "predict next change as 0",
                    "last": "predict next change as latest observed change in the training window",
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 1D and 2D Hankel-DMD experiments for HH/LL changes.")
    parser.add_argument("--observable-path", required=True, help="log_price observable.csv")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--m-values", default="5,10,20")
    parser.add_argument("--n-values", default="20,40,80")
    parser.add_argument("--rank-values", default="2,3,5")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    observable_path = Path(args.observable_path)
    out_dir = Path(args.out_dir) if args.out_dir else observable_path.parent / "hh_ll_block_hankel"
    paths = run_experiment(
        observable_path=observable_path,
        out_dir=out_dir,
        m_values=_parse_int_list(args.m_values),
        n_values=_parse_int_list(args.n_values),
        rank_values=_parse_int_list(args.rank_values),
    )
    print(f"output_dir: {out_dir}")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
