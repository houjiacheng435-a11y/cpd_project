"""Run a PyDMD HankelDMD grid for single-observable comparison."""

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

from Hankel_DMD import fit_hankel_dmd


def _parse_int_list(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def _read_observable(path: Path) -> np.ndarray:
    df = pd.read_csv(path)
    if "observable" not in df.columns:
        raise ValueError(f"{path} missing observable column")
    sort_cols = [col for col in ["date", "bar_index", "event_id"] if col in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)
    x = pd.to_numeric(df["observable"], errors="coerce").to_numpy(float)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("observable must be a non-empty 1D sequence")
    if not np.isfinite(x).all():
        raise ValueError("observable cannot contain NaN or inf")
    return x


def _fit_pydmd_hankel(x: np.ndarray, *, m: int, rank: int):
    try:
        from pydmd import HankelDMD
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Missing pydmd. Install it with: python -m pip install pydmd") from exc

    model = HankelDMD(svd_rank=rank, exact=False, d=m, reconstruction_method="first")
    model.fit(x.reshape(1, -1))
    return model


def _predict_next_value(model) -> float:
    high_order_last = model.ho_snapshots[:, -1]
    next_high_order = model.modes @ np.diag(model.eigs) @ np.linalg.pinv(model.modes) @ high_order_last
    return float(np.real(next_high_order[-1]))


def _prediction_errors(x: np.ndarray, *, m: int, n: int, rank: int) -> dict[str, float]:
    train_length = m + n + 1
    if train_length >= x.size:
        raise ValueError(
            f"train window length {train_length} needs at least 1 out-of-window true point, "
            f"but observable only has {x.size} points"
        )

    pred_values: list[float] = []
    true_values: list[float] = []
    for start in range(x.size - train_length):
        window = x[start : start + train_length]
        model = _fit_pydmd_hankel(window, m=m, rank=rank)
        pred_values.append(_predict_next_value(model))
        true_values.append(float(x[start + train_length]))

    pred_arr = np.asarray(pred_values, dtype=float)
    true_arr = np.asarray(true_values, dtype=float)
    error = pred_arr - true_arr
    abs_error = np.abs(error)
    midpoint = np.abs((pred_arr + true_arr) / 2.0)
    midpoint_relative_abs_error = np.full_like(abs_error, np.nan, dtype=float)
    valid_midpoint = midpoint > np.finfo(float).eps
    midpoint_relative_abs_error[valid_midpoint] = abs_error[valid_midpoint] / midpoint[valid_midpoint]

    return {
        "prediction_count": int(error.size),
        "prediction_error_mean": float(np.mean(error)),
        "prediction_error_std": float(np.std(error, ddof=0)),
        "prediction_mae": float(np.mean(abs_error)),
        "prediction_rmse": float(np.sqrt(np.mean(error**2))),
        "prediction_median_absolute_error": float(np.median(abs_error)),
        "midpoint_relative_abs_error_mean": float(np.nanmean(midpoint_relative_abs_error)),
        "midpoint_relative_abs_error_median": float(np.nanmedian(midpoint_relative_abs_error)),
        "midpoint_relative_abs_error_valid_count": int(np.sum(valid_midpoint)),
        "last_prediction_true": float(true_arr[-1]),
        "last_prediction_pred": float(pred_arr[-1]),
        "last_prediction_error": float(error[-1]),
    }


def _tail_spectrum(x: np.ndarray, *, m: int, n: int, rank: int) -> tuple[pd.DataFrame, np.ndarray]:
    train_length = m + n + 1
    window = x[-train_length:]
    model = _fit_pydmd_hankel(window, m=m, rank=rank)
    eigs = np.asarray(model.eigs, dtype=complex)
    eigenvalues = pd.DataFrame(
        {
            "mode_id": np.arange(eigs.size),
            "lambda_real": eigs.real,
            "lambda_imag": eigs.imag,
            "lambda_abs": np.abs(eigs),
            "lambda_angle": np.angle(eigs),
        }
    )
    x_matrix = np.asarray(model.ho_snapshots[:, :-1], dtype=float)
    singular_values = np.linalg.svd(x_matrix, compute_uv=False)
    return eigenvalues, singular_values


def _merge_local_comparison(out_dir: Path, local_grid_path: Path, pydmd_summary: pd.DataFrame) -> Path | None:
    if not local_grid_path.exists():
        return None
    local = pd.read_csv(local_grid_path)
    keys = ["m", "n", "rank"]
    cols = keys + [
        "prediction_count",
        "prediction_mae",
        "prediction_rmse",
        "prediction_median_absolute_error",
        "midpoint_relative_abs_error_mean",
        "midpoint_relative_abs_error_median",
    ]
    missing = [col for col in cols if col not in local.columns]
    if missing:
        raise ValueError(f"{local_grid_path} missing comparison columns: {missing}")

    local_small = local[cols].rename(columns={col: f"local_{col}" for col in cols if col not in keys})
    pydmd_small = pydmd_summary[cols].rename(columns={col: f"pydmd_{col}" for col in cols if col not in keys})
    merged = pd.merge(local_small, pydmd_small, on=keys, how="inner")
    merged["rmse_diff_pydmd_minus_local"] = merged["pydmd_prediction_rmse"] - merged["local_prediction_rmse"]
    merged["mae_diff_pydmd_minus_local"] = merged["pydmd_prediction_mae"] - merged["local_prediction_mae"]
    merged["midpoint_median_diff_pydmd_minus_local"] = (
        merged["pydmd_midpoint_relative_abs_error_median"]
        - merged["local_midpoint_relative_abs_error_median"]
    )
    path = out_dir / "comparison_with_local.csv"
    merged.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def _merge_local_eigenvalue_comparison(out_dir: Path, local_grid_path: Path, pydmd_leading: pd.DataFrame) -> Path | None:
    local_leading_path = local_grid_path.with_name("grid_leading_eigenvalues.csv")
    if not local_leading_path.exists():
        return None
    local = pd.read_csv(local_leading_path)
    if "train_observable_count" not in local.columns and "used_observable_count" in local.columns:
        local = local.rename(columns={"used_observable_count": "train_observable_count"})

    keys = ["m", "n", "rank", "train_observable_count", "leading_id"]
    cols = keys + ["lambda_real", "lambda_imag", "lambda_abs", "lambda_angle"]
    missing = [col for col in cols if col not in local.columns]
    if missing:
        raise ValueError(f"{local_leading_path} missing comparison columns: {missing}")

    local_small = local[cols].rename(columns={col: f"local_{col}" for col in cols if col not in keys})
    pydmd_small = pydmd_leading[cols].rename(columns={col: f"pydmd_{col}" for col in cols if col not in keys})
    merged = pd.merge(local_small, pydmd_small, on=keys, how="inner")
    for name in ["lambda_real", "lambda_imag", "lambda_abs", "lambda_angle"]:
        merged[f"{name}_diff_pydmd_minus_local"] = merged[f"pydmd_{name}"] - merged[f"local_{name}"]

    path = out_dir / "eigenvalue_comparison_with_local.csv"
    merged.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def _merge_local_mode_comparison(
    out_dir: Path,
    x: np.ndarray,
    *,
    m_values: Iterable[int],
    n_values: Iterable[int],
    rank: int,
) -> Path:
    rows: list[dict[str, float]] = []
    for m in m_values:
        for n in n_values:
            train_length = m + n + 1
            if train_length > x.size:
                continue
            window = x[-train_length:]
            local = fit_hankel_dmd(window, m=m, n=n, rank=rank)
            model = _fit_pydmd_hankel(window, m=m, rank=rank)
            local_modes = np.asarray(local.modes, dtype=complex)
            pydmd_modes = np.asarray(model.modes, dtype=complex)
            local_eigs = np.asarray(local.eigenvalues, dtype=complex)
            pydmd_eigs = np.asarray(model.eigs, dtype=complex)

            unused = set(range(pydmd_eigs.size))
            for local_id, lam in enumerate(local_eigs):
                pydmd_id = min(unused, key=lambda idx: abs(pydmd_eigs[idx] - lam))
                unused.remove(pydmd_id)
                local_mode = local_modes[:, local_id]
                pydmd_mode = pydmd_modes[:, pydmd_id]
                scale = np.vdot(pydmd_mode, local_mode) / np.vdot(pydmd_mode, pydmd_mode)
                aligned = scale * pydmd_mode
                rows.append(
                    {
                        "m": m,
                        "n": n,
                        "rank": rank,
                        "train_observable_count": train_length,
                        "mode_id_local": local_id,
                        "mode_id_pydmd": pydmd_id,
                        "lambda_match_abs_diff": float(abs(pydmd_eigs[pydmd_id] - lam)),
                        "mode_relative_error_after_complex_scaling": float(
                            np.linalg.norm(local_mode - aligned) / np.linalg.norm(local_mode)
                        ),
                        "mode_abs_correlation": float(
                            abs(np.vdot(local_mode, pydmd_mode))
                            / (np.linalg.norm(local_mode) * np.linalg.norm(pydmd_mode))
                        ),
                        "scale_abs": float(abs(scale)),
                        "scale_angle": float(np.angle(scale)),
                    }
                )

    path = out_dir / "mode_comparison_with_local.csv"
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    return path


def run_grid(
    *,
    run_dir: Path,
    observable_file: str,
    out_dir: Path,
    m_values: Iterable[int],
    n_values: Iterable[int],
    rank: int,
    local_grid_path: Path | None,
) -> dict[str, Path]:
    observable_path = run_dir / observable_file
    x = _read_observable(observable_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, float]] = []
    lead_rows: list[dict[str, float]] = []
    fail_rows: list[dict[str, str]] = []

    for m in m_values:
        for n in n_values:
            train_length = m + n + 1
            try:
                if train_length >= x.size:
                    raise ValueError(
                        f"train window length {train_length} needs at least 1 out-of-window true point, "
                        f"but observable only has {x.size} points"
                    )
                eigenvalues, singular_values = _tail_spectrum(x, m=m, n=n, rank=rank)
                errors = _prediction_errors(x, m=m, n=n, rank=rank)
                energy = singular_values**2
                energy_at_rank = float(np.cumsum(energy)[rank - 1] / np.sum(energy))
                summary_rows.append(
                    {
                        "m": m,
                        "n": n,
                        "rank": rank,
                        "train_observable_count": train_length,
                        "total_observable_count": int(x.size),
                        "prediction_count": errors["prediction_count"],
                        "singular_value_1": float(singular_values[0]),
                        "singular_value_rank": float(singular_values[rank - 1]),
                        "energy_at_rank": energy_at_rank,
                        **errors,
                        "max_lambda_abs": float(eigenvalues["lambda_abs"].max()),
                        "min_lambda_abs": float(eigenvalues["lambda_abs"].min()),
                    }
                )
                leading = eigenvalues.sort_values("lambda_abs", ascending=False).head(3).reset_index(drop=True)
                for lead_id, row in enumerate(leading.itertuples(index=False), start=1):
                    angle = float(row.lambda_angle)
                    lead_rows.append(
                        {
                            "m": m,
                            "n": n,
                            "rank": rank,
                            "train_observable_count": train_length,
                            "leading_id": lead_id,
                            "mode_id": int(row.mode_id),
                            "lambda_real": float(row.lambda_real),
                            "lambda_imag": float(row.lambda_imag),
                            "lambda_abs": float(row.lambda_abs),
                            "lambda_angle": angle,
                            "period_events": float(2 * np.pi / abs(angle)) if abs(angle) > 1e-12 else np.inf,
                        }
                    )
            except Exception as exc:
                fail_rows.append({"m": m, "n": n, "rank": rank, "train_observable_count": train_length, "error": str(exc)})

    summary = pd.DataFrame(summary_rows).sort_values(["prediction_rmse", "prediction_mae"]).reset_index(drop=True)
    leading = pd.DataFrame(lead_rows)
    failures = pd.DataFrame(fail_rows)

    paths: dict[str, Path] = {
        "summary": out_dir / "pydmd_grid_summary.csv",
        "leading_eigenvalues": out_dir / "pydmd_grid_leading_eigenvalues.csv",
        "config": out_dir / "config.json",
    }
    summary.to_csv(paths["summary"], index=False, encoding="utf-8-sig")
    leading.to_csv(paths["leading_eigenvalues"], index=False, encoding="utf-8-sig")
    if not failures.empty:
        paths["failures"] = out_dir / "failures.csv"
        failures.to_csv(paths["failures"], index=False, encoding="utf-8-sig")

    if local_grid_path is not None:
        comparison = _merge_local_comparison(out_dir, local_grid_path, summary)
        if comparison is not None:
            paths["comparison"] = comparison
        eigen_comparison = _merge_local_eigenvalue_comparison(out_dir, local_grid_path, leading)
        if eigen_comparison is not None:
            paths["eigenvalue_comparison"] = eigen_comparison
        paths["mode_comparison"] = _merge_local_mode_comparison(
            out_dir,
            x,
            m_values=m_values,
            n_values=n_values,
            rank=rank,
        )

    paths["config"].write_text(
        json.dumps(
            {
                "observable_path": str(observable_path),
                "total_observable_count": int(x.size),
                "m_values": list(m_values),
                "n_values": list(n_values),
                "rank": rank,
                "package": "pydmd",
                "model": "HankelDMD",
                "pydmd_parameters": {"svd_rank": rank, "exact": False, "d": "m", "reconstruction_method": "first"},
                "spectrum_window_rule": "use_tail_m_plus_n_plus_1",
                "error_window_rule": "rolling train window of m+n+1 points, predict next out-of-window observable value",
                "prediction_rule": "modes @ diag(eigs) @ pinv(modes) @ last_high_order_snapshot; take last component",
                "outputs": {name: str(path.name) for name, path in paths.items() if name != "config"},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a PyDMD HankelDMD observable parameter grid.")
    parser.add_argument("--run-dir", required=True, help="Run directory containing observable.csv.")
    parser.add_argument("--observable-file", default="observable.csv")
    parser.add_argument("--out-dir", default=None, help="Default: run-dir/pydmd_hankel_grid_r{rank}.")
    parser.add_argument("--m-values", default="10,20,40,100")
    parser.add_argument("--n-values", default="10,15,20,40,100")
    parser.add_argument("--rank", type=int, default=5)
    parser.add_argument(
        "--local-grid-path",
        default=None,
        help="Local grid_summary.csv. If provided, comparison_with_local.csv is generated.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / f"pydmd_hankel_grid_r{args.rank}"
    local_grid_path = Path(args.local_grid_path) if args.local_grid_path else None
    paths = run_grid(
        run_dir=run_dir,
        observable_file=args.observable_file,
        out_dir=out_dir,
        m_values=_parse_int_list(args.m_values),
        n_values=_parse_int_list(args.n_values),
        rank=args.rank,
        local_grid_path=local_grid_path,
    )
    print(f"output_dir: {out_dir}")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
