"""single-observable Hankel-DMD 最小诊断。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Hankel_DMD import fit_hankel_dmd


def singular_value_diagnostics(singular_values: Iterable[float]) -> pd.DataFrame:
    """计算奇异值、归一化奇异值和累计能量。"""

    s = np.asarray(list(singular_values), dtype=float)
    if s.ndim != 1 or s.size == 0:
        raise ValueError("singular_values 必须是一维非空序列")
    if not np.isfinite(s).all():
        raise ValueError("singular_values 不能包含 NaN 或 inf")
    if np.any(s < 0):
        raise ValueError("singular_values 不能为负")

    energy = s**2
    total_energy = float(energy.sum())
    if total_energy <= 0:
        raise ValueError("singular_values 的总能量必须大于 0")

    return pd.DataFrame(
        {
            "index": np.arange(s.size),
            "singular_value": s,
            "normalized_singular_value": s / s[0] if s[0] > 0 else np.nan,
            "energy_ratio": energy / total_energy,
            "cumulative_energy": np.cumsum(energy) / total_energy,
        }
    )


def eigenvalue_diagnostics(eigenvalues: Iterable[complex]) -> pd.DataFrame:
    """计算 DMD 特征值的实部、虚部、模长和相位角。"""

    lam = np.asarray(list(eigenvalues), dtype=complex)
    if lam.ndim != 1 or lam.size == 0:
        raise ValueError("eigenvalues 必须是一维非空序列")
    if not np.isfinite(lam.real).all() or not np.isfinite(lam.imag).all():
        raise ValueError("eigenvalues 不能包含 NaN 或 inf")

    return pd.DataFrame(
        {
            "mode_id": np.arange(lam.size),
            "lambda_real": lam.real,
            "lambda_imag": lam.imag,
            "lambda_abs": np.abs(lam),
            "lambda_angle": np.angle(lam),
        }
    )


def _validate_window_length(x: np.ndarray, *, m: int, n: int) -> int:
    required = m + n + 1
    if required > x.size:
        raise ValueError(f"样本不足：m+n+1={required}，但 observable 只有 {x.size} 个点")
    return required


def _predict_next_value_after_window(x: np.ndarray, *, m: int, n: int, rank: int) -> float:
    result = fit_hankel_dmd(x, m=m, n=n, rank=rank, return_matrices=True)
    if result.X is None or result.Y is None or result.A_hat is None:
        raise ValueError("缺少 X/Y/A_hat，无法计算新增点预测误差")

    W, _, _ = np.linalg.svd(result.X, full_matrices=False)
    W_r = W[:, : result.rank]
    a_full = W_r @ result.A_hat @ W_r.conj().T
    last_known_window = result.Y[:, -1]
    next_window_pred = a_full @ last_known_window
    return float(np.real(next_window_pred[-1]))


def reconstruction_errors(x: Iterable[float], *, m: int, n: int, rank: int) -> dict[str, float]:
    """滚动拟合 Hankel-DMD，并计算窗口外下一个 observable 点的预测误差。

    对每个长度为 ``m+n+1`` 的滚动窗口重新拟合一次 DMD，然后用该窗口内
    最后一个已知 delay 窗口预测下一个 delay 窗口。只取预测窗口最后一个
    分量，和窗口外真实的下一个 observable 标量比较。
    """

    arr = np.asarray(list(x), dtype=float)
    required = _validate_window_length(arr, m=m, n=n)
    if required >= arr.size:
        raise ValueError(f"样本不足：m+n+1={required} 时还需要至少 1 个窗口外真实点")

    true_values: list[float] = []
    pred_values: list[float] = []
    errors: list[float] = []

    for start in range(arr.size - required):
        window = arr[start : start + required]
        pred = _predict_next_value_after_window(window, m=m, n=n, rank=rank)
        true = float(arr[start + required])
        true_values.append(true)
        pred_values.append(pred)
        errors.append(pred - true)

    true_arr = np.asarray(true_values, dtype=float)
    pred_arr = np.asarray(pred_values, dtype=float)
    error_arr = np.asarray(errors, dtype=float)
    abs_error = np.abs(error_arr)
    midpoint = np.abs((pred_arr + true_arr) / 2.0)
    midpoint_relative_abs_error = np.full_like(abs_error, np.nan, dtype=float)
    valid_midpoint = midpoint > np.finfo(float).eps
    midpoint_relative_abs_error[valid_midpoint] = abs_error[valid_midpoint] / midpoint[valid_midpoint]

    return {
        "prediction_count": int(error_arr.size),
        "prediction_error_mean": float(np.mean(error_arr)),
        "prediction_error_std": float(np.std(error_arr, ddof=0)),
        "prediction_mae": float(np.mean(abs_error)),
        "prediction_rmse": float(np.sqrt(np.mean(error_arr**2))),
        "prediction_median_absolute_error": float(np.median(abs_error)),
        "midpoint_relative_abs_error_mean": float(np.nanmean(midpoint_relative_abs_error)),
        "midpoint_relative_abs_error_median": float(np.nanmedian(midpoint_relative_abs_error)),
        "midpoint_relative_abs_error_valid_count": int(np.sum(valid_midpoint)),
        "last_prediction_true": float(true_arr[-1]),
        "last_prediction_pred": float(pred_arr[-1]),
        "last_prediction_error": float(error_arr[-1]),
    }


def reconstruction_error(x: Iterable[float], *, m: int, n: int, rank: int) -> float:
    """兼容旧接口：返回窗口外下一个 observable 点的 RMSE。"""

    return reconstruction_errors(x, m=m, n=n, rank=rank)["prediction_rmse"]


def _read_eigenvalues(path: Path) -> np.ndarray:
    df = pd.read_csv(path)
    required = {"lambda_real", "lambda_imag"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} 缺少字段: {sorted(missing)}")
    return df["lambda_real"].to_numpy(float) + 1j * df["lambda_imag"].to_numpy(float)


def _read_singular_values(path: Path) -> np.ndarray:
    df = pd.read_csv(path)
    if "singular_value" not in df.columns:
        raise ValueError(f"{path} 缺少 singular_value 字段")
    return df["singular_value"].to_numpy(float)


def _read_observable(path: Path) -> np.ndarray:
    df = pd.read_csv(path)
    if "observable" not in df.columns:
        raise ValueError(f"{path} 缺少 observable 字段")
    return pd.to_numeric(df["observable"], errors="coerce").to_numpy(float)


def run_diagnostics(
    *,
    dmd_dir: str | Path,
    out_dir: str | Path | None = None,
    observable_path: str | Path | None = None,
    m: int | None = None,
    n: int | None = None,
    rank: int | None = None,
) -> dict[str, Path]:
    """读取 Hankel-DMD 输出，保存最小诊断结果。"""

    dmd_dir = Path(dmd_dir)
    output_dir = Path(out_dir) if out_dir is not None else dmd_dir / "diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)

    eigen_path = dmd_dir / "eigenvalue_summary.csv"
    if not eigen_path.exists():
        eigen_path = dmd_dir / "eigenvalues.csv"
    singular_path = dmd_dir / "singular_values.csv"
    if not eigen_path.exists():
        raise FileNotFoundError(f"找不到 eigenvalue 文件: {eigen_path}")
    if not singular_path.exists():
        raise FileNotFoundError(f"找不到 singular_values 文件: {singular_path}")

    paths: dict[str, Path] = {}
    sv_diag = singular_value_diagnostics(_read_singular_values(singular_path))
    eig_diag = eigenvalue_diagnostics(_read_eigenvalues(eigen_path))

    paths["singular_value_diagnostics"] = output_dir / "singular_value_diagnostics.csv"
    paths["eigenvalue_diagnostics"] = output_dir / "eigenvalue_diagnostics.csv"
    sv_diag.to_csv(paths["singular_value_diagnostics"], index=False, encoding="utf-8-sig")
    eig_diag.to_csv(paths["eigenvalue_diagnostics"], index=False, encoding="utf-8-sig")

    if observable_path is not None and m is not None and n is not None and rank is not None:
        x = _read_observable(Path(observable_path))
        errors = reconstruction_errors(x, m=m, n=n, rank=rank)
        rec = pd.DataFrame(
            [
                {
                    "m": m,
                    "n": n,
                    "rank": rank,
                    "train_observable_count": m + n + 1,
                    "window_rule": "rolling_train_window_predict_next_value",
                    "prediction_count": errors["prediction_count"],
                    "prediction_error_mean": errors["prediction_error_mean"],
                    "prediction_error_std": errors["prediction_error_std"],
                    "prediction_mae": errors["prediction_mae"],
                    "prediction_rmse": errors["prediction_rmse"],
                    "prediction_median_absolute_error": errors["prediction_median_absolute_error"],
                    "midpoint_relative_abs_error_mean": errors["midpoint_relative_abs_error_mean"],
                    "midpoint_relative_abs_error_median": errors["midpoint_relative_abs_error_median"],
                    "midpoint_relative_abs_error_valid_count": errors["midpoint_relative_abs_error_valid_count"],
                    "last_prediction_true": errors["last_prediction_true"],
                    "last_prediction_pred": errors["last_prediction_pred"],
                    "last_prediction_error": errors["last_prediction_error"],
                }
            ]
        )
        paths["reconstruction_error"] = output_dir / "reconstruction_error.csv"
        rec.to_csv(paths["reconstruction_error"], index=False, encoding="utf-8-sig")

    print(f"输出目录: {output_dir}")
    for name, path in paths.items():
        print(f"{name}: {path}")
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 single-observable Hankel-DMD 最小诊断。")
    parser.add_argument("--dmd-dir", required=True, help="Hankel-DMD 输出目录。")
    parser.add_argument("--out-dir", default=None, help="诊断结果输出目录；默认写到 dmd-dir/diagnostics。")
    parser.add_argument("--observable-path", default=None, help="observable.csv；提供后可计算新增 observable 点预测误差。")
    parser.add_argument("--m", type=int, default=None)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--rank", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_diagnostics(
        dmd_dir=args.dmd_dir,
        out_dir=args.out_dir,
        observable_path=args.observable_path,
        m=args.m,
        n=args.n,
        rank=args.rank,
    )


if __name__ == "__main__":
    main()
