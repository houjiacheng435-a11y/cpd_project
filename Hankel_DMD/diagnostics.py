"""single-observable Hankel-DMD 最小诊断。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Iterable

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


def reconstruction_error(x: Iterable[float], *, m: int, n: int, rank: int) -> float:
    """重新拟合 Hankel-DMD，并计算相对一步映射误差。"""

    result = fit_hankel_dmd(np.asarray(list(x), dtype=float), m=m, n=n, rank=rank, return_matrices=True)
    if result.X is None or result.Y is None or result.A_hat is None:
        raise ValueError("缺少 X/Y/A_hat，无法计算重构误差")

    W, _, _ = np.linalg.svd(result.X, full_matrices=False)
    W_r = W[:, : result.rank]
    y_pred = W_r @ result.A_hat @ W_r.conj().T @ result.X
    denom = np.linalg.norm(result.Y, ord="fro")
    if denom <= 0:
        raise ValueError("Y 的 Frobenius 范数为 0，无法计算相对误差")
    return float(np.linalg.norm(result.Y - y_pred, ord="fro") / denom)


def sample_size_stability(
    x: Iterable[float],
    *,
    sample_sizes: Iterable[int],
    m: int,
    n: int,
    rank: int,
    leading_count: int = 5,
) -> pd.DataFrame:
    """用不同末端样本长度重复拟合，记录 leading eigenvalues。"""

    arr = np.asarray(list(x), dtype=float)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("x 必须是一维非空序列")
    if not np.isfinite(arr).all():
        raise ValueError("x 不能包含 NaN 或 inf")

    rows: list[dict[str, Any]] = []
    for N in sample_sizes:
        N = int(N)
        if N <= 0 or N > arr.size:
            raise ValueError(f"非法样本长度 N={N}，有效范围为 1 到 {arr.size}")
        if m + n + 1 > N:
            raise ValueError(f"N={N} 样本不足：要求 m+n+1={m+n+1}")

        part = arr[-N:]
        result = fit_hankel_dmd(part, m=m, n=n, rank=rank)
        diag = eigenvalue_diagnostics(result.eigenvalues)
        diag = diag.sort_values("lambda_abs", ascending=False).head(leading_count)
        for lead_id, row in enumerate(diag.itertuples(index=False), start=1):
            rows.append(
                {
                    "N": N,
                    "m": m,
                    "n": n,
                    "rank": result.rank,
                    "leading_id": lead_id,
                    "mode_id": int(row.mode_id),
                    "lambda_real": float(row.lambda_real),
                    "lambda_imag": float(row.lambda_imag),
                    "lambda_abs": float(row.lambda_abs),
                    "lambda_angle": float(row.lambda_angle),
                }
            )
    return pd.DataFrame(rows)


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
    sample_sizes: Iterable[int] | None = None,
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
        rec_error = reconstruction_error(x, m=m, n=n, rank=rank)
        rec = pd.DataFrame([{"m": m, "n": n, "rank": rank, "relative_reconstruction_error": rec_error}])
        paths["reconstruction_error"] = output_dir / "reconstruction_error.csv"
        rec.to_csv(paths["reconstruction_error"], index=False, encoding="utf-8-sig")

        if sample_sizes is not None:
            stability = sample_size_stability(x, sample_sizes=sample_sizes, m=m, n=n, rank=rank)
            paths["sample_size_stability"] = output_dir / "sample_size_stability.csv"
            stability.to_csv(paths["sample_size_stability"], index=False, encoding="utf-8-sig")

    print(f"输出目录: {output_dir}")
    for name, path in paths.items():
        print(f"{name}: {path}")
    return paths


def _parse_sample_sizes(text: str | None) -> list[int] | None:
    if text is None or not text.strip():
        return None
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 single-observable Hankel-DMD 最小诊断。")
    parser.add_argument("--dmd-dir", required=True, help="Hankel-DMD 输出目录。")
    parser.add_argument("--out-dir", default=None, help="诊断结果输出目录；默认写到 dmd-dir/diagnostics。")
    parser.add_argument("--observable-path", default=None, help="observable.csv；提供后可计算重构误差和样本长度稳定性。")
    parser.add_argument("--m", type=int, default=None)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--sample-sizes", default=None, help="逗号分隔的样本长度，例如 80,100,120,147。")
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
        sample_sizes=_parse_sample_sizes(args.sample_sizes),
    )


if __name__ == "__main__":
    main()
