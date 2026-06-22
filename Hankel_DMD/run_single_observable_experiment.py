"""运行 single-observable Hankel-DMD 最小实验。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Hankel_DMD import fit_hankel_dmd


def _read_observable(data: pd.DataFrame | str | Path) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()
    path = Path(data)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"只支持 CSV observable 文件: {path}")


def _extract_sequence(df: pd.DataFrame) -> np.ndarray:
    if "observable" not in df.columns:
        raise ValueError("observable 数据必须包含 observable 列")

    sort_cols = [col for col in ["date", "bar_index", "event_id"] if col in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    x = pd.to_numeric(df["observable"], errors="coerce").to_numpy(float)
    if x.ndim != 1:
        raise ValueError("observable 必须是一维序列")
    if len(x) == 0:
        raise ValueError("observable 序列为空")
    if not np.isfinite(x).all():
        raise ValueError("observable 序列不能包含 NaN 或 inf")
    return x


def _eigenvalue_summary(eigenvalues: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "mode_id": np.arange(len(eigenvalues)),
            "lambda_real": np.real(eigenvalues),
            "lambda_imag": np.imag(eigenvalues),
            "lambda_abs": np.abs(eigenvalues),
            "lambda_angle": np.angle(eigenvalues),
        }
    )


def run_single_observable_experiment(
    observable_data: pd.DataFrame | str | Path,
    *,
    m: int = 20,
    n: int = 80,
    rank: int = 5,
    out_dir: str | Path | None = None,
    save_modes: bool = True,
) -> dict[str, Any]:
    """读取 single observable，运行 Hankel-DMD，并保存谱结果。"""

    df = _read_observable(observable_data)
    x = _extract_sequence(df)

    result = fit_hankel_dmd(x, m=m, n=n, rank=rank, return_matrices=True)
    eigenvalues = _eigenvalue_summary(result.eigenvalues)
    singular_values = pd.DataFrame(
        {
            "index": np.arange(len(result.singular_values)),
            "singular_value": result.singular_values,
        }
    )

    paths: dict[str, Path] = {}
    if out_dir is not None:
        output_dir = Path(out_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        paths["eigenvalue_summary"] = output_dir / "eigenvalue_summary.csv"
        paths["singular_values"] = output_dir / "singular_values.csv"
        eigenvalues.to_csv(paths["eigenvalue_summary"], index=False, encoding="utf-8-sig")
        singular_values.to_csv(paths["singular_values"], index=False, encoding="utf-8-sig")

        if save_modes:
            modes_real = pd.DataFrame(np.real(result.modes))
            modes_imag = pd.DataFrame(np.imag(result.modes))
            modes_real.columns = [f"mode_{i}" for i in range(modes_real.shape[1])]
            modes_imag.columns = [f"mode_{i}" for i in range(modes_imag.shape[1])]
            modes_real.insert(0, "delay_index", np.arange(modes_real.shape[0]))
            modes_imag.insert(0, "delay_index", np.arange(modes_imag.shape[0]))
            paths["modes_real"] = output_dir / "modes_real.csv"
            paths["modes_imag"] = output_dir / "modes_imag.csv"
            modes_real.to_csv(paths["modes_real"], index=False, encoding="utf-8-sig")
            modes_imag.to_csv(paths["modes_imag"], index=False, encoding="utf-8-sig")

    print(f"N: {len(x)}")
    print(f"m, n, rank: {m}, {n}, {result.rank}")
    print(f"X shape: {result.X.shape}")
    print(f"Y shape: {result.Y.shape}")
    print(f"eigenvalue count: {len(result.eigenvalues)}")
    print("singular values head:")
    print(singular_values.head().to_string(index=False))
    print("eigenvalues:")
    print(eigenvalues.to_string(index=False))
    if paths:
        print("outputs:")
        for name, path in paths.items():
            print(f"{name}: {path}")

    return {
        "result": result,
        "eigenvalue_summary": eigenvalues,
        "singular_values": singular_values,
        "paths": paths,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 single-observable Hankel-DMD 最小实验。")
    parser.add_argument("--observable-path", required=True, help="第 1 步输出的 observable.csv。")
    parser.add_argument("--out-dir", required=True, help="实验结果输出目录。")
    parser.add_argument("--m", type=int, default=20)
    parser.add_argument("--n", type=int, default=80)
    parser.add_argument("--rank", type=int, default=5)
    parser.add_argument("--no-save-modes", action="store_true", help="只保存 eigenvalues 和 singular values。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_single_observable_experiment(
        args.observable_path,
        m=args.m,
        n=args.n,
        rank=args.rank,
        out_dir=args.out_dir,
        save_modes=not args.no_save_modes,
    )


if __name__ == "__main__":
    main()
