"""读取 observable.csv，执行 Hankel-DMD，并保存核心结果。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Hankel_DMD import fit_hankel_dmd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="对 observable.csv 执行 Hankel-DMD 并保存结果。")
    parser.add_argument("--run-dir", required=True, help="包含 observable.csv 的 run 目录。")
    parser.add_argument("--observable-file", default="observable.csv")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="输出目录；默认写到 run-dir/hankel_dmd_m{m}_n{n}_r{rank}。",
    )
    parser.add_argument("--m", type=int, required=True, help="Hankel 矩阵行数。")
    parser.add_argument("--n", type=int, required=True, help="Hankel 矩阵列数减一，实际列数为 n+1。")
    parser.add_argument("--rank", type=int, required=True, help="SVD 截断 rank。")
    parser.add_argument("--save-matrices", action="store_true", help="是否保存 X/Y/A_hat 矩阵。")
    return parser.parse_args()


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _complex_frame(values: np.ndarray, prefix: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            f"{prefix}_real": np.real(values),
            f"{prefix}_imag": np.imag(values),
            f"{prefix}_abs": np.abs(values),
            f"{prefix}_angle": np.angle(values),
        }
    )


def _tail_hankel_window(x: np.ndarray, *, m: int, n: int) -> np.ndarray:
    required = m + n + 1
    if required > x.size:
        raise ValueError(f"样本不足：m+n+1={required}，但 observable 只有 {x.size} 个点")
    return x[-required:]


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    observable_path = run_dir / args.observable_file
    if not observable_path.is_file():
        raise FileNotFoundError(f"找不到 observable 文件: {observable_path}")

    default_out_name = f"hankel_dmd_m{args.m}_n{args.n}_r{args.rank}"
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / default_out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    observable = pd.read_csv(observable_path)
    if "observable" not in observable.columns:
        raise ValueError(f"{observable_path} 缺少 observable 列")
    sort_cols = [col for col in ["date", "bar_index", "event_id"] if col in observable.columns]
    if sort_cols:
        observable = observable.sort_values(sort_cols).reset_index(drop=True)
    x = pd.to_numeric(observable["observable"], errors="coerce").to_numpy(float)
    x_used = _tail_hankel_window(x, m=args.m, n=args.n)

    result = fit_hankel_dmd(
        x_used,
        m=args.m,
        n=args.n,
        rank=args.rank,
        return_matrices=args.save_matrices,
    )

    eigenvalues = _complex_frame(result.eigenvalues, "lambda")
    eigenvalues.insert(0, "mode_index", np.arange(len(eigenvalues)))
    eigenvalues.to_csv(out_dir / "eigenvalues.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame(
        {
            "index": np.arange(len(result.singular_values)),
            "singular_value": result.singular_values,
        }
    ).to_csv(out_dir / "singular_values.csv", index=False, encoding="utf-8-sig")

    modes_real = pd.DataFrame(np.real(result.modes))
    modes_imag = pd.DataFrame(np.imag(result.modes))
    modes_real.columns = [f"mode_{i}" for i in range(modes_real.shape[1])]
    modes_imag.columns = [f"mode_{i}" for i in range(modes_imag.shape[1])]
    modes_real.insert(0, "delay_index", np.arange(modes_real.shape[0]))
    modes_imag.insert(0, "delay_index", np.arange(modes_imag.shape[0]))
    modes_real.to_csv(out_dir / "modes_real.csv", index=False, encoding="utf-8-sig")
    modes_imag.to_csv(out_dir / "modes_imag.csv", index=False, encoding="utf-8-sig")

    if args.save_matrices:
        pd.DataFrame(result.X).to_csv(out_dir / "X.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(result.Y).to_csv(out_dir / "Y.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(np.real(result.A_hat)).to_csv(out_dir / "A_hat_real.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(np.imag(result.A_hat)).to_csv(out_dir / "A_hat_imag.csv", index=False, encoding="utf-8-sig")

    _write_json(
        out_dir / "config.json",
        {
            "observable_path": str(observable_path),
            "m": args.m,
            "n": args.n,
            "rank": result.rank,
            "total_observable_count": int(x.size),
            "used_observable_count": int(x_used.size),
            "used_window": "tail_m_plus_n_plus_1",
            "save_matrices": bool(args.save_matrices),
            "outputs": {
                "eigenvalues": "eigenvalues.csv",
                "singular_values": "singular_values.csv",
                "modes_real": "modes_real.csv",
                "modes_imag": "modes_imag.csv",
            },
        },
    )

    print(f"输出目录: {out_dir}")
    print(f"特征值: {out_dir / 'eigenvalues.csv'}")
    print(f"奇异值: {out_dir / 'singular_values.csv'}")
    print(f"modes: {out_dir / 'modes_real.csv'} / {out_dir / 'modes_imag.csv'}")


if __name__ == "__main__":
    main()
