"""从原始日频数据重新生成 CPD 诊断和 automatic123 极值点结果。"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cpd.market_state_vector import MarketStateVector


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


@dataclass
class RunConfig:
    input_csv: str
    out_root: str
    run_name: str
    start: str | None
    end: str | None
    log_price: bool
    detector_method: str
    detector_q: float
    cpd_confirm_lag: int
    reset_structure_on_cpd: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="重新从日频价格生成 CPD 和 automatic123 极值点结果。")
    parser.add_argument("--config", default=None, help="JSON 配置文件路径。")
    parser.add_argument("--input-csv", default=None, help="单只股票的日频 CSV 文件。")
    parser.add_argument("--out-root", default="Hankel_DMD/outputs/runs", help="统一输出根目录。")
    parser.add_argument("--run-name", default=None, help="本次运行名称；不填则使用时间戳。")
    parser.add_argument("--start", default=None, help="开始日期，例如 2020-01-01。")
    parser.add_argument("--end", default=None, help="结束日期，例如 2026-05-29。")
    parser.add_argument("--log-price", action="store_true", help="从 CPD 开始使用 log OHLC。")
    parser.add_argument("--detector-method", default="cusum")
    parser.add_argument("--detector-q", type=float, default=1.0)
    parser.add_argument("--cpd-confirm-lag", type=int, default=0)
    parser.add_argument("--reset-structure-on-cpd", action="store_true")
    return parser.parse_args()


def _normalise_ohlc(path: Path, *, log_price: bool, start: str | None, end: str | None) -> pd.DataFrame:
    df = pd.read_csv(path)
    rename = {
        "date": "Date",
        "datetime": "Date",
        "timestamp": "Date",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
        "amount": "Amount",
    }
    df = df.rename(columns={col: rename.get(col, col) for col in df.columns})
    if "Date" not in df.columns:
        raise ValueError(f"{path} 缺少 date/Date 字段")
    required = {"Open", "High", "Low", "Close"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} 缺少 OHLC 字段: {sorted(missing)}")

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").set_index("Date")
    for col in ["Open", "High", "Low", "Close", "Volume", "Amount"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Open", "High", "Low", "Close"])

    if start is not None:
        df = df[df.index >= pd.Timestamp(start)]
    if end is not None:
        df = df[df.index <= pd.Timestamp(end)]
    if log_price:
        price_cols = ["Open", "High", "Low", "Close"]
        if (df[price_cols] <= 0).any().any():
            raise ValueError(f"{path} 含有非正价格，不能取 log")
        df.loc[:, price_cols] = np.log(df[price_cols])
    return df


def _build_one_symbol(
    path: Path,
    *,
    config: RunConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stock_id = path.stem.zfill(6)
    ohlc = _normalise_ohlc(path, log_price=config.log_price, start=config.start, end=config.end)
    if ohlc.empty:
        empty_diag = pd.DataFrame()
        empty_ext = pd.DataFrame()
        return empty_diag, empty_ext

    builder = MarketStateVector(
        detector_method=config.detector_method,
        detector_q=config.detector_q,
        cpd_confirm_lag=config.cpd_confirm_lag,
        reset_structure_on_cpd=config.reset_structure_on_cpd,
    )
    state = builder.build_market_state_vector(data=ohlc[["Open", "High", "Low", "Close"]])
    state = state.reset_index().rename(columns={"timestamp": "date"})
    state.insert(0, "stock_id", stock_id)

    diagnostics_cols = [
        "stock_id",
        "date",
        "Open",
        "High",
        "Low",
        "Close",
        "direction",
        "excep",
        "status",
        "new_extremum_confirmed",
        "confirmed_extremum_kind",
        "confirmed_extremum_idx",
        "confirmed_extremum_value",
        "confirmed_extremum_confirmed_at",
        "cpd_confirm_event",
        "time_since_last_cpd",
        "time_since_last_cpd_confirm",
    ]
    diagnostics = state[[col for col in diagnostics_cols if col in state.columns]].copy()

    dates = pd.DatetimeIndex(ohlc.index)
    extrema_rows: list[dict[str, Any]] = []
    confirmed = state[state["new_extremum_confirmed"].fillna(False)].copy()
    for seq, row in enumerate(confirmed.itertuples(index=False), start=1):
        kind = getattr(row, "confirmed_extremum_kind", None)
        idx_value = getattr(row, "confirmed_extremum_idx", np.nan)
        price = getattr(row, "confirmed_extremum_value", np.nan)
        if kind not in {"high", "low"} or pd.isna(idx_value) or pd.isna(price):
            continue
        idx = int(idx_value)
        if idx < 0 or idx >= len(dates):
            continue
        extrema_rows.append(
            {
                "event_id": f"{stock_id}_{seq:06d}",
                "stock_id": stock_id,
                "extreme_type": str(kind),
                "extreme_idx": idx,
                "extreme_date": pd.Timestamp(dates[idx]),
                "extreme_price": float(price),
                "confirmed_idx": int(ohlc.index.get_loc(pd.Timestamp(getattr(row, "date")))),
                "confirmed_date": pd.Timestamp(getattr(row, "date")),
            }
        )
    extrema = pd.DataFrame(extrema_rows)
    if not extrema.empty:
        dedup_key = ["stock_id", "extreme_type", "extreme_idx", "extreme_date", "extreme_price"]
        extrema = (
            extrema.sort_values(["confirmed_idx", "confirmed_date", "event_id"])
            .drop_duplicates(subset=dedup_key, keep="first")
            .sort_values(["extreme_idx", "confirmed_idx"])
            .reset_index(drop=True)
        )
        extrema["event_id"] = [f"{stock_id}_{i:06d}" for i in range(1, len(extrema) + 1)]
    return diagnostics, extrema


def main() -> None:
    args = parse_args()
    file_config: dict[str, Any] = {}
    if args.config:
        file_config = json.loads(Path(args.config).read_text(encoding="utf-8"))

    input_csv = file_config.get("input_csv", args.input_csv)
    out_root = file_config.get("out_root", args.out_root)
    start = file_config.get("start", args.start)
    end = file_config.get("end", args.end)
    log_price = bool(file_config.get("log_price", args.log_price))
    detector_method = file_config.get("detector_method", args.detector_method)
    detector_q = float(file_config.get("detector_q", args.detector_q))
    cpd_confirm_lag = int(file_config.get("cpd_confirm_lag", args.cpd_confirm_lag))
    reset_structure_on_cpd = bool(file_config.get("reset_structure_on_cpd", args.reset_structure_on_cpd))
    run_name_config = file_config.get("run_name", None)

    if not input_csv:
        raise ValueError("必须通过配置文件或 --input-csv 指定一个具体 CSV 文件")

    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S_cpd_extrema")
    if args.run_name is None and run_name_config:
        run_name = str(run_name_config)
    config = RunConfig(
        input_csv=input_csv,
        out_root=out_root,
        run_name=run_name,
        start=start,
        end=end,
        log_price=log_price,
        detector_method=detector_method,
        detector_q=detector_q,
        cpd_confirm_lag=cpd_confirm_lag,
        reset_structure_on_cpd=reset_structure_on_cpd,
    )

    input_path = Path(config.input_csv)
    if not input_path.is_file():
        raise FileNotFoundError(f"输入 CSV 不存在: {input_path}")

    out_dir = Path(config.out_root) / config.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "config.json", asdict(config))

    diagnostics, extrema = _build_one_symbol(input_path, config=config)

    diagnostics.to_csv(out_dir / "cpd_diagnostics.csv", index=False, encoding="utf-8-sig")
    extrema.to_csv(out_dir / "confirmed_extrema.csv", index=False, encoding="utf-8-sig")

    print(f"输出目录: {out_dir}")
    print(f"CPD 诊断: {out_dir / 'cpd_diagnostics.csv'}")
    print(f"极值事件: {out_dir / 'confirmed_extrema.csv'}")


if __name__ == "__main__":
    main()
