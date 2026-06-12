from __future__ import annotations

import argparse
import time
from pathlib import Path

import akshare as ak
import pandas as pd
from tqdm import tqdm


COLUMN_MAP = {
    "日期": "date",
    "股票代码": "symbol",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交额": "amount",
    "振幅": "amplitude",
    "涨跌幅": "pct_change",
    "涨跌额": "change",
    "换手率": "turnover",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download A-share daily OHLCV data with AKShare."
    )
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--start", default="20100101")
    parser.add_argument("--end", default="20260529")
    parser.add_argument(
        "--adjust",
        default="",
        choices=["", "qfq", "hfq"],
        help="'' means raw price, qfq means forward adjusted, hfq means backward adjusted.",
    )
    parser.add_argument(
        "--out-dir",
        default="data/a_share_1d_akshare",
        help="Output directory for per-symbol CSV files and summary files.",
    )
    parser.add_argument("--sleep", type=float, default=0.35)
    parser.add_argument("--retries", type=int, default=3)
    return parser.parse_args()


def normalize_history(df: pd.DataFrame, name: str) -> pd.DataFrame:
    df = df.rename(columns=COLUMN_MAP)
    df["name"] = name
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    ordered = [
        "date",
        "symbol",
        "name",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "amplitude",
        "pct_change",
        "change",
        "turnover",
    ]
    return df[[col for col in ordered if col in df.columns]]


def fetch_history(
    symbol: str,
    name: str,
    start: str,
    end: str,
    adjust: str,
    retries: int,
) -> pd.DataFrame:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            df = ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=start,
                end_date=end,
                adjust=adjust,
            )
            if df.empty:
                return df
            return normalize_history(df, name)
        except Exception as exc:  # AKShare may fail transiently under rate limits.
            last_error = exc
            time.sleep(min(2.0 * attempt, 6.0))
    raise RuntimeError(f"{symbol} failed after {retries} retries: {last_error}")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    symbol_dir = out_dir / "symbols"
    symbol_dir.mkdir(parents=True, exist_ok=True)

    stock_list = ak.stock_info_a_code_name()
    stock_list = stock_list[["code", "name"]].drop_duplicates("code")
    selected = stock_list.head(args.count).reset_index(drop=True)
    selected.to_csv(out_dir / "selected_symbols.csv", index=False, encoding="utf-8-sig")

    summary: list[dict[str, object]] = []
    frames: list[pd.DataFrame] = []

    for row in tqdm(selected.itertuples(index=False), total=len(selected)):
        symbol = str(row.code).zfill(6)
        name = str(row.name)
        status = "ok"
        error = ""
        rows = 0

        try:
            df = fetch_history(
                symbol=symbol,
                name=name,
                start=args.start,
                end=args.end,
                adjust=args.adjust,
                retries=args.retries,
            )
            rows = len(df)
            df.to_csv(symbol_dir / f"{symbol}.csv", index=False, encoding="utf-8-sig")
            if rows:
                frames.append(df)
        except Exception as exc:
            status = "failed"
            error = str(exc)

        summary.append(
            {
                "symbol": symbol,
                "name": name,
                "rows": rows,
                "status": status,
                "error": error,
            }
        )
        time.sleep(args.sleep)

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(out_dir / "download_summary.csv", index=False, encoding="utf-8-sig")

    if frames:
        all_df = pd.concat(frames, ignore_index=True)
        all_df.to_parquet(out_dir / "a_share_1d_200.parquet", index=False)
        all_df.to_csv(out_dir / "a_share_1d_200.csv", index=False, encoding="utf-8-sig")

    ok = int((summary_df["status"] == "ok").sum())
    failed = int((summary_df["status"] == "failed").sum())
    print(f"Downloaded {ok} symbols, failed {failed}. Output: {out_dir}")


if __name__ == "__main__":
    main()
