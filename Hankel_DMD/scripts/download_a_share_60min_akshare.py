"""Download A-share 60-minute OHLC data with AkShare.

Some AkShare minute endpoints can fail behind local proxies with SSL record
layer errors. This script adds common quote hosts to NO_PROXY before importing
AkShare, then writes only per-symbol CSV files.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import pandas as pd

NO_PROXY_HOSTS = ("push2his.eastmoney.com", "quotes.sina.cn", "finance.sina.com.cn")


def _ensure_no_proxy() -> None:
    for key in ("NO_PROXY", "no_proxy"):
        value = os.environ.get(key, "")
        hosts = [item.strip() for item in value.split(",") if item.strip()]
        for host in NO_PROXY_HOSTS:
            if host not in hosts:
                hosts.append(host)
        os.environ[key] = ",".join(hosts)


_ensure_no_proxy()

import akshare as ak  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download A-share 60-minute OHLC data.")
    parser.add_argument(
        "--symbols",
        default="000001,000002,000006,000008,000009",
        help="Comma-separated stock symbols, for example 000001,000002.",
    )
    parser.add_argument("--start", default="2020-01-01 09:30:00")
    parser.add_argument("--end", default="2026-05-29 15:00:00")
    parser.add_argument("--adjust", default="", choices=["", "qfq", "hfq"])
    parser.add_argument(
        "--source",
        default="sina",
        choices=["sina", "eastmoney"],
        help="sina usually returns about 1970 recent minute bars; eastmoney often returns a shorter recent window.",
    )
    parser.add_argument("--out-dir", default="data/a_share_60min_akshare")
    parser.add_argument("--sleep", type=float, default=0.5)
    parser.add_argument("--retries", type=int, default=4)
    return parser.parse_args()


def _parse_symbols(text: str) -> list[str]:
    symbols = [item.strip().zfill(6) for item in text.split(",") if item.strip()]
    if not symbols:
        raise ValueError("At least one symbol is required")
    return symbols


def _empty_history() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "date",
            "symbol",
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
    )


def _filter_datetime(df: pd.DataFrame, *, start: str, end: str) -> pd.DataFrame:
    if df.empty:
        return df
    dt = pd.to_datetime(df["date"])
    mask = (dt >= pd.Timestamp(start)) & (dt <= pd.Timestamp(end))
    return df.loc[mask].reset_index(drop=True)


def _normalise_eastmoney_history(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if df.empty:
        return _empty_history()

    if df.shape[1] < 11:
        raise ValueError(f"{symbol} returned too few columns: {df.columns.tolist()}")

    work = df.iloc[:, :11].copy()
    work.columns = [
        "date",
        "open",
        "close",
        "high",
        "low",
        "pct_change",
        "change",
        "volume",
        "amount",
        "amplitude",
        "turnover",
    ]
    work.insert(1, "symbol", symbol)
    work["date"] = pd.to_datetime(work["date"])
    for col in ["open", "high", "low", "close", "volume", "amount", "amplitude", "pct_change", "change", "turnover"]:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.dropna(subset=["date", "open", "high", "low", "close"])
    work = work.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    work["date"] = work["date"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return work[_empty_history().columns]


def _normalise_sina_history(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if df.empty:
        return _empty_history()

    required = {"day", "open", "high", "low", "close", "volume", "amount"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"{symbol} sina response missing columns: {missing}")

    work = df[["day", "open", "high", "low", "close", "volume", "amount"]].copy()
    work = work.rename(columns={"day": "date"})
    work.insert(1, "symbol", symbol)
    work["date"] = pd.to_datetime(work["date"])
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.dropna(subset=["date", "open", "high", "low", "close"])
    work = work.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    work["amplitude"] = pd.NA
    work["pct_change"] = pd.NA
    work["change"] = pd.NA
    work["turnover"] = pd.NA
    work["date"] = work["date"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return work[_empty_history().columns]


def _sina_symbol(symbol: str) -> str:
    if symbol.startswith(("0", "3")):
        return f"sz{symbol}"
    if symbol.startswith("6"):
        return f"sh{symbol}"
    if symbol.startswith(("4", "8")):
        return f"bj{symbol}"
    return symbol


def fetch_60min_history(
    symbol: str,
    *,
    start: str,
    end: str,
    adjust: str,
    source: str,
    retries: int,
) -> pd.DataFrame:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            if source == "eastmoney":
                raw = ak.stock_zh_a_hist_min_em(
                    symbol=symbol,
                    start_date=start,
                    end_date=end,
                    period="60",
                    adjust=adjust,
                )
                return _filter_datetime(_normalise_eastmoney_history(raw, symbol), start=start, end=end)
            if source == "sina":
                raw = ak.stock_zh_a_minute(symbol=_sina_symbol(symbol), period="60", adjust=adjust)
                return _filter_datetime(_normalise_sina_history(raw, symbol), start=start, end=end)
            raise ValueError(f"Unsupported data source: {source}")
        except Exception as exc:
            last_error = exc
            time.sleep(min(2.0 * attempt, 8.0))
            _ensure_no_proxy()
    raise RuntimeError(f"{symbol} download failed after {retries} retries: {last_error}")


def check_completeness(frames: list[pd.DataFrame]) -> tuple[int, int]:
    if not frames:
        return 0, 0

    panel = pd.concat(frames, ignore_index=True)
    panel["symbol"] = panel["symbol"].astype(str).str.zfill(6)
    panel["date"] = pd.to_datetime(panel["date"])
    common_calendar = pd.Index(sorted(panel["date"].drop_duplicates()))

    total = 0
    complete = 0
    for symbol, group in panel.groupby("symbol", sort=True):
        dates = pd.Index(sorted(group["date"].drop_duplicates()))
        span_calendar = common_calendar[(common_calendar >= dates.min()) & (common_calendar <= dates.max())]
        missing = span_calendar.difference(dates)
        duplicated = int(group["date"].duplicated().sum())
        bars_per_day = group.assign(day=group["date"].dt.date).groupby("day").size()
        is_complete = len(missing) == 0 and duplicated == 0
        total += 1
        complete += int(is_complete)
        print(
            f"{symbol}: completeness={int(is_complete)}, "
            f"range={dates.min().strftime('%Y-%m-%d %H:%M:%S')}..{dates.max().strftime('%Y-%m-%d %H:%M:%S')}, "
            f"rows={len(group)}, trading_days={bars_per_day.size}, "
            f"bars_per_day={int(bars_per_day.min()) if not bars_per_day.empty else 0}.."
            f"{int(bars_per_day.max()) if not bars_per_day.empty else 0}, "
            f"missing_common={len(missing)}, duplicates={duplicated}"
        )
    return complete, total


def main() -> None:
    args = parse_args()
    symbols = _parse_symbols(args.symbols)
    out_dir = Path(args.out_dir)
    symbol_dir = out_dir / "symbols"
    symbol_dir.mkdir(parents=True, exist_ok=True)

    ok = 0
    frames: list[pd.DataFrame] = []

    for symbol in symbols:
        status = "ok"
        error = ""
        rows = 0
        try:
            df = fetch_60min_history(
                symbol,
                start=args.start,
                end=args.end,
                adjust=args.adjust,
                source=args.source,
                retries=args.retries,
            )
            rows = len(df)
            df.to_csv(symbol_dir / f"{symbol}.csv", index=False, encoding="utf-8-sig")
            ok += 1
            if not df.empty:
                frames.append(df)
        except Exception as exc:
            status = "failed"
            error = str(exc)

        print(f"{symbol}: {status}, rows={rows}")
        if error:
            print(f"{symbol}: error={error}")
        time.sleep(args.sleep)

    complete, checked = check_completeness(frames)
    print(f"downloaded {ok}/{len(symbols)} symbols; completeness passed {complete}/{checked}; out_dir={out_dir}")


if __name__ == "__main__":
    main()
