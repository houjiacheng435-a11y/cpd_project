"""从 automatic123 极值点构造单变量 event-time observable。

本模块只消费已有的 automatic123 极值点结果，不重新检测极值，也不写出文件。
输出是 Hankel-DMD 实验所需的最小事件时间序列。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd


PriceMode = Literal["auto", "raw", "log"]


def _read_table(data: pd.DataFrame | str | Path) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()
    path = Path(data)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"不支持的文件类型: {path}")


def _normalise_extreme_type(value: Any) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip().lower()
    if text in {"h", "high", "max", "top"}:
        return "high"
    if text in {"l", "low", "min", "bottom"}:
        return "low"
    return None


def _infer_price_mode(price: pd.Series) -> Literal["raw", "log"]:
    values = pd.to_numeric(price, errors="coerce").dropna()
    if values.empty:
        return "raw"
    # 原始价格也可能很小，所以调用方可以显式覆盖这个自动判断。
    return "log" if values.gt(0).all() and values.quantile(0.95) < 20 else "raw"


def _log_level(values: pd.Series, mode: Literal["raw", "log"]) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if mode == "log":
        return numeric
    out = pd.Series(np.nan, index=numeric.index, dtype=float)
    mask = numeric > 0
    out.loc[mask] = np.log(numeric.loc[mask])
    return out


def _extract_extrema_from_state(
    state: pd.DataFrame,
    *,
    symbol_col: str,
    date_col: str,
    new_extremum_col: str,
    extreme_type_col: str,
    extreme_price_col: str,
    extreme_index_col: str,
) -> pd.DataFrame:
    required = {
        symbol_col,
        date_col,
        new_extremum_col,
        extreme_type_col,
        extreme_price_col,
    }
    missing = sorted(required.difference(state.columns))
    if missing:
        raise ValueError(f"缺少 automatic123 极值点字段: {missing}")

    src = state[state[new_extremum_col].fillna(False)].copy()
    if src.empty:
        return pd.DataFrame(
            columns=["symbol", "confirm_date", "event_date", "bar_index", "extreme_type", "extreme_price"]
        )

    src["symbol"] = src[symbol_col].astype(str).str.zfill(6)
    src["confirm_date"] = pd.to_datetime(src[date_col])
    src["event_date"] = src["confirm_date"]
    src["bar_index"] = pd.to_numeric(src.get(extreme_index_col), errors="coerce")
    src["extreme_type"] = src[extreme_type_col].map(_normalise_extreme_type)
    src["extreme_price"] = pd.to_numeric(src[extreme_price_col], errors="coerce")

    if extreme_index_col in src.columns:
        date_lookup = {
            symbol: grp.sort_values(date_col)[date_col].reset_index(drop=True)
            for symbol, grp in state.assign(**{date_col: pd.to_datetime(state[date_col])}).groupby(
                state[symbol_col].astype(str).str.zfill(6), sort=False
            )
        }
        event_dates: list[pd.Timestamp] = []
        for row in src.itertuples(index=False):
            symbol = getattr(row, "symbol")
            idx = getattr(row, "bar_index")
            event_date = getattr(row, "confirm_date")
            if symbol in date_lookup and pd.notna(idx):
                idx_int = int(idx)
                dates = date_lookup[symbol]
                if 0 <= idx_int < len(dates):
                    event_date = pd.Timestamp(dates.iloc[idx_int])
            event_dates.append(pd.Timestamp(event_date))
        src["event_date"] = event_dates

    keep = ["symbol", "confirm_date", "event_date", "bar_index", "extreme_type", "extreme_price"]
    return (
        src[keep]
        .dropna(subset=["symbol", "event_date", "extreme_type", "extreme_price"])
        .sort_values(["symbol", "event_date", "confirm_date"])
        .reset_index(drop=True)
    )


def _normalise_extrema_table(extrema: pd.DataFrame) -> pd.DataFrame:
    candidates = {
        "symbol": ["symbol", "stock_id"],
        "confirm_date": ["extrema_time", "confirm_date", "date"],
        "event_date": ["extreme_date", "extrema_event_time", "event_date", "date"],
        "extreme_type": ["extrema_type", "extreme_type", "confirmed_extremum_kind"],
        "extreme_price": ["extrema_price", "extreme_price", "confirmed_extremum_value"],
        "bar_index": ["extreme_idx", "bar_index", "confirmed_extremum_idx", "extrema_index"],
    }

    def pick(name: str, required: bool = True) -> str | None:
        for col in candidates[name]:
            if col in extrema.columns:
                return col
        if required:
            raise ValueError(f"找不到字段: {name}")
        return None

    symbol_col = pick("symbol")
    confirm_col = pick("confirm_date", required=False) or pick("event_date")
    event_col = pick("event_date")
    type_col = pick("extreme_type")
    price_col = pick("extreme_price")
    index_col = pick("bar_index", required=False)

    out = pd.DataFrame(
        {
            "symbol": extrema[symbol_col].astype(str).str.zfill(6),
            "confirm_date": pd.to_datetime(extrema[confirm_col]),
            "event_date": pd.to_datetime(extrema[event_col]),
            "extreme_type": extrema[type_col].map(_normalise_extreme_type),
            "extreme_price": pd.to_numeric(extrema[price_col], errors="coerce"),
        }
    )
    out["bar_index"] = pd.to_numeric(extrema[index_col], errors="coerce") if index_col else np.nan
    return (
        out.dropna(subset=["symbol", "event_date", "extreme_type", "extreme_price"])
        .sort_values(["symbol", "event_date", "confirm_date"])
        .reset_index(drop=True)
    )


def _prepare_price_table(
    price_data: pd.DataFrame,
    *,
    symbol_col: str,
    date_col: str,
    close_col: str,
    price_mode: Literal["raw", "log"],
    volatility_window: int,
    volatility_min_periods: int,
) -> pd.DataFrame:
    required = {symbol_col, date_col, close_col}
    missing = sorted(required.difference(price_data.columns))
    if missing:
        raise ValueError(f"缺少价格字段: {missing}")

    prices = price_data[[symbol_col, date_col, close_col]].copy()
    prices["symbol"] = prices[symbol_col].astype(str).str.zfill(6)
    prices["date"] = pd.to_datetime(prices[date_col])
    prices["log_close"] = _log_level(prices[close_col], price_mode)
    prices = prices.dropna(subset=["symbol", "date", "log_close"]).sort_values(["symbol", "date"])
    prices["bar_index"] = prices.groupby("symbol").cumcount()
    prices["log_return"] = prices.groupby("symbol")["log_close"].diff()
    prices["hist_vol"] = (
        prices.groupby("symbol")["log_return"]
        .rolling(window=volatility_window, min_periods=volatility_min_periods)
        .std()
        .reset_index(level=0, drop=True)
    )
    return prices[["symbol", "date", "bar_index", "hist_vol"]]


def _robust_standardise(x_raw: pd.Series) -> pd.Series:
    x = pd.to_numeric(x_raw, errors="coerce")
    median = float(x.median())
    mad = float((x - median).abs().median())
    if np.isfinite(mad) and mad > 0:
        return (x - median) / mad

    mean = float(x.mean())
    std = float(x.std(ddof=0))
    if np.isfinite(std) and std > 0:
        return (x - mean) / std
    return pd.Series(np.nan, index=x.index, dtype=float)


def build_extremum_observable(
    extrema_data: pd.DataFrame | str | Path,
    price_data: pd.DataFrame | str | Path | None = None,
    *,
    symbol_col: str = "stock_id",
    date_col: str = "date",
    close_col: str = "close",
    new_extremum_col: str = "new_extremum_confirmed",
    extreme_type_col: str = "confirmed_extremum_kind",
    extreme_price_col: str = "confirmed_extremum_value",
    extreme_index_col: str = "confirmed_extremum_idx",
    price_mode: PriceMode = "auto",
    volatility_window: int = 20,
    volatility_min_periods: int = 10,
) -> pd.DataFrame:
    """构造单变量“同类极值迁移” observable。

    参数
    ----
    extrema_data:
        已有的 automatic123 极值点输出。可以是包含 ``confirmed_extremum_*``
        字段的完整表，也可以是已经清洗好的极值点表，例如包含
        ``extrema_type`` 和 ``extrema_price`` 字段的表。
    price_data:
        日频价格表，只用于计算历史波动率。如果不传，则复用
        ``extrema_data``。这种情况下，``extrema_data`` 必须包含配置好的
        symbol/date/close 字段。
    price_mode:
        如果价格已经是 log price，用 ``"log"``；如果是正的原始价格，用
        ``"raw"``；如果希望从 close 列自动判断，用 ``"auto"``。

    返回
    ----
    pandas.DataFrame
        最小 event-time observable 表，只包含以下列：
        ``event_id``, ``date``, ``bar_index``, ``extreme_type``,
        ``extreme_price`` 和 ``observable``。
    """

    if volatility_window < 2:
        raise ValueError("volatility_window 必须至少为 2")
    if volatility_min_periods < 2:
        raise ValueError("volatility_min_periods 必须至少为 2")

    source = _read_table(extrema_data)
    prices_source = _read_table(price_data) if price_data is not None else source.copy()

    if {new_extremum_col, extreme_type_col, extreme_price_col}.issubset(source.columns):
        extrema = _extract_extrema_from_state(
            source,
            symbol_col=symbol_col,
            date_col=date_col,
            new_extremum_col=new_extremum_col,
            extreme_type_col=extreme_type_col,
            extreme_price_col=extreme_price_col,
            extreme_index_col=extreme_index_col,
        )
    else:
        extrema = _normalise_extrema_table(source)

    if extrema.empty:
        return pd.DataFrame(columns=["event_id", "date", "bar_index", "extreme_type", "extreme_price", "observable"])

    resolved_price_mode: Literal["raw", "log"]
    if price_mode == "auto":
        resolved_price_mode = _infer_price_mode(prices_source[close_col])
    elif price_mode in {"raw", "log"}:
        resolved_price_mode = price_mode
    else:
        raise ValueError("price_mode 必须是 auto、raw 或 log")

    prices = _prepare_price_table(
        prices_source,
        symbol_col=symbol_col,
        date_col=date_col,
        close_col=close_col,
        price_mode=resolved_price_mode,
        volatility_window=volatility_window,
        volatility_min_periods=volatility_min_periods,
    )

    extrema["log_extreme_price"] = _log_level(extrema["extreme_price"], resolved_price_mode)
    extrema = extrema.dropna(subset=["log_extreme_price"]).copy()

    price_by_symbol = {
        symbol: grp.sort_values("date").reset_index(drop=True)
        for symbol, grp in prices.groupby("symbol", sort=False)
    }

    rows: list[dict[str, Any]] = []
    for symbol, grp in extrema.groupby("symbol", sort=False):
        if symbol not in price_by_symbol:
            continue
        price_grp = price_by_symbol[symbol]
        event_dates = price_grp["date"].to_numpy(dtype="datetime64[ns]")
        last_same: dict[str, dict[str, Any]] = {}
        event_seq = 0

        for ext in grp.sort_values(["event_date", "confirm_date"]).itertuples(index=False):
            kind = str(ext.extreme_type)
            event_date = pd.Timestamp(ext.event_date)
            pos = int(np.searchsorted(event_dates, np.datetime64(event_date), side="left"))
            if pos >= len(price_grp) or pd.Timestamp(price_grp.loc[pos, "date"]) != event_date:
                pos = int(np.searchsorted(event_dates, np.datetime64(event_date), side="right") - 1)
            if pos < 0:
                continue

            current = {
                "event_date": event_date,
                "bar_index": int(price_grp.loc[pos, "bar_index"]),
                "log_price": float(ext.log_extreme_price),
                "extreme_price": float(ext.extreme_price),
                "hist_vol": float(price_grp.loc[pos, "hist_vol"]),
            }

            previous = last_same.get(kind)
            last_same[kind] = current
            if previous is None:
                continue

            d_k = current["bar_index"] - int(previous["bar_index"])
            sigma = float(previous["hist_vol"])
            if d_k <= 0 or not np.isfinite(sigma) or sigma <= 0:
                continue

            delta = current["log_price"] - float(previous["log_price"])
            z_k = delta / (sigma * np.sqrt(float(d_k)))
            if not np.isfinite(z_k):
                continue

            event_seq += 1
            rows.append(
                {
                    "event_id": f"{symbol}_{event_seq:06d}",
                    "date": event_date,
                    "bar_index": current["bar_index"],
                    "extreme_type": kind,
                    "extreme_price": current["extreme_price"],
                    "x_raw": float(np.arcsinh(z_k)),
                    "symbol": symbol,
                }
            )

    if not rows:
        return pd.DataFrame(columns=["event_id", "date", "bar_index", "extreme_type", "extreme_price", "observable"])

    out = pd.DataFrame(rows).sort_values(["symbol", "date", "bar_index"]).reset_index(drop=True)
    out["observable"] = out.groupby("symbol", group_keys=False)["x_raw"].apply(_robust_standardise)
    out = out.dropna(subset=["observable"])
    return out[["event_id", "date", "bar_index", "extreme_type", "extreme_price", "observable"]].reset_index(drop=True)


__all__ = ["build_extremum_observable"]
