"""Search Hankel-delay extremum maps for interpretable price/change modes.

The experiment keeps the input clean:

* price:  log extreme price only
* change: same-type change only (HH for highs, LL for lows)

For each input it fits Hankel-delay versions of:

1. same-section return maps: High->High and Low->Low;
2. two-step event map: x_k -> x_{k+2};
3. phase maps and composition maps: High->Low, Low->High, then products.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

STATE_MODES = {"price": "log_price", "change": "same_type_change"}


@dataclass(frozen=True)
class ScalarScaler:
    mean: float
    std: float

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.std

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return values * self.std + self.mean


@dataclass(frozen=True)
class DelayMap:
    name: str
    method: str
    state_mode: str
    source_section: str
    target_section: str
    m: int
    requested_rank: int
    actual_rank: int
    A: np.ndarray
    source_scaler: ScalarScaler
    target_scaler: ScalarScaler
    source_values: np.ndarray
    target_values: np.ndarray
    singular_values: np.ndarray


def _parse_int_list(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def _parse_extreme_type(value: object) -> str:
    text = str(value).strip().lower()
    if text.startswith("h") or text in {"1", "top", "max"}:
        return "high"
    if text.startswith("l") or text in {"-1", "bottom", "min"}:
        return "low"
    raise ValueError(f"Unsupported extreme_type: {value!r}")


def _read_events(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"event_id", "date", "bar_index", "extreme_type", "observable"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")

    df = df.sort_values(["date", "bar_index", "event_id"]).reset_index(drop=True)
    df["extreme_type"] = df["extreme_type"].map(_parse_extreme_type)
    df["log_price"] = pd.to_numeric(df["observable"], errors="coerce")
    if not np.isfinite(df["log_price"]).all():
        raise ValueError("observable contains NaN or inf")
    df["same_type_change"] = df.groupby("extreme_type", sort=False)["log_price"].diff()
    df = df.dropna(subset=["same_type_change"]).reset_index(drop=True)
    return df


def _fit_scaler(values: np.ndarray) -> ScalarScaler:
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=0))
    if std <= np.finfo(float).eps:
        std = 1.0
    return ScalarScaler(mean=mean, std=std)


def _delay_matrix(values: np.ndarray, m: int) -> np.ndarray:
    if values.ndim != 1:
        raise ValueError("values must be 1D")
    columns = values.size - m + 1
    if columns < 2:
        raise ValueError(f"not enough values={values.size} for delay m={m}")
    return np.column_stack([values[i : i + m] for i in range(columns)])


def _fit_operator(X: np.ndarray, Y: np.ndarray, requested_rank: int) -> tuple[np.ndarray, int, np.ndarray]:
    u, s, vh = np.linalg.svd(X, full_matrices=False)
    valid = s > np.finfo(float).eps * max(X.shape) * s[0]
    max_rank = int(np.sum(valid))
    if max_rank <= 0:
        raise ValueError("X has zero numerical rank")
    actual_rank = min(int(requested_rank), max_rank)
    if actual_rank <= 0:
        raise ValueError(f"rank must be positive, got {requested_rank}")
    u_r = u[:, :actual_rank]
    s_r = s[:actual_rank]
    vh_r = vh[:actual_rank, :]
    A = Y @ vh_r.T @ np.diag(1.0 / s_r) @ u_r.T
    return A, actual_rank, s


def _fit_delay_map(
    *,
    name: str,
    method: str,
    state_mode: str,
    source_section: str,
    target_section: str,
    source_values: np.ndarray,
    target_values: np.ndarray,
    m: int,
    rank: int,
    source_scaler: ScalarScaler,
    target_scaler: ScalarScaler,
) -> DelayMap:
    length = min(source_values.size, target_values.size)
    source_values = np.asarray(source_values[:length], dtype=float)
    target_values = np.asarray(target_values[:length], dtype=float)
    source_z = source_scaler.transform(source_values)
    target_z = target_scaler.transform(target_values)
    X = _delay_matrix(source_z, m)
    Y = _delay_matrix(target_z, m)
    A, actual_rank, s = _fit_operator(X, Y, requested_rank=rank)
    return DelayMap(
        name=name,
        method=method,
        state_mode=state_mode,
        source_section=source_section,
        target_section=target_section,
        m=m,
        requested_rank=rank,
        actual_rank=actual_rank,
        A=A,
        source_scaler=source_scaler,
        target_scaler=target_scaler,
        source_values=source_values,
        target_values=target_values,
        singular_values=s,
    )


def _paired_phase_values(events: pd.DataFrame, source: str, target: str, column: str) -> tuple[np.ndarray, np.ndarray]:
    xs: list[float] = []
    ys: list[float] = []
    for i in range(len(events) - 1):
        if events.loc[i, "extreme_type"] == source and events.loc[i + 1, "extreme_type"] == target:
            xs.append(float(events.loc[i, column]))
            ys.append(float(events.loc[i + 1, column]))
    if len(xs) < 3:
        raise ValueError(f"not enough {source}->{target} pairs")
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def _windows_for_raw_values(values: np.ndarray, scaler: ScalarScaler, m: int) -> np.ndarray:
    return _delay_matrix(scaler.transform(values), m)


def _predict_last_values(model: DelayMap, A: np.ndarray | None = None, source_values: np.ndarray | None = None, target_values: np.ndarray | None = None, source_scaler: ScalarScaler | None = None, target_scaler: ScalarScaler | None = None) -> tuple[np.ndarray, np.ndarray]:
    A = model.A if A is None else A
    source_values = model.source_values if source_values is None else source_values
    target_values = model.target_values if target_values is None else target_values
    source_scaler = model.source_scaler if source_scaler is None else source_scaler
    target_scaler = model.target_scaler if target_scaler is None else target_scaler
    length = min(source_values.size, target_values.size)
    X = _windows_for_raw_values(source_values[:length], source_scaler, model.m)
    Y = _delay_matrix(target_scaler.transform(target_values[:length]), model.m)
    pred_z = A @ X
    pred_last = target_scaler.inverse_transform(pred_z[-1])
    true_last = target_scaler.inverse_transform(Y[-1])
    return pred_last, true_last


def _baseline_last_values(model: DelayMap, source_values: np.ndarray | None = None, target_values: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    source_values = model.source_values if source_values is None else source_values
    target_values = model.target_values if target_values is None else target_values
    length = min(source_values.size, target_values.size)
    source_windows = _delay_matrix(source_values[:length], model.m)
    target_windows = _delay_matrix(target_values[:length], model.m)
    if model.state_mode == "price":
        baseline = source_windows[-1]
    else:
        baseline = np.zeros(source_windows.shape[1])
    return baseline, target_windows[-1]


def _metric_dict(model: DelayMap, *, A: np.ndarray | None = None, source_values: np.ndarray | None = None, target_values: np.ndarray | None = None, name: str | None = None, method: str | None = None, source_section: str | None = None, target_section: str | None = None) -> dict[str, float | str | int]:
    pred, true = _predict_last_values(model, A=A, source_values=source_values, target_values=target_values)
    baseline, _ = _baseline_last_values(model, source_values=source_values, target_values=target_values)
    err = pred - true
    base_err = baseline - true
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    base_rmse = float(np.sqrt(np.mean(base_err**2)))
    base_mae = float(np.mean(np.abs(base_err)))
    energy = model.singular_values**2
    energy_at_rank = float(np.cumsum(energy)[model.actual_rank - 1] / np.sum(energy)) if np.sum(energy) > 0 else np.nan
    return {
        "name": name or model.name,
        "method": method or model.method,
        "state_mode": model.state_mode,
        "source_section": source_section or model.source_section,
        "target_section": target_section or model.target_section,
        "m": model.m,
        "requested_rank": model.requested_rank,
        "actual_rank": model.actual_rank,
        "sample_count": int(pred.size),
        "rmse": rmse,
        "mae": mae,
        "baseline_rmse": base_rmse,
        "baseline_mae": base_mae,
        "rmse_improvement_vs_baseline": (base_rmse - rmse) / base_rmse if base_rmse > 0 else np.nan,
        "mae_improvement_vs_baseline": (base_mae - mae) / base_mae if base_mae > 0 else np.nan,
        "energy_at_rank": energy_at_rank,
    }


def _rmse(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(values**2)))


def _mae(values: np.ndarray) -> float:
    return float(np.mean(np.abs(values)))


def _raw_delay_pair(model: DelayMap) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    length = min(model.source_values.size, model.target_values.size)
    source_values = model.source_values[:length]
    target_values = model.target_values[:length]
    x_raw = _delay_matrix(source_values, model.m)
    y_raw = _delay_matrix(target_values, model.m)
    return source_values, target_values, x_raw, y_raw


def _default_baseline_for_columns(model: DelayMap, x_raw: np.ndarray, count: int) -> np.ndarray:
    if model.state_mode == "price":
        return x_raw[-1, -count:]
    return np.zeros(count, dtype=float)


def _baseline_metrics(
    *,
    model: DelayMap,
    x_raw: np.ndarray,
    y_raw: np.ndarray,
    train_cols: int,
    true: np.ndarray,
) -> dict[str, float | str]:
    default_baseline = _default_baseline_for_columns(model, x_raw[:, train_cols:], true.size)
    mean_baseline = np.full(true.size, float(np.mean(y_raw[-1, :train_cols])))
    persistence_baseline = y_raw[-1, train_cols - 1 : y_raw.shape[1] - 1]

    default_err = default_baseline - true
    mean_err = mean_baseline - true
    persistence_err = persistence_baseline - true
    baseline_rmses = {
        "default": _rmse(default_err),
        "target_train_mean": _rmse(mean_err),
        "target_persistence": _rmse(persistence_err),
    }
    best_name = min(baseline_rmses, key=baseline_rmses.get)
    return {
        "default_baseline_rmse": baseline_rmses["default"],
        "default_baseline_mae": _mae(default_err),
        "target_mean_baseline_rmse": baseline_rmses["target_train_mean"],
        "target_mean_baseline_mae": _mae(mean_err),
        "target_persistence_baseline_rmse": baseline_rmses["target_persistence"],
        "target_persistence_baseline_mae": _mae(persistence_err),
        "best_baseline": best_name,
        "best_baseline_rmse": baseline_rmses[best_name],
    }


def _holdout_fit(model: DelayMap, train_fraction: float) -> dict[str, object]:
    source_values, target_values, x_raw, y_raw = _raw_delay_pair(model)
    n_cols = x_raw.shape[1]
    train_cols = max(model.requested_rank + 1, int(np.floor(n_cols * train_fraction)))
    if train_cols >= n_cols:
        raise ValueError(f"not enough holdout columns for {model.name}, m={model.m}, rank={model.requested_rank}")

    source_scaler = _fit_scaler(source_values[: train_cols + model.m - 1])
    target_scaler = _fit_scaler(target_values[: train_cols + model.m - 1])
    x = _delay_matrix(source_scaler.transform(source_values), model.m)
    y = _delay_matrix(target_scaler.transform(target_values), model.m)
    A, actual_rank, s = _fit_operator(x[:, :train_cols], y[:, :train_cols], requested_rank=model.requested_rank)
    return {
        "A": A,
        "actual_rank": actual_rank,
        "singular_values": s,
        "train_cols": train_cols,
        "x": x,
        "y": y,
        "x_raw": x_raw,
        "y_raw": y_raw,
        "target_scaler": target_scaler,
    }


def _holdout_metric_dict(model: DelayMap, train_fraction: float) -> tuple[dict[str, float | str | int], np.ndarray]:
    fitted = _holdout_fit(model, train_fraction)
    A = fitted["A"]
    train_cols = int(fitted["train_cols"])
    x_raw = fitted["x_raw"]
    y_raw = fitted["y_raw"]
    target_scaler = fitted["target_scaler"]
    pred_z = A @ fitted["x"][:, train_cols:]
    pred = target_scaler.inverse_transform(pred_z[-1])
    true = y_raw[-1, train_cols:]
    err = pred - true
    baseline = _baseline_metrics(model=model, x_raw=x_raw, y_raw=y_raw, train_cols=train_cols, true=true)
    s = fitted["singular_values"]
    energy = s**2
    actual_rank = int(fitted["actual_rank"])
    energy_at_rank = float(np.cumsum(energy)[actual_rank - 1] / np.sum(energy)) if np.sum(energy) > 0 else np.nan
    rmse = _rmse(err)
    mae = _mae(err)
    row = {
        "name": model.name,
        "method": model.method,
        "state_mode": model.state_mode,
        "source_section": model.source_section,
        "target_section": model.target_section,
        "m": model.m,
        "requested_rank": model.requested_rank,
        "actual_rank": actual_rank,
        "train_cols": train_cols,
        "test_count": int(true.size),
        "test_rmse": rmse,
        "test_mae": mae,
        "test_median_abs_error": float(np.median(np.abs(err))),
        "energy_at_rank_train": energy_at_rank,
        **baseline,
    }
    row["test_rmse_improvement_vs_default_baseline"] = (
        (row["default_baseline_rmse"] - rmse) / row["default_baseline_rmse"] if row["default_baseline_rmse"] > 0 else np.nan
    )
    row["test_rmse_improvement_vs_best_baseline"] = (
        (row["best_baseline_rmse"] - rmse) / row["best_baseline_rmse"] if row["best_baseline_rmse"] > 0 else np.nan
    )
    return row, A


def _expanding_one_step_metric_dict(model: DelayMap, min_train_fraction: float) -> dict[str, float | str | int]:
    source_values, target_values, x_raw, y_raw = _raw_delay_pair(model)
    n_cols = x_raw.shape[1]
    start_col = max(model.requested_rank + 1, int(np.floor(n_cols * min_train_fraction)))
    if start_col >= n_cols:
        raise ValueError(f"not enough expanding test columns for {model.name}, m={model.m}, rank={model.requested_rank}")

    pred_values: list[float] = []
    true_values: list[float] = []
    default_values: list[float] = []
    mean_values: list[float] = []
    persistence_values: list[float] = []
    last_actual_rank = 0

    for col in range(start_col, n_cols):
        source_scaler = _fit_scaler(source_values[: col + model.m - 1])
        target_scaler = _fit_scaler(target_values[: col + model.m - 1])
        x = _delay_matrix(source_scaler.transform(source_values), model.m)
        y = _delay_matrix(target_scaler.transform(target_values), model.m)
        A, actual_rank, _ = _fit_operator(x[:, :col], y[:, :col], requested_rank=model.requested_rank)
        pred = float(target_scaler.inverse_transform((A @ x[:, [col]])[-1:]).reshape(-1)[0])
        true = float(y_raw[-1, col])
        default = float(x_raw[-1, col]) if model.state_mode == "price" else 0.0
        pred_values.append(pred)
        true_values.append(true)
        default_values.append(default)
        mean_values.append(float(np.mean(y_raw[-1, :col])))
        persistence_values.append(float(y_raw[-1, col - 1]))
        last_actual_rank = actual_rank

    pred_arr = np.asarray(pred_values)
    true_arr = np.asarray(true_values)
    default_arr = np.asarray(default_values)
    mean_arr = np.asarray(mean_values)
    persistence_arr = np.asarray(persistence_values)
    err = pred_arr - true_arr
    default_err = default_arr - true_arr
    mean_err = mean_arr - true_arr
    persistence_err = persistence_arr - true_arr
    baseline_rmses = {
        "default": _rmse(default_err),
        "target_train_mean": _rmse(mean_err),
        "target_persistence": _rmse(persistence_err),
    }
    best_name = min(baseline_rmses, key=baseline_rmses.get)
    rmse = _rmse(err)
    row = {
        "name": model.name,
        "method": model.method,
        "state_mode": model.state_mode,
        "source_section": model.source_section,
        "target_section": model.target_section,
        "m": model.m,
        "requested_rank": model.requested_rank,
        "actual_rank_last": int(last_actual_rank),
        "start_col": int(start_col),
        "test_count": int(true_arr.size),
        "expanding_rmse": rmse,
        "expanding_mae": _mae(err),
        "expanding_median_abs_error": float(np.median(np.abs(err))),
        "default_baseline_rmse": baseline_rmses["default"],
        "target_mean_baseline_rmse": baseline_rmses["target_train_mean"],
        "target_persistence_baseline_rmse": baseline_rmses["target_persistence"],
        "best_baseline": best_name,
        "best_baseline_rmse": baseline_rmses[best_name],
    }
    row["expanding_rmse_improvement_vs_default_baseline"] = (
        (row["default_baseline_rmse"] - rmse) / row["default_baseline_rmse"] if row["default_baseline_rmse"] > 0 else np.nan
    )
    row["expanding_rmse_improvement_vs_best_baseline"] = (
        (row["best_baseline_rmse"] - rmse) / row["best_baseline_rmse"] if row["best_baseline_rmse"] > 0 else np.nan
    )
    return row


def _dynamic_label(lam: complex) -> str:
    if abs(lam.imag) > 1e-10:
        return "oscillatory"
    if lam.real < 0:
        return "alternating_sign"
    if abs(lam) < 1:
        return "decaying_same_sign"
    if abs(lam) > 1:
        return "amplifying_same_sign"
    return "neutral"


def _mode_shape_stats(mode: np.ndarray) -> dict[str, float | int]:
    real = np.real(mode)
    signs = np.sign(real)
    signs = signs[signs != 0]
    sign_changes = int(np.sum(signs[1:] != signs[:-1])) if signs.size > 1 else 0
    max_idx = int(np.argmax(np.abs(mode)))
    endpoint_ratio = float(abs(mode[-1]) / max(abs(mode[0]), np.finfo(float).eps))
    return {
        "dominant_delay": max_idx,
        "sign_changes_real": sign_changes,
        "endpoint_abs_ratio": endpoint_ratio,
    }


def _is_delay0_anchor_artifact(mode: np.ndarray, tol: float = 1e-8) -> bool:
    """Filter modes dominated by a real-valued first delay coordinate.

    The sign/phase of an eigenvector is arbitrary, so both +1 and -1 are
    treated as the same anchor artifact when delay 0 is the dominant entry.
    """
    if mode.size == 0:
        return False
    dominant_delay = int(np.argmax(np.abs(mode)))
    first = complex(mode[0])
    return (
        dominant_delay == 0
        and abs(abs(first) - 1.0) <= tol
        and abs(first.imag) <= tol
        and abs(abs(first.real) - 1.0) <= tol
    )


def _eigen_rows(model: DelayMap, *, A: np.ndarray | None = None, name: str | None = None, method: str | None = None, source_section: str | None = None, target_section: str | None = None, top_k: int = 5) -> tuple[list[dict[str, float | str | int]], list[dict[str, float | str | int]]]:
    A = model.A if A is None else A
    eigs, vecs = np.linalg.eig(A)
    order = np.argsort(-np.abs(eigs))[: min(top_k, eigs.size)]
    rows: list[dict[str, float | str | int]] = []
    profile_rows: list[dict[str, float | str | int]] = []
    kept_rank = 0
    for mode_id in order:
        lam = complex(eigs[mode_id])
        mode = vecs[:, mode_id]
        max_abs = np.max(np.abs(mode))
        if max_abs > 0:
            mode = mode / max_abs
        if _is_delay0_anchor_artifact(mode):
            continue
        kept_rank += 1
        angle = float(np.angle(lam))
        stats = _mode_shape_stats(mode)
        row = {
            "name": name or model.name,
            "method": method or model.method,
            "state_mode": model.state_mode,
            "source_section": source_section or model.source_section,
            "target_section": target_section or model.target_section,
            "m": model.m,
            "requested_rank": model.requested_rank,
            "actual_rank": model.actual_rank,
            "mode_rank": kept_rank,
            "mode_id": int(mode_id),
            "lambda_real": float(lam.real),
            "lambda_imag": float(lam.imag),
            "lambda_abs": float(abs(lam)),
            "lambda_angle": angle,
            "period_sections": float(2 * np.pi / abs(angle)) if abs(angle) > 1e-12 else np.inf,
            "dynamic_label": _dynamic_label(lam),
            **stats,
        }
        rows.append(row)
        for delay_idx, value in enumerate(mode):
            profile_rows.append(
                {
                    "name": row["name"],
                    "method": row["method"],
                    "state_mode": model.state_mode,
                    "m": model.m,
                    "requested_rank": model.requested_rank,
                    "actual_rank": model.actual_rank,
                    "mode_rank": kept_rank,
                    "mode_id": int(mode_id),
                    "delay_index": int(delay_idx),
                    "mode_real": float(complex(value).real),
                    "mode_imag": float(complex(value).imag),
                    "mode_abs": float(abs(value)),
                }
            )
    return rows, profile_rows


def _latest_score_rows(model: DelayMap, *, A: np.ndarray | None = None, source_values: np.ndarray | None = None, name: str | None = None, method: str | None = None) -> list[dict[str, float | str | int]]:
    A = model.A if A is None else A
    source_values = model.source_values if source_values is None else source_values
    source_z = model.source_scaler.transform(source_values)
    latest_window = source_z[-model.m :]
    eigs, vecs = np.linalg.eig(A)
    coeffs = np.linalg.pinv(vecs) @ latest_window
    rows = []
    for mode_id, coeff in enumerate(coeffs):
        mode = vecs[:, mode_id]
        max_abs = np.max(np.abs(mode))
        if max_abs > 0:
            mode = mode / max_abs
        if _is_delay0_anchor_artifact(mode):
            continue
        lam = complex(eigs[mode_id])
        rows.append(
            {
                "name": name or model.name,
                "method": method or model.method,
                "state_mode": model.state_mode,
                "m": model.m,
                "requested_rank": model.requested_rank,
                "actual_rank": model.actual_rank,
                "mode_id": int(mode_id),
                "lambda_real": float(lam.real),
                "lambda_imag": float(lam.imag),
                "coefficient_real": float(complex(coeff).real),
                "coefficient_imag": float(complex(coeff).imag),
                "coefficient_abs": float(abs(coeff)),
            }
        )
    return rows


def _series(events: pd.DataFrame, column: str) -> dict[str, np.ndarray]:
    highs = events[events["extreme_type"] == "high"][column].to_numpy(float)
    lows = events[events["extreme_type"] == "low"][column].to_numpy(float)
    event = events[column].to_numpy(float)
    hl_x, hl_y = _paired_phase_values(events, "high", "low", column)
    lh_x, lh_y = _paired_phase_values(events, "low", "high", column)
    return {
        "high": highs,
        "low": lows,
        "event": event,
        "hl_x": hl_x,
        "hl_y": hl_y,
        "lh_x": lh_x,
        "lh_y": lh_y,
    }


def _run_one(
    *,
    events: pd.DataFrame,
    state_mode: str,
    m_values: Iterable[int],
    rank_values: Iterable[int],
    train_fraction: float,
    expanding_min_train_fraction: float,
) -> tuple[
    list[dict[str, float | str | int]],
    list[dict[str, float | str | int]],
    list[dict[str, float | str | int]],
    list[dict[str, float | str | int]],
    list[dict[str, float | str | int]],
    list[dict[str, float | str | int]],
    list[dict[str, float | str | int]],
    list[dict[str, str | int]],
]:
    column = STATE_MODES[state_mode]
    data = _series(events, column)
    high_scaler = _fit_scaler(data["high"])
    low_scaler = _fit_scaler(data["low"])
    event_scaler = _fit_scaler(data["event"])

    summary_rows: list[dict[str, float | str | int]] = []
    eigen_rows: list[dict[str, float | str | int]] = []
    profile_rows: list[dict[str, float | str | int]] = []
    latest_rows: list[dict[str, float | str | int]] = []
    holdout_rows: list[dict[str, float | str | int]] = []
    holdout_eigen_rows: list[dict[str, float | str | int]] = []
    expanding_rows: list[dict[str, float | str | int]] = []
    failure_rows: list[dict[str, str | int]] = []

    for m in m_values:
        for rank in rank_values:
            fits: list[DelayMap] = []
            try:
                fits.append(
                    _fit_delay_map(
                        name="return_high",
                        method="same_section_return_map",
                        state_mode=state_mode,
                        source_section="high",
                        target_section="high",
                        source_values=data["high"][:-1],
                        target_values=data["high"][1:],
                        m=m,
                        rank=rank,
                        source_scaler=high_scaler,
                        target_scaler=high_scaler,
                    )
                )
                fits.append(
                    _fit_delay_map(
                        name="return_low",
                        method="same_section_return_map",
                        state_mode=state_mode,
                        source_section="low",
                        target_section="low",
                        source_values=data["low"][:-1],
                        target_values=data["low"][1:],
                        m=m,
                        rank=rank,
                        source_scaler=low_scaler,
                        target_scaler=low_scaler,
                    )
                )
                fits.append(
                    _fit_delay_map(
                        name="two_step_same_section",
                        method="two_step_event_map",
                        state_mode=state_mode,
                        source_section="event",
                        target_section="event_plus_2",
                        source_values=data["event"][:-2],
                        target_values=data["event"][2:],
                        m=m,
                        rank=rank,
                        source_scaler=event_scaler,
                        target_scaler=event_scaler,
                    )
                )
                hl = _fit_delay_map(
                    name="phase_high_to_low",
                    method="phase_map",
                    state_mode=state_mode,
                    source_section="high",
                    target_section="low",
                    source_values=data["hl_x"],
                    target_values=data["hl_y"],
                    m=m,
                    rank=rank,
                    source_scaler=high_scaler,
                    target_scaler=low_scaler,
                )
                lh = _fit_delay_map(
                    name="phase_low_to_high",
                    method="phase_map",
                    state_mode=state_mode,
                    source_section="low",
                    target_section="high",
                    source_values=data["lh_x"],
                    target_values=data["lh_y"],
                    m=m,
                    rank=rank,
                    source_scaler=low_scaler,
                    target_scaler=high_scaler,
                )
                fits.extend([hl, lh])
            except Exception as exc:
                failure_rows.append({"state_mode": state_mode, "m": m, "rank": rank, "method": "fit_base_maps", "error": str(exc)})
                continue

            for fit in fits:
                try:
                    summary_rows.append(_metric_dict(fit))
                    eig, prof = _eigen_rows(fit)
                    eigen_rows.extend(eig)
                    profile_rows.extend(prof)
                    latest_rows.extend(_latest_score_rows(fit))
                    holdout, holdout_A = _holdout_metric_dict(fit, train_fraction=train_fraction)
                    holdout_rows.append(holdout)
                    holdout_eig, _ = _eigen_rows(fit, A=holdout_A)
                    for row in holdout_eig:
                        row["train_cols"] = holdout["train_cols"]
                        row["test_count"] = holdout["test_count"]
                    holdout_eigen_rows.extend(holdout_eig)
                    expanding_rows.append(
                        _expanding_one_step_metric_dict(
                            fit,
                            min_train_fraction=expanding_min_train_fraction,
                        )
                    )
                except Exception as exc:
                    failure_rows.append({"state_mode": state_mode, "m": m, "rank": rank, "method": fit.name, "error": str(exc)})

            try:
                comp_high_A = lh.A @ hl.A
                comp_low_A = hl.A @ lh.A
                # Evaluate products against same-section return windows.
                ref_high = fits[0]
                ref_low = fits[1]
                summary_rows.append(
                    _metric_dict(
                        ref_high,
                        A=comp_high_A,
                        name="composition_high_return",
                        method="composition_map",
                        source_section="high",
                        target_section="high",
                    )
                )
                eig, prof = _eigen_rows(ref_high, A=comp_high_A, name="composition_high_return", method="composition_map", source_section="high", target_section="high")
                eigen_rows.extend(eig)
                profile_rows.extend(prof)
                latest_rows.extend(_latest_score_rows(ref_high, A=comp_high_A, name="composition_high_return", method="composition_map"))

                summary_rows.append(
                    _metric_dict(
                        ref_low,
                        A=comp_low_A,
                        name="composition_low_return",
                        method="composition_map",
                        source_section="low",
                        target_section="low",
                    )
                )
                eig, prof = _eigen_rows(ref_low, A=comp_low_A, name="composition_low_return", method="composition_map", source_section="low", target_section="low")
                eigen_rows.extend(eig)
                profile_rows.extend(prof)
                latest_rows.extend(_latest_score_rows(ref_low, A=comp_low_A, name="composition_low_return", method="composition_map"))
            except Exception as exc:
                failure_rows.append({"state_mode": state_mode, "m": m, "rank": rank, "method": "composition", "error": str(exc)})

    return summary_rows, eigen_rows, profile_rows, latest_rows, holdout_rows, holdout_eigen_rows, expanding_rows, failure_rows


def run_search(
    *,
    observable_path: Path,
    out_dir: Path,
    state_modes: Iterable[str],
    m_values: Iterable[int],
    rank_values: Iterable[int],
    train_fraction: float = 0.7,
    expanding_min_train_fraction: float = 0.5,
) -> dict[str, Path]:
    events = _read_events(observable_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, float | str | int]] = []
    eigen_rows: list[dict[str, float | str | int]] = []
    profile_rows: list[dict[str, float | str | int]] = []
    latest_rows: list[dict[str, float | str | int]] = []
    holdout_rows: list[dict[str, float | str | int]] = []
    holdout_eigen_rows: list[dict[str, float | str | int]] = []
    expanding_rows: list[dict[str, float | str | int]] = []
    failure_rows: list[dict[str, str | int]] = []
    for state_mode in state_modes:
        s, e, p, l, h, he, x, f = _run_one(
            events=events,
            state_mode=state_mode,
            m_values=m_values,
            rank_values=rank_values,
            train_fraction=train_fraction,
            expanding_min_train_fraction=expanding_min_train_fraction,
        )
        summary_rows.extend(s)
        eigen_rows.extend(e)
        profile_rows.extend(p)
        latest_rows.extend(l)
        holdout_rows.extend(h)
        holdout_eigen_rows.extend(he)
        expanding_rows.extend(x)
        failure_rows.extend(f)

    summary = pd.DataFrame(summary_rows).sort_values(["state_mode", "rmse", "mae"]).reset_index(drop=True)
    eigen = pd.DataFrame(eigen_rows)
    profiles = pd.DataFrame(profile_rows)
    latest = pd.DataFrame(latest_rows)
    holdout = pd.DataFrame(holdout_rows).sort_values(["state_mode", "test_rmse_improvement_vs_best_baseline"], ascending=[True, False]).reset_index(drop=True)
    holdout_eigen = pd.DataFrame(holdout_eigen_rows)
    expanding = pd.DataFrame(expanding_rows).sort_values(["state_mode", "expanding_rmse_improvement_vs_best_baseline"], ascending=[True, False]).reset_index(drop=True)
    failures = pd.DataFrame(failure_rows)

    best = summary.sort_values(["state_mode", "method", "rmse"]).groupby(["state_mode", "method"], as_index=False).head(3)

    paths = {
        "summary": out_dir / "hankel_map_summary.csv",
        "best_by_method": out_dir / "best_by_method.csv",
        "eigenmodes": out_dir / "hankel_map_eigenmodes.csv",
        "mode_profiles": out_dir / "hankel_map_mode_profiles.csv",
        "latest_scores": out_dir / "hankel_map_latest_scores.csv",
        "holdout_summary": out_dir / "hankel_map_holdout_summary.csv",
        "holdout_eigenmodes": out_dir / "hankel_map_holdout_eigenmodes.csv",
        "expanding_one_step": out_dir / "hankel_map_expanding_one_step.csv",
        "config": out_dir / "config.json",
    }
    summary.to_csv(paths["summary"], index=False, encoding="utf-8-sig")
    best.to_csv(paths["best_by_method"], index=False, encoding="utf-8-sig")
    eigen.to_csv(paths["eigenmodes"], index=False, encoding="utf-8-sig")
    profiles.to_csv(paths["mode_profiles"], index=False, encoding="utf-8-sig")
    latest.to_csv(paths["latest_scores"], index=False, encoding="utf-8-sig")
    holdout.to_csv(paths["holdout_summary"], index=False, encoding="utf-8-sig")
    holdout_eigen.to_csv(paths["holdout_eigenmodes"], index=False, encoding="utf-8-sig")
    expanding.to_csv(paths["expanding_one_step"], index=False, encoding="utf-8-sig")
    if not failures.empty:
        paths["failures"] = out_dir / "failures.csv"
        failures.to_csv(paths["failures"], index=False, encoding="utf-8-sig")
    else:
        stale_failures = out_dir / "failures.csv"
        if stale_failures.exists():
            stale_failures.unlink()

    paths["config"].write_text(
        json.dumps(
            {
                "observable_path": str(observable_path),
                "state_modes": list(state_modes),
                "m_values": list(m_values),
                "rank_values": list(rank_values),
                "train_fraction": train_fraction,
                "expanding_min_train_fraction": expanding_min_train_fraction,
                "methods": [
                    "same-section return maps: High->High and Low->Low",
                    "two-step event map: x_k -> x_{k+2}",
                    "phase maps and composition maps: High->Low, Low->High, then same-section products",
                ],
                "baseline": {
                    "price": "predict next delayed target last value as source delayed last value",
                    "change": "predict next delayed target last value as 0",
                    "target_train_mean": "predict next delayed target last value as the training mean of target last values",
                    "target_persistence": "predict next delayed target last value as the previous target last value",
                },
                "holdout_metric": "Fit A on the first train_fraction delay-window columns and score only the newest value in each held-out target window.",
                "expanding_one_step_metric": "For each test column, refit A using only earlier delay-window columns, then score the newest predicted target value.",
                "mode_profile": "Eigenvectors are normalized delay profiles. delay_index=m-1 is the newest point in the delay window.",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search Hankel-delay extremum maps.")
    parser.add_argument("--observable-path", required=True)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--state-modes", default="price,change")
    parser.add_argument("--m-values", default="3,5,10,20,40")
    parser.add_argument("--rank-values", default="1,2,3,5,10")
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--expanding-min-train-fraction", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    observable_path = Path(args.observable_path)
    out_dir = Path(args.out_dir) if args.out_dir else observable_path.parent / "extremum_hankel_map_search"
    paths = run_search(
        observable_path=observable_path,
        out_dir=out_dir,
        state_modes=[item.strip() for item in args.state_modes.split(",") if item.strip()],
        m_values=_parse_int_list(args.m_values),
        rank_values=_parse_int_list(args.rank_values),
        train_fraction=args.train_fraction,
        expanding_min_train_fraction=args.expanding_min_train_fraction,
    )
    print(f"output_dir: {out_dir}")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
