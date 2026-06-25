"""Run interpretable extremum-map DMD experiments.

This script deliberately separates two inputs instead of mixing them:

* price:  state = [log_price]
* change: state = [same_type_change], where change is HH for highs and LL for lows

For each input it runs the same three map families:

1. same-section return maps: High->High and Low->Low;
2. two-step event map: x_k -> x_{k+2};
3. composition maps: High->Low, Low->High, then their same-section products.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

STATE_MODES = {
    "price": ["log_price"],
    "change": ["same_type_change"],
}


@dataclass(frozen=True)
class Scaler:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean[:, None]) / self.std[:, None]

    def inverse_transform(self, z: np.ndarray) -> np.ndarray:
        return z * self.std[:, None] + self.mean[:, None]


@dataclass(frozen=True)
class FittedMap:
    name: str
    method: str
    source_section: str
    target_section: str
    state_mode: str
    state_columns: list[str]
    A: np.ndarray
    x_scaler: Scaler
    y_scaler: Scaler
    x_raw: np.ndarray
    y_raw: np.ndarray
    x_meta: pd.DataFrame
    y_meta: pd.DataFrame


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
    df["state_index"] = np.arange(len(df))
    return df


def _state_matrix(df: pd.DataFrame, state_columns: list[str]) -> np.ndarray:
    values = df[state_columns].to_numpy(float)
    return values.T


def _fit_scaler(x: np.ndarray) -> Scaler:
    mean = np.mean(x, axis=1)
    std = np.std(x, axis=1, ddof=0)
    std = np.where(std <= np.finfo(float).eps, 1.0, std)
    return Scaler(mean=mean, std=std)


def _fit_operator(x_z: np.ndarray, y_z: np.ndarray, rank: int) -> np.ndarray:
    if x_z.ndim != 2 or y_z.ndim != 2:
        raise ValueError("x_z and y_z must be 2D matrices")
    if x_z.shape != y_z.shape:
        raise ValueError(f"shape mismatch: X={x_z.shape}, Y={y_z.shape}")
    if x_z.shape[1] < 2:
        raise ValueError("at least two snapshot pairs are required")

    u, s, vh = np.linalg.svd(x_z, full_matrices=False)
    valid = s > np.finfo(float).eps * max(x_z.shape) * s[0]
    max_rank = int(np.sum(valid))
    if max_rank <= 0:
        raise ValueError("X has zero numerical rank")
    r = min(int(rank), max_rank)
    if r <= 0:
        raise ValueError(f"rank must be positive, got {rank}")
    u_r = u[:, :r]
    s_r = s[:r]
    vh_r = vh[:r, :]
    return y_z @ vh_r.T @ np.diag(1.0 / s_r) @ u_r.T


def _fit_map(
    *,
    name: str,
    method: str,
    source_section: str,
    target_section: str,
    state_mode: str,
    state_columns: list[str],
    x_df: pd.DataFrame,
    y_df: pd.DataFrame,
    rank: int,
    x_scaler: Scaler | None = None,
    y_scaler: Scaler | None = None,
) -> FittedMap:
    x_raw = _state_matrix(x_df, state_columns)
    y_raw = _state_matrix(y_df, state_columns)
    x_scaler = x_scaler or _fit_scaler(x_raw)
    y_scaler = y_scaler or _fit_scaler(y_raw)
    A = _fit_operator(x_scaler.transform(x_raw), y_scaler.transform(y_raw), rank=rank)
    return FittedMap(
        name=name,
        method=method,
        source_section=source_section,
        target_section=target_section,
        state_mode=state_mode,
        state_columns=state_columns,
        A=A,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        x_raw=x_raw,
        y_raw=y_raw,
        x_meta=x_df.reset_index(drop=True),
        y_meta=y_df.reset_index(drop=True),
    )


def _section_return_pairs(events: pd.DataFrame, section: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    part = events[events["extreme_type"] == section].reset_index(drop=True)
    if len(part) < 3:
        raise ValueError(f"not enough {section} events")
    return part.iloc[:-1].reset_index(drop=True), part.iloc[1:].reset_index(drop=True)


def _two_step_pairs(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows_x = []
    rows_y = []
    for i in range(len(events) - 2):
        if events.loc[i, "extreme_type"] == events.loc[i + 2, "extreme_type"]:
            rows_x.append(events.iloc[i])
            rows_y.append(events.iloc[i + 2])
    if len(rows_x) < 3:
        raise ValueError("not enough two-step same-section pairs")
    return pd.DataFrame(rows_x).reset_index(drop=True), pd.DataFrame(rows_y).reset_index(drop=True)


def _phase_pairs(events: pd.DataFrame, source: str, target: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows_x = []
    rows_y = []
    for i in range(len(events) - 1):
        if events.loc[i, "extreme_type"] == source and events.loc[i + 1, "extreme_type"] == target:
            rows_x.append(events.iloc[i])
            rows_y.append(events.iloc[i + 1])
    if len(rows_x) < 3:
        raise ValueError(f"not enough {source}->{target} phase pairs")
    return pd.DataFrame(rows_x).reset_index(drop=True), pd.DataFrame(rows_y).reset_index(drop=True)


def _predict_raw(fitted: FittedMap) -> np.ndarray:
    pred_z = fitted.A @ fitted.x_scaler.transform(fitted.x_raw)
    return fitted.y_scaler.inverse_transform(pred_z)


def _baseline_raw(fitted: FittedMap) -> np.ndarray:
    baseline = np.zeros_like(fitted.y_raw)
    for i, col in enumerate(fitted.state_columns):
        if col == "log_price":
            baseline[i, :] = fitted.x_raw[i, :]
        elif col == "same_type_change":
            baseline[i, :] = 0.0
        else:
            baseline[i, :] = fitted.x_raw[i, :]
    return baseline


def _metric_rows(fitted: FittedMap) -> list[dict[str, float | str | int]]:
    pred = _predict_raw(fitted)
    baseline = _baseline_raw(fitted)
    rows: list[dict[str, float | str | int]] = []
    for i, col in enumerate(fitted.state_columns):
        err = pred[i] - fitted.y_raw[i]
        base_err = baseline[i] - fitted.y_raw[i]
        rmse = float(np.sqrt(np.mean(err**2)))
        mae = float(np.mean(np.abs(err)))
        base_rmse = float(np.sqrt(np.mean(base_err**2)))
        base_mae = float(np.mean(np.abs(base_err)))
        rows.append(
            {
                "name": fitted.name,
                "method": fitted.method,
                "state_mode": fitted.state_mode,
                "state_column": col,
                "source_section": fitted.source_section,
                "target_section": fitted.target_section,
                "sample_count": int(fitted.x_raw.shape[1]),
                "rmse": rmse,
                "mae": mae,
                "baseline_rmse": base_rmse,
                "baseline_mae": base_mae,
                "rmse_improvement_vs_baseline": (base_rmse - rmse) / base_rmse if base_rmse > 0 else np.nan,
                "mae_improvement_vs_baseline": (base_mae - mae) / base_mae if base_mae > 0 else np.nan,
            }
        )
    return rows


def _composition_metrics(
    *,
    name: str,
    section: str,
    state_mode: str,
    state_columns: list[str],
    A: np.ndarray,
    scaler: Scaler,
    x_df: pd.DataFrame,
    y_df: pd.DataFrame,
) -> list[dict[str, float | str | int]]:
    x_raw = _state_matrix(x_df, state_columns)
    y_raw = _state_matrix(y_df, state_columns)
    pred = scaler.inverse_transform(A @ scaler.transform(x_raw))
    baseline = np.zeros_like(y_raw)
    for i, col in enumerate(state_columns):
        baseline[i] = x_raw[i] if col == "log_price" else 0.0

    rows = []
    for i, col in enumerate(state_columns):
        err = pred[i] - y_raw[i]
        base_err = baseline[i] - y_raw[i]
        rmse = float(np.sqrt(np.mean(err**2)))
        mae = float(np.mean(np.abs(err)))
        base_rmse = float(np.sqrt(np.mean(base_err**2)))
        base_mae = float(np.mean(np.abs(base_err)))
        rows.append(
            {
                "name": name,
                "method": "composition_map",
                "state_mode": state_mode,
                "state_column": col,
                "source_section": section,
                "target_section": section,
                "sample_count": int(x_raw.shape[1]),
                "rmse": rmse,
                "mae": mae,
                "baseline_rmse": base_rmse,
                "baseline_mae": base_mae,
                "rmse_improvement_vs_baseline": (base_rmse - rmse) / base_rmse if base_rmse > 0 else np.nan,
                "mae_improvement_vs_baseline": (base_mae - mae) / base_mae if base_mae > 0 else np.nan,
            }
        )
    return rows


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


def _normalise_vector_pair(v1: np.ndarray, v2: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    max_abs = float(np.max(np.abs(v1)))
    if v2 is not None:
        max_abs = max(max_abs, float(np.max(np.abs(v2))))
    if max_abs <= 0:
        return v1, v2
    return v1 / max_abs, None if v2 is None else v2 / max_abs


def _mode_rows(
    *,
    name: str,
    method: str,
    state_mode: str,
    state_columns: list[str],
    section: str,
    A: np.ndarray,
    counterpart_section: str | None = None,
    counterpart_A: np.ndarray | None = None,
) -> list[dict[str, float | str | int]]:
    eigs, vecs = np.linalg.eig(A)
    order = np.argsort(-np.abs(eigs))
    rows: list[dict[str, float | str | int]] = []
    for out_id, mode_id in enumerate(order, start=1):
        lam = complex(eigs[mode_id])
        v = vecs[:, mode_id]
        counterpart = counterpart_A @ v if counterpart_A is not None else None
        v_norm, counterpart_norm = _normalise_vector_pair(v, counterpart)
        angle = float(np.angle(lam))
        row: dict[str, float | str | int] = {
            "name": name,
            "method": method,
            "state_mode": state_mode,
            "section": section,
            "mode_rank": out_id,
            "mode_id": int(mode_id),
            "lambda_real": float(lam.real),
            "lambda_imag": float(lam.imag),
            "lambda_abs": float(abs(lam)),
            "lambda_angle": angle,
            "period_sections": float(2 * np.pi / abs(angle)) if abs(angle) > 1e-12 else np.inf,
            "dynamic_label": _dynamic_label(lam),
        }
        for i, col in enumerate(state_columns):
            row[f"section_{col}_real"] = float(v_norm[i].real)
            row[f"section_{col}_imag"] = float(v_norm[i].imag)
        if counterpart_norm is not None and counterpart_section is not None:
            row["counterpart_section"] = counterpart_section
            for i, col in enumerate(state_columns):
                row[f"counterpart_{col}_real"] = float(counterpart_norm[i].real)
                row[f"counterpart_{col}_imag"] = float(counterpart_norm[i].imag)
        rows.append(row)
    return rows


def _latest_score_rows(
    *,
    name: str,
    method: str,
    state_mode: str,
    state_columns: list[str],
    section: str,
    A: np.ndarray,
    scaler: Scaler,
    latest_row: pd.Series,
) -> list[dict[str, float | str | int]]:
    eigs, vecs = np.linalg.eig(A)
    z = scaler.transform(_state_matrix(pd.DataFrame([latest_row]), state_columns)).reshape(-1)
    coeffs = np.linalg.pinv(vecs) @ z
    rows = []
    for mode_id, coeff in enumerate(coeffs):
        lam = complex(eigs[mode_id])
        row: dict[str, float | str | int] = {
            "name": name,
            "method": method,
            "state_mode": state_mode,
            "section": section,
            "event_id": str(latest_row["event_id"]),
            "date": str(latest_row["date"]),
            "bar_index": int(latest_row["bar_index"]),
            "mode_id": int(mode_id),
            "lambda_real": float(lam.real),
            "lambda_imag": float(lam.imag),
            "coefficient_real": float(complex(coeff).real),
            "coefficient_imag": float(complex(coeff).imag),
            "coefficient_abs": float(abs(coeff)),
        }
        for col in state_columns:
            row[col] = float(latest_row[col])
        rows.append(row)
    return rows


def _operator_rows(*, name: str, method: str, state_mode: str, state_columns: list[str], source_section: str, target_section: str, A: np.ndarray) -> list[dict[str, float | str]]:
    rows = []
    for i, target_col in enumerate(state_columns):
        for j, source_col in enumerate(state_columns):
            rows.append(
                {
                    "name": name,
                    "method": method,
                    "state_mode": state_mode,
                    "source_section": source_section,
                    "target_section": target_section,
                    "target_component": target_col,
                    "source_component": source_col,
                    "coefficient": float(A[i, j]),
                }
            )
    return rows


def run_experiment(*, observable_path: Path, out_dir: Path, rank: int, state_mode: str) -> dict[str, Path]:
    if state_mode not in STATE_MODES:
        raise ValueError(f"state_mode must be one of {sorted(STATE_MODES)}")
    state_columns = STATE_MODES[state_mode]
    events = _read_events(observable_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    high_x, high_y = _section_return_pairs(events, "high")
    low_x, low_y = _section_return_pairs(events, "low")
    two_x, two_y = _two_step_pairs(events)
    hl_x, hl_y = _phase_pairs(events, "high", "low")
    lh_x, lh_y = _phase_pairs(events, "low", "high")

    high_scaler = _fit_scaler(_state_matrix(pd.concat([high_x, high_y], ignore_index=True), state_columns))
    low_scaler = _fit_scaler(_state_matrix(pd.concat([low_x, low_y], ignore_index=True), state_columns))
    event_scaler = _fit_scaler(_state_matrix(pd.concat([two_x, two_y], ignore_index=True), state_columns))

    direct_high = _fit_map(name="return_high", method="same_section_return_map", source_section="high", target_section="high", state_mode=state_mode, state_columns=state_columns, x_df=high_x, y_df=high_y, rank=rank, x_scaler=high_scaler, y_scaler=high_scaler)
    direct_low = _fit_map(name="return_low", method="same_section_return_map", source_section="low", target_section="low", state_mode=state_mode, state_columns=state_columns, x_df=low_x, y_df=low_y, rank=rank, x_scaler=low_scaler, y_scaler=low_scaler)
    two_step = _fit_map(name="two_step_same_section", method="two_step_event_map", source_section="event", target_section="event_plus_2", state_mode=state_mode, state_columns=state_columns, x_df=two_x, y_df=two_y, rank=rank, x_scaler=event_scaler, y_scaler=event_scaler)
    hl = _fit_map(name="phase_high_to_low", method="phase_map", source_section="high", target_section="low", state_mode=state_mode, state_columns=state_columns, x_df=hl_x, y_df=hl_y, rank=rank, x_scaler=high_scaler, y_scaler=low_scaler)
    lh = _fit_map(name="phase_low_to_high", method="phase_map", source_section="low", target_section="high", state_mode=state_mode, state_columns=state_columns, x_df=lh_x, y_df=lh_y, rank=rank, x_scaler=low_scaler, y_scaler=high_scaler)

    comp_high_A = lh.A @ hl.A
    comp_low_A = hl.A @ lh.A

    summary_rows: list[dict[str, float | str | int]] = []
    for item in [direct_high, direct_low, two_step, hl, lh]:
        summary_rows.extend(_metric_rows(item))
    summary_rows.extend(_composition_metrics(name="composition_high_return", section="high", state_mode=state_mode, state_columns=state_columns, A=comp_high_A, scaler=high_scaler, x_df=high_x, y_df=high_y))
    summary_rows.extend(_composition_metrics(name="composition_low_return", section="low", state_mode=state_mode, state_columns=state_columns, A=comp_low_A, scaler=low_scaler, x_df=low_x, y_df=low_y))

    mode_rows: list[dict[str, float | str | int]] = []
    mode_rows.extend(_mode_rows(name="return_high", method="same_section_return_map", state_mode=state_mode, state_columns=state_columns, section="high", A=direct_high.A))
    mode_rows.extend(_mode_rows(name="return_low", method="same_section_return_map", state_mode=state_mode, state_columns=state_columns, section="low", A=direct_low.A))
    mode_rows.extend(_mode_rows(name="two_step_same_section", method="two_step_event_map", state_mode=state_mode, state_columns=state_columns, section="event", A=two_step.A))
    mode_rows.extend(_mode_rows(name="composition_high_return", method="composition_map", state_mode=state_mode, state_columns=state_columns, section="high", A=comp_high_A, counterpart_section="low", counterpart_A=hl.A))
    mode_rows.extend(_mode_rows(name="composition_low_return", method="composition_map", state_mode=state_mode, state_columns=state_columns, section="low", A=comp_low_A, counterpart_section="high", counterpart_A=lh.A))

    latest_high = events[events["extreme_type"] == "high"].iloc[-1]
    latest_low = events[events["extreme_type"] == "low"].iloc[-1]
    latest_event = events.iloc[-1]
    latest_rows: list[dict[str, float | str | int]] = []
    latest_rows.extend(_latest_score_rows(name="return_high", method="same_section_return_map", state_mode=state_mode, state_columns=state_columns, section="high", A=direct_high.A, scaler=high_scaler, latest_row=latest_high))
    latest_rows.extend(_latest_score_rows(name="return_low", method="same_section_return_map", state_mode=state_mode, state_columns=state_columns, section="low", A=direct_low.A, scaler=low_scaler, latest_row=latest_low))
    latest_rows.extend(_latest_score_rows(name="two_step_same_section", method="two_step_event_map", state_mode=state_mode, state_columns=state_columns, section="event", A=two_step.A, scaler=event_scaler, latest_row=latest_event))
    latest_rows.extend(_latest_score_rows(name="composition_high_return", method="composition_map", state_mode=state_mode, state_columns=state_columns, section="high", A=comp_high_A, scaler=high_scaler, latest_row=latest_high))
    latest_rows.extend(_latest_score_rows(name="composition_low_return", method="composition_map", state_mode=state_mode, state_columns=state_columns, section="low", A=comp_low_A, scaler=low_scaler, latest_row=latest_low))

    operator_rows: list[dict[str, float | str]] = []
    for item in [direct_high, direct_low, two_step, hl, lh]:
        operator_rows.extend(_operator_rows(name=item.name, method=item.method, state_mode=state_mode, state_columns=state_columns, source_section=item.source_section, target_section=item.target_section, A=item.A))
    operator_rows.extend(_operator_rows(name="composition_high_return", method="composition_map", state_mode=state_mode, state_columns=state_columns, source_section="high", target_section="high", A=comp_high_A))
    operator_rows.extend(_operator_rows(name="composition_low_return", method="composition_map", state_mode=state_mode, state_columns=state_columns, source_section="low", target_section="low", A=comp_low_A))

    paths = {
        "summary": out_dir / "map_fit_summary.csv",
        "operators": out_dir / "operators.csv",
        "eigenmodes": out_dir / "eigenmodes.csv",
        "latest_scores": out_dir / "latest_state_scores.csv",
        "config": out_dir / "config.json",
    }
    pd.DataFrame(summary_rows).to_csv(paths["summary"], index=False, encoding="utf-8-sig")
    pd.DataFrame(operator_rows).to_csv(paths["operators"], index=False, encoding="utf-8-sig")
    pd.DataFrame(mode_rows).to_csv(paths["eigenmodes"], index=False, encoding="utf-8-sig")
    pd.DataFrame(latest_rows).to_csv(paths["latest_scores"], index=False, encoding="utf-8-sig")
    paths["config"].write_text(
        json.dumps(
            {
                "observable_path": str(observable_path),
                "rank_requested": rank,
                "state_mode": state_mode,
                "state_columns": state_columns,
                "state_definition": {
                    "price": "state=[log_price]",
                    "change": "state=[same_type_change], HH for high and LL for low",
                },
                "methods": [
                    "same-section return maps: High->High and Low->Low",
                    "two-step event map: x_k -> x_{k+2}",
                    "composition maps: High->Low, Low->High, then High->High and Low->Low products",
                ],
                "standardization": "Each map is fitted in z-scored coordinates for the selected input only.",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run interpretable extremum-map DMD experiments.")
    parser.add_argument("--observable-path", required=True)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--state-mode", choices=sorted(STATE_MODES), required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    observable_path = Path(args.observable_path)
    out_dir = Path(args.out_dir) if args.out_dir else observable_path.parent / f"extremum_map_dmd_{args.state_mode}"
    paths = run_experiment(observable_path=observable_path, out_dir=out_dir, rank=args.rank, state_mode=args.state_mode)
    print(f"output_dir: {out_dir}")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
