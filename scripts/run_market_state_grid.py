from __future__ import annotations

import json
import multiprocessing as mp
import sys
import time
import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cpd import (
    build_market_state_vector,
    evaluate_state_vector,
    get_core_state_vector,
    load_data,
)


PARAM_GRID: dict[str, dict[str, float]] = {
    "cusum": {
        "about_4000_cp": 0.20,
        "about_2000_cp": 0.30,
        "about_1000_cp": 0.50,
        "about_500_cp": 0.70,
    },
    "cusum_ls": {
        "about_4000_cp": 0.10,
        "about_2000_cp": 0.50,
        "about_1000_cp": 1.00,
        "about_500_cp": 1.50,
    },
    "sprt": {
        "about_4000_cp": 0.35,
        "about_2000_cp": 0.50,
        "about_1000_cp": 0.70,
        "about_500_cp": 1.00,
    },
    "gma": {
        "about_4000_cp": 0.30,
        "about_2000_cp": 0.40,
        "about_1000_cp": 0.50,
        "about_500_cp": 0.60,
    },
    "glr": {
        "about_4000_cp": 0.30,
        "about_2000_cp": 0.70,
        "about_1000_cp": 1.20,
        "about_500_cp": 1.80,
    },
    "brandt_glr": {
        "about_4000_cp": 0.55,
        "about_2000_cp": 0.66,
        "about_1000_cp": 0.73,
        "about_500_cp": 0.78,
    },
    "shewhart": {
        "about_4000_cp": 0.16,
        "about_2000_cp": 0.30,
        "about_1000_cp": 0.50,
        "about_500_cp": 0.70,
    },
    "recursive_param_seg": {
        "about_4000_cp": 1.50,
        "about_2000_cp": 2.50,
        "about_1000_cp": 3.00,
        "about_500_cp": 3.60,
    },
    "e_detector": {
        "about_4000_cp": 0.21,
        "about_2000_cp": 0.34,
        "about_1000_cp": 0.53,
        "about_500_cp": 0.88,
    },
    "ssr_cusum": {
        "about_4000_cp": 0.50,
        "about_2000_cp": 0.60,
        "about_1000_cp": 0.80,
        "about_500_cp": 1.00,
    },
    "adaptive_cusum": {
        "about_4000_cp": 0.32,
        "about_2000_cp": 0.44,
        "about_1000_cp": 0.55,
        "about_500_cp": 0.71,
    },
}


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, float) and np.isnan(value):
        return None
    return str(value)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )


def _write_evaluation(evaluation: dict[str, Any], output_dir: Path) -> None:
    eval_dir = output_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    evaluation["state_duration_table"].to_csv(eval_dir / "state_duration_table.csv", index=False)
    evaluation["state_duration_summary"].to_csv(eval_dir / "state_duration_summary.csv")
    evaluation["phase_duration_table"].to_csv(eval_dir / "phase_duration_table.csv", index=False)
    evaluation["phase_duration_summary"].to_csv(eval_dir / "phase_duration_summary.csv")
    evaluation["transition_matrix"].to_csv(eval_dir / "transition_matrix.csv")
    evaluation["phase_transition_matrix"].to_csv(eval_dir / "phase_transition_matrix.csv")
    _write_json(eval_dir / "switch_counts.json", evaluation["switch_counts"])

    position_dir = eval_dir / "position_future_returns"
    position_dir.mkdir(exist_ok=True)
    for horizon, table in evaluation["position_future_returns"].items():
        table.to_csv(position_dir / f"horizon_{horizon}.csv")

    trend_dir = eval_dir / "trend_future_returns"
    trend_dir.mkdir(exist_ok=True)
    for horizon, table in evaluation["trend_future_returns"].items():
        table.to_csv(trend_dir / f"horizon_{horizon}.csv")

    phase_dir = eval_dir / "phase_future_returns"
    phase_dir.mkdir(exist_ok=True)
    for horizon, table in evaluation["phase_future_returns"].items():
        table.to_csv(phase_dir / f"horizon_{horizon}.csv")

    evaluation["point23_interval_table"].to_csv(eval_dir / "point23_interval_table.csv", index=False)
    evaluation["point23_interval_summary"].to_csv(eval_dir / "point23_interval_summary.csv")


def _run_combo_worker(method: str, target_label: str, q: float, combo_dir_str: str) -> None:
    combo_dir = Path(combo_dir_str)

    raw_df = load_data(symbol="spx", interval="1d")
    if (raw_df <= 0).any().any():
        raise ValueError("OHLC prices must be positive before taking logs.")
    data = np.log(raw_df)

    state_df = build_market_state_vector(
        data=data,
        detector_method=method,
        detector_q=q,
    )
    core_df = get_core_state_vector(state_df)
    evaluation = evaluate_state_vector(state_df, price_is_log=True)

    core_df.to_csv(combo_dir / "core_df.csv", index=True)
    _write_evaluation(evaluation, combo_dir)

    result = {
        "rows": int(len(core_df)),
        "core_columns": list(core_df.columns),
        "cpd_confirm_event_count": int(
            state_df.get("cpd_confirm_event", pd.Series(False, index=state_df.index)).sum()
        ),
        "position_notna_count": int(core_df["position"].notna().sum()),
    }
    _write_json(combo_dir / "result_summary.json", result)


def _read_existing_metadata(combo_dir: Path) -> dict[str, Any] | None:
    metadata_path = combo_dir / "metadata.json"
    if not metadata_path.exists():
        return None
    try:
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _run_combo_with_timeout(
    method: str,
    target_label: str,
    q: float,
    combo_dir: Path,
    output_root: Path,
    timeout_seconds: int,
) -> dict[str, Any]:
    combo_dir.mkdir(parents=True, exist_ok=True)
    combo_rel = str(combo_dir.relative_to(output_root))

    existing = _read_existing_metadata(combo_dir)
    if existing is not None and existing.get("status") == "ok" and (combo_dir / "core_df.csv").exists():
        existing["skipped_existing"] = True
        return existing

    row: dict[str, Any] = {
        "method": method,
        "target_label": target_label,
        "q": q,
        "combo_dir": combo_rel,
        "status": "running",
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "timeout_seconds": timeout_seconds,
    }
    _write_json(combo_dir / "metadata.json", row)

    start = time.perf_counter()
    ctx = mp.get_context("spawn")
    process = ctx.Process(
        target=_run_combo_worker,
        args=(method, target_label, q, str(combo_dir)),
    )
    process.start()
    process.join(timeout_seconds)

    elapsed = time.perf_counter() - start
    if process.is_alive():
        process.terminate()
        process.join(15)
        if process.is_alive():
            process.kill()
            process.join(15)
        row.update(
            {
                "status": "timeout",
                "finished_at": datetime.now().isoformat(timespec="seconds"),
                "elapsed_seconds": round(elapsed, 3),
                "error": f"Timed out after {timeout_seconds} seconds",
            }
        )
    elif process.exitcode == 0:
        result_path = combo_dir / "result_summary.json"
        result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {}
        row.update(
            {
                "status": "ok",
                "finished_at": datetime.now().isoformat(timespec="seconds"),
                "elapsed_seconds": round(elapsed, 3),
                **result,
            }
        )
    else:
        row.update(
            {
                "status": "error",
                "finished_at": datetime.now().isoformat(timespec="seconds"),
                "elapsed_seconds": round(elapsed, 3),
                "error": f"Worker exited with code {process.exitcode}",
            }
        )

    _write_json(combo_dir / "metadata.json", row)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--combo-timeout", type=int, default=600)
    args = parser.parse_args()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = args.output_root or PROJECT_ROOT / "outputs" / f"market_state_grid_{run_id}"
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    raw_df = load_data(symbol="spx", interval="1d")
    if (raw_df <= 0).any().any():
        raise ValueError("OHLC prices must be positive before taking logs.")
    data = np.log(raw_df)

    config = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "symbol": "spx",
        "interval": "1d",
        "input_file": "data/spx_1d.csv",
        "price_transform": "np.log(raw_df)",
        "evaluation_price_is_log": True,
        "evaluation_horizons": [5],
        "combo_timeout_seconds": args.combo_timeout,
        "param_grid": PARAM_GRID,
    }
    _write_json(output_root / "run_config.json", config)

    manifest_rows: list[dict[str, Any]] = []
    total = sum(len(qs) for qs in PARAM_GRID.values())
    completed = 0

    print(f"Output root: {output_root}", flush=True)
    print(f"Total combinations: {total}", flush=True)

    for method, q_by_target in PARAM_GRID.items():
        for target_label, q in q_by_target.items():
            completed += 1
            combo_name = f"{method}__{target_label}__q_{q:.2f}".replace(".", "p")
            combo_dir = output_root / combo_name
            print(f"[{completed}/{total}] running {method}, {target_label}, q={q:.2f}", flush=True)
            row = _run_combo_with_timeout(
                method=method,
                target_label=target_label,
                q=q,
                combo_dir=combo_dir,
                output_root=output_root,
                timeout_seconds=args.combo_timeout,
            )
            if row.get("skipped_existing"):
                print(f"[{completed}/{total}] skipped existing {combo_name}", flush=True)
            elif row["status"] == "ok":
                print(f"[{completed}/{total}] done {combo_name} in {row['elapsed_seconds']:.1f}s", flush=True)
            else:
                print(f"[{completed}/{total}] {row['status'].upper()} {combo_name}: {row.get('error', '')}", flush=True)

            manifest_rows.append(row)
            pd.DataFrame(manifest_rows).to_csv(output_root / "manifest.csv", index=False)

    print(f"Finished. Manifest: {output_root / 'manifest.csv'}", flush=True)


if __name__ == "__main__":
    main()
