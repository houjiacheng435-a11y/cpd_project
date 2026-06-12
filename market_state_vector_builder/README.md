# Market State Vector Builder

Current main workflow: manual market-state vector, no PCA.

Generated results should go under `market_state_vector_builder/outputs/`.
Project notes are under `market_state_vector_builder/Note/`.

Recent design iteration notes:

- `manual_state_iteration_notes.md` under the practice notes directory.

## 01_data

- `download_a_share_akshare.py`: download A-share daily OHLCV CSV files.

## 02_build_state

- `build_manual_state.py`: build `shape_state_variables.parquet` and the final manual `final_state_vector.parquet`.
- `shape_state_variables.py`: online structural-window variable construction used by `build_manual_state.py`.

Main output variables:

- `range_pct`
- `expansion`
- `structure_dir`
- `range_pos`
- `upper_break`
- `lower_break`
- `move_age`
- `extrema_freq`
- `speed_imbalance`
- `speed_level`

## 03_examples

- `plot_examples.py`: plot K-line examples with structural extrema and manual state bars.

## 04_validation

- `analyze_manual_adf.py`: rolling ADF summary by manual-state quantile.

## 05_explosive

- `export_manual_exuber_windows.py`: sample windows for R `exuber`.
- `run_exuber_window_sample.R`: run GSADF/BSADF window tests in R.
- `analyze_manual_exuber_results.py`: summarize explosive-test results by manual-state quantile and composite state.

## 06_lowess

- `analyze_manual_lowess.py`: LOWESS smoothing of each manual variable against future log-return and future volatility targets.
