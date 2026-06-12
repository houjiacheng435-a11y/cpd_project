# 06 Lowess Analysis

This step analyzes the final manual market-state vector with LOWESS smoothing.

Inputs:

- `market_state_vector_builder/outputs/shape_state_analysis_manual/final_state_vector.parquet`

Outputs:

- `market_state_vector_builder/outputs/shape_state_analysis_manual/06_lowess/`

Per target subfolder:

- `return_k05/`
- `return_k10/`
- `return_k20/`
- `vol_k05/`
- `vol_k10/`
- `vol_k20/`

Each subfolder contains one figure per manual state variable plus an `index.csv`.

Binary state variables `upper_break` and `lower_break` are excluded from LOWESS plots.
By default, the x-axis uses the corresponding `*_score` columns; raw state columns are optional.

Run:

```powershell
python market_state_vector_builder/06_lowess/analyze_manual_lowess.py
```
