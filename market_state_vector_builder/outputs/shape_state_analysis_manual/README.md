# Manual Market State Vector

This is the current main workflow. It does not use PCA.

## State Variables

- `range_pct`: structural width in log-price space.
- `expansion`: width change, equal to `high_change - low_change`.
- `structure_dir`: continuous structural direction, equal to `high_change + low_change`.
- `range_pos`: current position in the static structural range.
- `upper_break`: 1 if close is above the dynamic upper line, otherwise 0.
- `lower_break`: 1 if close is below the dynamic lower line, otherwise 0.
- `move_age`: bars since the latest structural-window endpoint, scaled by recent median extremum gap.
- `extrema_freq`: confirmed-extrema frequency inside the structural window.
- `speed_imbalance`: `high_speed - low_speed`.
- `speed_level`: `high_speed + low_speed`.

## Scores

Continuous `*_score` columns use each stock's prior 1000 valid states to estimate
rolling 5%/95% quantiles and map the current value to `[-1, 1]`. The current row
is excluded from its own quantile estimate. `upper_break_score` and
`lower_break_score` use fixed binary mapping: `0 -> -1`, `1 -> 1`.
