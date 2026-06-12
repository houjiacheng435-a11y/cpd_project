# CPD Project

This repository contains a simple Python package for change point detection.

## Project structure

- `cpd/`
  - `__init__.py` - package exports all public functions
  - `brk_detector.py` - main `Brk` detector class and implementations
  - `wbs_lepage.py` - WBS-Lepage nonparametric detector
  - `utils.py` - input validation utilities
  - `multi_detect.py` - batch detection with multiple methods and parameters
  - `visualization.py` - plotting functions for results and heatmaps
- `notebooks/`
  - `test_CPD.ipynb` - example notebook showing basic usage
- `data/`
  - `TEST.parquet(1).gzip` - sample dataset
- `example.py` - basic single-method example
- `example_multi.py` - multi-method comparison with Jaccard heatmap
- `example_market_state.ipynb` - local CSV market state vector example
- `requirements.txt` - Python dependencies
- `.gitignore` - files to ignore in Git

## How to use

### 1. Run example scripts

From the `cpd_project` directory:

**Single method example:**

```bash
python example.py
```

Loads data and runs CUSUM detection, prints results.

**Multiple methods with comparison:**

```bash
python example_multi.py
```

Runs multiple detection methods, prints summary, and shows Jaccard similarity heatmap.

### 2. Import the package

```python
from cpd import Brk, generate_breaks

# series must be a pandas Series with a DatetimeIndex
brk = Brk(method='cusum', q=1.0)
change_points = brk.detect(series)
```

### 3. Use the notebook

Open `notebooks/test_CPD.ipynb` in Jupyter. If the notebook is opened from `cpd_project`, it can import the package directly using:

```python
from cpd import Brk
```

### Input requirements

- The input must be a `pandas.Series`.
- The series index must be a `DatetimeIndex`.
- The sample notebook expects the dataset column named `p`.

## Available detection methods

Use `Brk(method='...')` with one of these names:

- `cusum`
- `cusum_ls`
- `sprt`
- `gma`
- `glr`
- `brandt_glr`
- `shewhart`
- `recursive_param_seg`
- `e_detector`
- `ssr_cusum`
- `adaptive_cusum`
- `wbs_lepage` (alias: `wbs`)

Note: `wbs_lepage` is a nonparametric Wild Binary Segmentation implementation. It can be computationally intensive on large datasets or with a large `M` setting; reduce `M` for faster runs.

### Sensitivity parameter `q`

All methods support the `q` parameter to adjust sensitivity:

- `q > 1.0` - less sensitive, fewer change points detected
- `q = 1.0` - default sensitivity
- `q < 1.0` - more sensitive, more change points detected

Example:

```python
brk = Brk(method='cusum', q=0.5)  # More sensitive
change_points = brk.detect(series)
```

### Detecting change points on new data

To detect change points on your own dataset:

1. Load your data as a `pandas.Series` with a `DatetimeIndex`:

```python
import pandas as pd
from cpd import Brk

# Load your data
df = pd.read_parquet('your_data.parquet')  # or CSV, etc.
series = df['your_column']  # Select your data column

# Ensure index is DatetimeIndex
if not isinstance(series.index, pd.DatetimeIndex):
    series.index = pd.to_datetime(series.index)  # or set from a date column
```

2. Create a detector and run detection:

```python
brk = Brk(method='cusum', q=1.0)  # Adjust method and q as needed
change_points = brk.detect(series)

print(f'Found {len(change_points)} change points:')
for cp in change_points:
    print(cp)
```

Call `Brk.detect(series)` to get a list of detected change point timestamps.

### 2.1 Market state vector builder

The current main workflow builds a manual, interpretable market-state vector. PCA is not used as the final state representation. The core goal is:

```text
raw daily OHLCV data -> final_state_vector.parquet
```

Here, "realtime" means that the state for a given date does not use data after that date. In practice, you can pass the full history available up to the current date and run the script once. The CPD confirmation, structural window construction, and rolling scores are all computed with an online data convention.

**Input data**

Put one CSV file per stock under an input directory, for example:

```text
data/a_share_1d_akshare/symbols/
```

Each CSV should contain daily OHLC data. The current workflow uses log price from the CPD stage onward, so keep `--log-price` enabled for the main run.

**Core command**

```bash
python market_state_vector_builder/02_build_state/build_manual_state.py ^
  --input-dir data/a_share_1d_akshare/symbols ^
  --out-dir market_state_vector_builder/outputs/shape_state_analysis_manual ^
  --max-symbols 80 ^
  --start 2010-01-01 ^
  --end 2026-05-29 ^
  --log-price ^
  --detector-method cusum ^
  --detector-q 1.0 ^
  --cpd-confirm-lag 0 ^
  --n-lags 0 ^
  --score-window 1000 ^
  --score-min-periods 100
```

The only core output needed for downstream use is:

```text
market_state_vector_builder/outputs/shape_state_analysis_manual/final_state_vector.parquet
```

Other files, such as `shape_state_variables.parquet`, `manual_state_summary.csv`, `manual_state_correlation.csv`, `manual_state_score_config.csv`, and `manual_state_correlation_heatmap.png`, are intermediate files, documentation outputs, or diagnostic checks. They are not the final state vector itself.

`shape_state_variables.parquet` is the main intermediate table. It keeps the online CPD diagnostics and the structural-window variables before the final manual state table is trimmed down. With `--log-price`, the OHLC and structural price-level columns are in log-price space.

| Column | Meaning |
|---|---|
| `stock_id` | Six-digit stock identifier. |
| `date` | Bar date. |
| `open` | Open price used by the state builder; log open when `--log-price` is enabled. |
| `high` | High price used by the state builder; log high when `--log-price` is enabled. |
| `low` | Low price used by the state builder; log low when `--log-price` is enabled. |
| `close` | Close price used by the state builder; log close when `--log-price` is enabled. |
| `num_confirmed_extrema` | Number of online-confirmed extrema accumulated up to this bar. |
| `has_structural_window` | Whether at least four alternating confirmed extrema are available. |
| `volume` | Raw trading volume from the input file, if available. |
| `amount` | Raw traded amount from the input file, if available. |
| `starts_high` | `1` if the latest four-extremum window starts with a high, otherwise `0`. |
| `high1` | Earlier high in the latest four-extremum window. |
| `high2` | Later high in the latest four-extremum window. |
| `low1` | Earlier low in the latest four-extremum window. |
| `low2` | Later low in the latest four-extremum window. |
| `high_change` | Change from `high1` to `high2`. |
| `low_change` | Change from `low1` to `low2`. |
| `expansion` | Width change: `high_change - low_change`. |
| `range_pos` | Current close position inside the static high-low structural range. |
| `upper_break` | `1` if close is above the dynamic upper line, otherwise `0`. |
| `lower_break` | `1` if close is below the dynamic lower line, otherwise `0`. |
| `structure_dir` | Structural direction: `high_change + low_change`. |
| `range_pct` | Static structural width: `max(high1, high2) - min(low1, low2)`. |
| `last_width_pct` | Width between the later high and later low: `high2 - low2`. |
| `move_age` | Bars since the latest structural-window endpoint, scaled by recent median extremum gap. |
| `high_speed` | Absolute high-point change per bar between `high1` and `high2`. |
| `low_speed` | Absolute low-point change per bar between `low1` and `low2`. |
| `extrema_freq` | Confirmed-extrema frequency inside the window: `4 / window_bars`. |
| `direction` | Online direction signal from `MarketStateVector`. |
| `excep` | Online exception flag from `MarketStateVector`. |
| `status` | Online status label from `MarketStateVector`. |
| `current_trend` | Current trend label from `MarketStateVector`. |
| `current_trend_code` | Numeric current-trend code from `MarketStateVector`. |
| `current_phase` | Current phase label from `MarketStateVector`. |
| `trend_candidate` | Current trend-candidate label from `MarketStateVector`. |
| `new_extremum_confirmed` | Whether this bar confirms a new extremum. |
| `confirmed_extremum_kind` | Confirmed extremum type: `high` or `low`. |
| `confirmed_extremum_idx` | Original bar index of the confirmed extremum. |
| `confirmed_extremum_value` | Price value of the confirmed extremum. |
| `cpd_confirm_event` | Whether a CPD event is confirmed on this bar. |
| `time_since_last_cpd` | Bars since the last raw CPD event. |
| `time_since_last_cpd_confirm` | Bars since the last confirmed CPD event. |

If `shape_state_variables.parquet` already exists and you only want to regenerate the final state vector and scores, reuse it without rerunning CPD:

```bash
python market_state_vector_builder/02_build_state/build_manual_state.py ^
  --state-path market_state_vector_builder/outputs/shape_state_analysis_manual/shape_state_variables.parquet ^
  --out-dir market_state_vector_builder/outputs/shape_state_analysis_manual ^
  --log-price ^
  --score-window 1000 ^
  --score-min-periods 100
```

**What is inside final_state_vector.parquet**

The final table contains one row per stock-date state. The main variables are:

| Variable | Meaning |
|---|---|
| `range_pct` | Structural log-width: `log(upper_ref) - log(lower_ref)`. |
| `expansion` | Width change: `high_change - low_change`; positive means expansion, negative means contraction. |
| `structure_dir` | Continuous structural direction: `high_change + low_change`; positive means highs and lows move up overall. |
| `range_pos` | Current price position inside the static structural range. |
| `upper_break` | `1` if the current price is above the dynamic upper line, otherwise `0`. |
| `lower_break` | `1` if the current price is below the dynamic lower line, otherwise `0`. |
| `move_age` | Time since the latest structural-window endpoint, scaled by the recent median extremum gap. |
| `extrema_freq` | Confirmed-extrema frequency inside the structural window: `4 / window_bars`. |
| `speed_imbalance` | Upper/lower boundary speed imbalance: `high_speed - low_speed`. |
| `speed_level` | Overall boundary speed level: `high_speed + low_speed`. |

Each variable also has a matching `*_score` column. For continuous variables, the score uses each stock's prior 1000 valid states to estimate rolling 5%/95% quantiles, then maps the current value to `[-1, 1]`. The current row is excluded from its own quantile estimate, so the score does not use future data. `upper_break_score` and `lower_break_score` use a fixed binary mapping: `0 -> -1`, `1 -> 1`.

**Optional visual check**

After generating the final state vector, you can plot a few K-line examples to inspect structural extrema and state values manually:

```bash
python market_state_vector_builder/03_examples/plot_examples.py ^
  --final-state-path market_state_vector_builder/outputs/shape_state_analysis_manual/final_state_vector.parquet ^
  --state-path market_state_vector_builder/outputs/shape_state_analysis_manual/shape_state_variables.parquet ^
  --out-dir market_state_vector_builder/outputs/shape_state_analysis_manual/examples ^
  --n-examples 10 ^
  --state-price-is-log
```

ADF and explosive tests are downstream research checks. They are not required to generate the realtime state vector.

## Visualization

### Plot change points on time series

After detection, visualize results:

```python
from cpd import Brk, plot_change_points

brk = Brk(method='cusum', q=1.0)
change_points = brk.detect(series)

# Plot the time series with detected change points marked as vertical lines
plot_change_points(series, change_points, method='cusum', q=1.0)
```

### Multiple method comparison with Jaccard heatmap

To compare multiple detection methods and parameters:

```python
from cpd import detect_multiple, print_detection_summary, plot_jaccard_heatmap

# Define multiple configurations: (method, q) tuples
configs = [
    ('cusum', 0.5),
    ('cusum', 1.0),
    ('cusum', 1.5),
    ('glr', 1.0),
    ('adaptive_cusum', 1.0),
]

# Run all detections
results = detect_multiple(series, configs)

# Print summary of detection counts
print_detection_summary(results)

# Plot Jaccard similarity matrix as heatmap
# Jaccard similarity measures how similar the detected change points are between methods
plot_jaccard_heatmap(results)
```

This generates:

- A printed summary showing how many change points each method found
- A heatmap showing Jaccard similarity (0 = no overlap, 1 = identical) between all method pairs

### Run multiple methods example

Try the example script that demonstrates all features:

```bash
python example_multi.py
```

This loads data, runs multiple detection methods, prints results, and shows a Jaccard heatmap.

**Market state vector example:**

```bash
jupyter notebook example_market_state.ipynb
```

This loads local OHLC data from `data/spx_1d.csv`, builds the online market
state vector, and prints evaluation summaries.
