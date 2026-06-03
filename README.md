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

### 2.1 Build an online market state vector

The package also exposes a separate market-state layer that uses the existing
change-point detectors as online prefix-confirmation signals:

```python
import numpy as np

from cpd import (
    load_data,
    build_market_state_vector,
    get_core_state_vector,
    evaluate_state_vector,
)

# Put OHLC CSV files under data/ using the name {symbol}_{interval}.csv.
# For example, data/spx_1d.csv can be loaded with symbol="spx", interval="1d".
raw_df = load_data(symbol="spx", interval="1d")

# builds the state vector on log OHLC prices.
df = np.log(raw_df)

# Full state vector for every timestamp in df.
state_df = build_market_state_vector(data=df, detector_method="cusum", detector_q=1.0)

# Compact 6-column core market state vector for every timestamp in state_df.
core_state_df = get_core_state_vector(state_df)

evaluation = evaluate_state_vector(state_df, price_is_log=True)
```

`state_df` includes the requested online features such as `direction`,
`status`, `current_trend`, `position`, `distance_to_point2`,
`distance_to_point3`, `time_since_last_cpd`, `regime_slope`, and
`regime_volatility`. The `cpd_event` flag marks the bar on which a change
point is first confirmed online.

`core_state_df` keeps the same index as `state_df` and returns the core market
state variables for all available timestamps:

```python
[
    "current_trend",
    "current_phase",
    "position",
    "regime_slope",
    "regime_volatility",
    "dist_point3_pct",
]
```

`dist_point3_pct` is computed as `distance_to_point3 / Close`; missing values,
zero `Close`, and infinite results are returned as `NaN`.

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
