from typing import Optional

import pandas as pd


def validate_series(ds: pd.Series, column_name: Optional[str] = None) -> pd.Series:
    """Validate input data and return a datetime-indexed pandas Series."""
    if isinstance(ds, pd.DataFrame):
        if column_name is None:
            raise ValueError("If ds is a DataFrame, column_name must be provided")
        if column_name not in ds.columns:
            raise ValueError(f"Column '{column_name}' not found in DataFrame")
        ds = ds[column_name]

    if not isinstance(ds, pd.Series):
        raise TypeError("ds must be a pandas Series")
    if not isinstance(ds.index, pd.DatetimeIndex):
        raise ValueError("Series index must be datetime type")

    return ds
