import pandas as pd
from pathlib import Path

from cpd import Brk


def load_series(path: str) -> pd.Series:
    df = pd.read_parquet(path)
    if 'p' not in df.columns:
        raise ValueError("The dataset must contain a 'p' column.")

    series = df['p']
    if not isinstance(series.index, pd.DatetimeIndex):
        if 'time' in df.columns:
            series.index = pd.to_datetime(df['time'])
        else:
            raise ValueError('Series must have a DatetimeIndex or a time column to use as index.')

    return series


def main() -> None:
    project_root = Path(__file__).resolve().parent
    path = project_root / 'data' / 'TEST.parquet(1).gzip'
    series = load_series(path)

    detector = Brk(method='cusum', q=1.0)
    change_points = detector.detect(series)

    print(f'Found {len(change_points)} change points:')
    for cp in change_points:
        print(cp)


if __name__ == '__main__':
    main()
