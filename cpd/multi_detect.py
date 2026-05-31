"""Batch detection with multiple methods and parameters."""

from typing import List, Tuple, Dict
import pandas as pd

from .brk_detector import Brk
from .utils import validate_series


def detect_multiple(series: pd.Series, 
                   configs: List[Tuple[str, float]]) -> Dict[Tuple[str, float], List]:
    """
    Run change point detection with multiple method and q parameter combinations.
    
    Parameters
    ----------
    series : pd.Series
        Time series data with DatetimeIndex
    configs : List[Tuple[str, float]]
        List of (method, q) tuples. Example: [('cusum', 1.0), ('glr', 0.5), ('cusum', 0.5)]
    
    Returns
    -------
    dict
        Dictionary mapping (method, q) -> list of detected change points
    
    Example
    -------
    >>> configs = [('cusum', 1.0), ('cusum', 0.5), ('glr', 1.0)]
    >>> results = detect_multiple(series, configs)
    >>> for (method, q), cps in results.items():
    ...     print(f'{method} (q={q}): {len(cps)} change points')
    """
    series = validate_series(series)
    results = {}
    
    for method, q in configs:
        brk = Brk(method=method, q=q)
        change_points = brk.detect(series)
        results[(method, q)] = change_points
    
    return results


def print_detection_summary(results: Dict[Tuple[str, float], List]) -> None:
    """
    Print a summary of detection results from detect_multiple.
    
    Parameters
    ----------
    results : dict
        Output from detect_multiple()
    """
    print("Detection Summary:")
    print("-" * 60)
    for (method, q), cps in results.items():
        print(f"Method: {method:20s} | q: {q:4.1f} | Change points: {len(cps):3d}")
    print("-" * 60)
