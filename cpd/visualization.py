"""Visualization utilities for change point detection results."""

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


def plot_change_points(series: pd.Series, change_points: list, method: str, q: float = 1.0) -> None:
    """
    Plot time series with detected change points.
    
    Parameters
    ----------
    series : pd.Series
        Time series data with DatetimeIndex
    change_points : list
        List of detected change points (timestamps)
    method : str
        Name of the detection method
    q : float
        Sensitivity parameter used
    """
    plt.figure(figsize=(14, 6))
    plt.plot(series.index, series.values, color='blue', linewidth=0.8, label='Data')
    
    for i, cp in enumerate(change_points):
        label = 'Change Points' if i == 0 else ''
        plt.axvline(x=cp, color='gold', linestyle='-', linewidth=0.8, alpha=0.9, label=label)
    
    plt.title(f'Change Point Detection (Method: {method}, q={q:.2f}, Count: {len(change_points)})', fontsize=12)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_jaccard_heatmap(results_dict: dict) -> None:
    """
    Plot Jaccard similarity matrix as a heatmap.
    
    Parameters
    ----------
    results_dict : dict
        Dictionary mapping (method, q) tuples to lists of change points
        Example: {('cusum', 1.0): [ts1, ts2, ...], ('glr', 0.5): [...]}
    """
    labels = [f"{method}\nq={q}" for method, q in results_dict.keys()]
    change_points_list = list(results_dict.values())
    
    n = len(results_dict)
    jaccard_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            jaccard_matrix[i, j] = _jaccard_similarity(change_points_list[i], change_points_list[j])
    
    plt.figure(figsize=(8, 7))
    sns.heatmap(jaccard_matrix, annot=True, fmt='.3f', cmap='YlOrRd', 
                xticklabels=labels, yticklabels=labels, cbar_kws={'label': 'Jaccard Similarity'})
    plt.title('Jaccard Similarity Matrix of Change Point Detections')
    plt.tight_layout()
    plt.show()


def _jaccard_similarity(points1: list, points2: list) -> float:
    """
    Compute Jaccard similarity between two lists of timestamps.
    
    Jaccard = |intersection| / |union|
    """
    if len(points1) == 0 and len(points2) == 0:
        return 1.0
    if len(points1) == 0 or len(points2) == 0:
        return 0.0
    
    set1 = set(points1)
    set2 = set(points2)
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    
    return intersection / union if union > 0 else 0.0
