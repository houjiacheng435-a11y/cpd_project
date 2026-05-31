"""Example: Multiple detection methods with Jaccard similarity heatmap."""

import pandas as pd
from pathlib import Path

from cpd import detect_multiple, print_detection_summary, plot_change_points, plot_jaccard_heatmap


def main() -> None:
    project_root = Path(__file__).resolve().parent
    data_path = project_root / 'data' / 'TEST.parquet(1).gzip'
    
    # Load data
    df = pd.read_parquet(data_path)
    series = df['p']
    
    # Define multiple detection configurations
    # Each tuple is (method, q_parameter)
    configs = [
        ('cusum', 0.5),
        ('cusum', 1.0),
        ('cusum', 1.5),
        ('glr', 1.0),
        ('adaptive_cusum', 1.0),
    ]
    
    # Run detections
    results = detect_multiple(series, configs)
    
    # Print summary
    print_detection_summary(results)
    
    # Plot Jaccard similarity heatmap
    print("\nGenerating Jaccard similarity heatmap...")
    plot_jaccard_heatmap(results)
    
    # Optionally plot one method result
    print("\nPlotting CUSUM (q=1.0) result...")
    change_points_cusum = results[('cusum', 1.0)]
    plot_change_points(series, change_points_cusum, method='cusum', q=1.0)


if __name__ == '__main__':
    main()
