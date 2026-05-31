"""CPD change point detection package."""

from .brk_detector import Brk, generate_breaks
from .multi_detect import detect_multiple, print_detection_summary
from .visualization import plot_change_points, plot_jaccard_heatmap

__all__ = ["Brk", "generate_breaks", "detect_multiple", "print_detection_summary", 
           "plot_change_points", "plot_jaccard_heatmap"]
