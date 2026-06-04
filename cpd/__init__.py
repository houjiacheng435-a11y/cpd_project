"""CPD change point detection package."""

from .brk_detector import Brk, generate_breaks
from .market_state_vector import (
    MarketStateVector,
    build_market_state_vector,
    evaluate_state_vector,
    get_core_state_vector,
    load_data,
    plot_price_state_background,
)
from .multi_detect import detect_multiple, print_detection_summary
from .visualization import plot_change_points, plot_jaccard_heatmap

__all__ = [
    "Brk",
    "generate_breaks",
    "detect_multiple",
    "print_detection_summary",
    "plot_change_points",
    "plot_jaccard_heatmap",
    "MarketStateVector",
    "load_data",
    "build_market_state_vector",
    "get_core_state_vector",
    "evaluate_state_vector",
    "plot_price_state_background",
]
