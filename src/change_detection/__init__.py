"""Change detection submodule."""
from .compare import (
    calculate_change_metrics,
    classify_discrepancy_and_risk,
    detect_cadastral_changes,
)

__all__ = [
    "calculate_change_metrics",
    "classify_discrepancy_and_risk",
    "detect_cadastral_changes",
]

