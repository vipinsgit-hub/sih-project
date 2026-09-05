"""Geometry and boundary processing submodule."""
from .boundary import (
    extract_boundaries,
    extract_feature_binary_mask,
    clean_binary_mask,
    render_boundary_overlay,
    DEFAULT_PARCEL_FEATURE_CLASSES,
)

__all__ = [
    "extract_boundaries",
    "extract_feature_binary_mask",
    "clean_binary_mask",
    "render_boundary_overlay",
    "DEFAULT_PARCEL_FEATURE_CLASSES",
]
