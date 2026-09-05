"""Geospatial and cadastral comparison submodule."""
from .cadastral import (
    load_cadastral_geojson,
    get_cadastral_statistics,
    CadastralDataError,
)
from .comparison import (
    calculate_spatial_overlap,
    classify_discrepancy,
    compare_cadastral_vs_candidate_parcels,
)

__all__ = [
    "load_cadastral_geojson",
    "get_cadastral_statistics",
    "CadastralDataError",
    "calculate_spatial_overlap",
    "classify_discrepancy",
    "compare_cadastral_vs_candidate_parcels",
]
