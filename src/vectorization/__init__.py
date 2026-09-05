"""Vectorization submodule."""
from .polygons import (
    generate_candidate_parcels,
    generate_parcel_polygons,
    contour_to_shapely_polygon,
    validate_and_repair_geometry,
    simplify_and_regularize_polygon,
    parcels_to_geojson_dict,
    export_parcels_geojson,
    render_parcel_overlay,
)

__all__ = [
    "generate_candidate_parcels",
    "generate_parcel_polygons",
    "contour_to_shapely_polygon",
    "validate_and_repair_geometry",
    "simplify_and_regularize_polygon",
    "parcels_to_geojson_dict",
    "export_parcels_geojson",
    "render_parcel_overlay",
]
