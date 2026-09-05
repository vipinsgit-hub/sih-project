"""
Utility functions for GeoJSON export, formatting, and metrics.
"""
from typing import Any, Dict, List
import json

def format_area_sqm(area: float) -> str:
    """Format area in square meters and hectares."""
    return f"{area:,.2f} sq.m ({area / 10000:,.4f} ha)"

def export_geojson(features: List[Dict[str, Any]], output_path: str) -> bool:
    """Export geo features to GeoJSON file."""
    fc = {
        "type": "FeatureCollection",
        "features": features
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(fc, f, indent=2)
    return True
