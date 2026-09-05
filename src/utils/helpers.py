"""
Utility functions for GeoJSON export, formatting, and metrics.
"""
from typing import Any, Dict, List, Optional
import json
from shapely.geometry import mapping


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


def encroachments_to_geojson_dict(
    change_records: List[Dict[str, Any]],
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Serialize flagged potential encroachment protrusion polygons to GeoJSON FeatureCollection.
    
    Args:
        change_records: List of change detection analysis records.
        metadata: Optional metadata dictionary.
        
    Returns:
        GeoJSON FeatureCollection dict.
    """
    features = []
    for r in change_records:
        ext_geom = r.get("potential_extension_geom")
        if r.get("potential_encroachment") and ext_geom is not None and not ext_geom.is_empty:
            feat = {
                "type": "Feature",
                "id": f"ENC-{r['parcel_id']}",
                "geometry": mapping(ext_geom),
                "properties": {
                    "cadastral_id": r["parcel_id"],
                    "candidate_id": r["candidate_id"],
                    "potential_extension_area_px": r["potential_extension_area_px"],
                    "extension_ratio": r["extension_ratio"],
                    "iou": r["iou"],
                    "risk_level": r["risk_level"],
                    "discrepancy_type": r["discrepancy_type"],
                    "verification_status": r["verification_status"],
                    "notice": "Flagged potential boundary extension. Requires certified surveyor verification."
                }
            }
            features.append(feat)

    return {
        "type": "FeatureCollection",
        "name": "Potential_Cadastral_Encroachments",
        "crs": {
            "type": "name",
            "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84 (Pixel Coordinate Space)"}
        },
        "metadata": metadata or {},
        "features": features
    }

