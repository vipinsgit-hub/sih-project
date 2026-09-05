"""
Cadastral GeoJSON parser, geometry validator, and dataset statistics module.
"""
import json
import os
from typing import Any, Dict, List, Optional, Union
from shapely.geometry import shape, Polygon
from src.vectorization.polygons import validate_and_repair_geometry


class CadastralDataError(Exception):
    """Exception raised for errors in cadastral data loading or structure."""
    pass


def load_cadastral_geojson(
    file_source: Union[str, bytes, Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Safely load, parse, and validate cadastral parcel geometries from GeoJSON.
    
    Args:
        file_source: Path to .geojson file, raw bytes, or pre-loaded dictionary.
        
    Returns:
        List of validated cadastral parcel dictionaries.
        
    Raises:
        CadastralDataError: If the GeoJSON format is invalid or unreadable.
    """
    try:
        if isinstance(file_source, str):
            if not os.path.exists(file_source):
                raise CadastralDataError(f"Cadastral file not found at path: {file_source}")
            with open(file_source, "r", encoding="utf-8") as f:
                data = json.load(f)
        elif isinstance(file_source, bytes):
            data = json.loads(file_source.decode("utf-8"))
        elif isinstance(file_source, dict):
            data = file_source
        else:
            raise CadastralDataError(f"Unsupported cadastral source type: {type(file_source)}")

        if not isinstance(data, dict) or data.get("type") != "FeatureCollection":
            raise CadastralDataError("Invalid GeoJSON: Root object must be a 'FeatureCollection'.")

        features = data.get("features", [])
        validated_parcels: List[Dict[str, Any]] = []

        for idx, feat in enumerate(features):
            if not isinstance(feat, dict):
                continue

            raw_geom = feat.get("geometry")
            if not raw_geom:
                continue

            try:
                shapely_geom = shape(raw_geom)
                valid_poly = validate_and_repair_geometry(shapely_geom)
            except Exception:
                continue

            if valid_poly is None or valid_poly.is_empty or valid_poly.area <= 0:
                continue

            props = feat.get("properties", {})
            cadastral_id = props.get("cadastral_id") or feat.get("id") or f"CAD-{idx+1:03d}"
            registered_area = props.get("registered_area_px") or round(float(valid_poly.area), 2)

            validated_parcels.append({
                "cadastral_id": str(cadastral_id),
                "geometry": valid_poly,
                "pixel_area": round(float(valid_poly.area), 2),
                "pixel_perimeter": round(float(valid_poly.length), 2),
                "registered_area_px": registered_area,
                "land_use": props.get("land_use", "Unspecified"),
                "owner_reference": props.get("owner_reference", "N/A"),
                "description": props.get("description", ""),
                "properties": props,
                "source": "Existing Cadastral Survey Record",
            })

        return validated_parcels

    except json.JSONDecodeError as e:
        raise CadastralDataError(f"Failed to parse GeoJSON: {str(e)}") from e
    except Exception as e:
        if isinstance(e, CadastralDataError):
            raise
        raise CadastralDataError(f"Error reading cadastral data: {str(e)}") from e


def get_cadastral_statistics(
    cadastral_parcels: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Compute aggregate summary statistics for a collection of cadastral records.
    
    Args:
        cadastral_parcels: List of cadastral parcel dictionaries.
        
    Returns:
        Dictionary of aggregate statistics.
    """
    total_count = len(cadastral_parcels)
    if total_count == 0:
        return {
            "total_parcels": 0,
            "total_area_px": 0.0,
            "avg_area_px": 0.0,
            "land_use_breakdown": {},
        }

    total_area = sum(p["pixel_area"] for p in cadastral_parcels)
    avg_area = total_area / total_count

    land_uses: Dict[str, int] = {}
    for p in cadastral_parcels:
        lu = p.get("land_use", "Unspecified")
        land_uses[lu] = land_uses.get(lu, 0) + 1

    return {
        "total_parcels": total_count,
        "total_area_px": round(total_area, 2),
        "avg_area_px": round(avg_area, 2),
        "land_use_breakdown": land_uses,
    }
