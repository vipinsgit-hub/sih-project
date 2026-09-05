"""
Polygon vectorization, validation, simplification, and candidate parcel feature generation.
"""
import json
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import MultiPolygon, Polygon, mapping
from shapely.validation import make_valid


def contour_to_shapely_polygon(contour: np.ndarray) -> Optional[Polygon]:
    """
    Convert an OpenCV contour (N, 1, 2) or (N, 2) into a Shapely Polygon.
    
    Args:
        contour: Numpy array of 2D vertex coordinates.
        
    Returns:
        Shapely Polygon or None if contour has fewer than 3 unique points.
    """
    pts = contour.squeeze()
    if pts.ndim != 2 or len(pts) < 3:
        return None

    # Ensure closed ring
    coords = pts.tolist()
    if coords[0] != coords[-1]:
        coords.append(coords[0])

    try:
        poly = Polygon(coords)
        return poly
    except Exception:
        return None


def validate_and_repair_geometry(geom: Any) -> Optional[Polygon]:
    """
    Check geometry validity and perform safe repair (using make_valid / buffer(0)).
    Extracts the largest valid Polygon if a MultiPolygon is returned.
    
    Args:
        geom: Shapely geometry object.
        
    Returns:
        Repaired Shapely Polygon, or None if invalid/empty.
    """
    if geom is None or geom.is_empty:
        return None

    if not geom.is_valid:
        try:
            geom = make_valid(geom)
        except Exception:
            try:
                geom = geom.buffer(0)
            except Exception:
                return None

    if geom.is_empty:
        return None

    # If repair resulted in MultiPolygon, return the constituent polygon with the largest area
    if isinstance(geom, MultiPolygon):
        valid_polys = [p for p in geom.geoms if p.is_valid and not p.is_empty]
        if not valid_polys:
            return None
        geom = max(valid_polys, key=lambda p: p.area)

    if isinstance(geom, Polygon) and geom.is_valid and geom.area > 0:
        return geom

    return None


def simplify_and_regularize_polygon(
    poly: Polygon,
    tolerance: float = 1.5,
    min_area: float = 100.0
) -> Optional[Polygon]:
    """
    Apply Douglas-Peucker simplification while preserving topology.
    
    Args:
        poly: Input Shapely Polygon.
        tolerance: Maximum distance between original curve and simplified polygon.
        min_area: Minimum acceptable area in pixels.
        
    Returns:
        Simplified and validated Polygon, or None if filtered out.
    """
    if poly is None or poly.is_empty:
        return None

    try:
        simplified = poly.simplify(tolerance=tolerance, preserve_topology=True)
        repaired = validate_and_repair_geometry(simplified)

        if repaired is not None and repaired.area >= min_area:
            return repaired
        return None
    except Exception:
        return validate_and_repair_geometry(poly)


def generate_candidate_parcels(
    contours: List[np.ndarray],
    tolerance: float = 1.5,
    min_area: float = 100.0,
    id_prefix: str = "P-"
) -> Dict[str, Any]:
    """
    Convert raw boundary contours into structured, regularized candidate parcel records.
    
    Args:
        contours: List of OpenCV boundary contours.
        tolerance: Polygon simplification tolerance (Douglas-Peucker).
        min_area: Minimum acceptable parcel area in pixels.
        id_prefix: Prefix for sequential parcel IDs (e.g. "P-").
        
    Returns:
        Dictionary containing:
            - parcels: List of parcel dictionaries.
            - total_input_contours: Count of input contours.
            - valid_parcels_count: Count of successfully generated parcels.
            - rejected_count: Count of discarded degenerate/small polygons.
    """
    parcels: List[Dict[str, Any]] = []
    rejected_count = 0
    parcel_idx = 1

    for cnt in contours:
        raw_poly = contour_to_shapely_polygon(cnt)
        if raw_poly is None:
            rejected_count += 1
            continue

        regularized_poly = simplify_and_regularize_polygon(
            raw_poly,
            tolerance=tolerance,
            min_area=min_area
        )

        if regularized_poly is None:
            rejected_count += 1
            continue

        # Compute geometric properties (strictly in pixel units for uncalibrated imagery)
        pixel_area = round(float(regularized_poly.area), 2)
        pixel_perimeter = round(float(regularized_poly.length), 2)
        centroid = regularized_poly.centroid
        minx, miny, maxx, maxy = regularized_poly.bounds

        # Convexity / Solidity score (ratio of polygon area to convex hull area)
        convex_hull = regularized_poly.convex_hull
        solidity = round(float(regularized_poly.area / convex_hull.area), 3) if convex_hull.area > 0 else 1.0

        parcel_id = f"{id_prefix}{parcel_idx:03d}"
        parcel_idx += 1

        parcels.append({
            "parcel_id": parcel_id,
            "geometry": regularized_poly,
            "pixel_area": pixel_area,
            "pixel_perimeter": pixel_perimeter,
            "centroid_pixel": (round(float(centroid.x), 1), round(float(centroid.y), 1)),
            "bounding_box": (round(minx, 1), round(miny, 1), round(maxx, 1), round(maxy, 1)),
            "vertex_count": len(regularized_poly.exterior.coords),
            "solidity": solidity,
            "source": "AI-Assisted Boundary Extraction",
            "status": "Candidate",
        })

    return {
        "parcels": parcels,
        "total_input_contours": len(contours),
        "valid_parcels_count": len(parcels),
        "rejected_count": rejected_count,
    }


# Alias for backward compatibility
generate_parcel_polygons = generate_candidate_parcels


def parcels_to_geojson_dict(
    parcels: List[Dict[str, Any]],
    image_metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Serialize candidate parcels to standard GeoJSON FeatureCollection (in pixel coordinate space).
    
    Args:
        parcels: List of candidate parcel dictionaries.
        image_metadata: Optional image metadata to include in GeoJSON collection properties.
        
    Returns:
        GeoJSON FeatureCollection dictionary.
    """
    features = []

    for p in parcels:
        geom = p["geometry"]
        feat = {
            "type": "Feature",
            "id": p["parcel_id"],
            "geometry": mapping(geom),
            "properties": {
                "parcel_id": p["parcel_id"],
                "pixel_area": p["pixel_area"],
                "pixel_perimeter": p["pixel_perimeter"],
                "vertex_count": p["vertex_count"],
                "solidity": p["solidity"],
                "source": p["source"],
                "status": p["status"],
                "unit": "pixels",
                "notice": "Candidate parcel geometry extracted by AI prototype. Pending certified surveyor verification."
            }
        }
        features.append(feat)

    collection = {
        "type": "FeatureCollection",
        "name": "Candidate_Urban_Parcels",
        "crs": {
            "type": "name",
            "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84 (Pixel Coordinate Space)"}
        },
        "metadata": image_metadata or {},
        "features": features
    }
    return collection


def export_parcels_geojson(parcels: List[Dict[str, Any]], output_path: str) -> bool:
    """Save parcels FeatureCollection directly to a .geojson file."""
    fc = parcels_to_geojson_dict(parcels)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(fc, f, indent=2)
    return True


def render_parcel_overlay(
    image: Union[Image.Image, np.ndarray],
    parcels: List[Dict[str, Any]],
    boundary_color: Tuple[int, int, int] = (0, 220, 255),
    fill_color: Tuple[int, int, int] = (0, 180, 255),
    fill_alpha: float = 0.35,
    show_labels: bool = True
) -> Image.Image:
    """
    Render parcel polygons with translucent fills, crisp boundary borders, and centered ID labels.
    
    Args:
        image: Original RGB PIL Image or numpy array.
        parcels: List of parcel dictionaries.
        boundary_color: RGB tuple for perimeter strokes.
        fill_color: RGB tuple for polygon interior fill.
        fill_alpha: Transparency of interior fill (0.0 to 1.0).
        show_labels: Whether to render parcel IDs (e.g. 'P-001') at centroids.
        
    Returns:
        Rendered PIL.Image.Image.
    """
    if isinstance(image, np.ndarray):
        base_img = Image.fromarray(image).convert("RGBA")
    else:
        base_img = image.convert("RGBA")

    w, h = base_img.size

    # Layer for translucent polygon fills
    overlay_fill = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw_fill = ImageDraw.Draw(overlay_fill)

    # Layer for solid boundaries & text
    overlay_lines = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw_lines = ImageDraw.Draw(overlay_lines)

    fill_rgba = (fill_color[0], fill_color[1], fill_color[2], int(255 * fill_alpha))
    border_rgba = (boundary_color[0], boundary_color[1], boundary_color[2], 255)

    for p in parcels:
        poly: Polygon = p["geometry"]
        coords = [(int(x), int(y)) for x, y in poly.exterior.coords]

        if len(coords) >= 3:
            # 1. Fill polygon
            draw_fill.polygon(coords, fill=fill_rgba)
            # 2. Draw outline
            draw_lines.line(coords, fill=border_rgba, width=2)

            # 3. Label at centroid
            if show_labels:
                cx, cy = int(poly.centroid.x), int(poly.centroid.y)
                label_text = p["parcel_id"]

                # Text bounding badge
                box_pad = 3
                text_w = len(label_text) * 7
                text_h = 12
                draw_lines.rectangle(
                    [cx - text_w // 2 - box_pad, cy - text_h // 2 - box_pad,
                     cx + text_w // 2 + box_pad, cy + text_h // 2 + box_pad],
                    fill=(15, 23, 42, 220),
                    outline=(255, 255, 255, 200),
                    width=1
                )
                draw_lines.text(
                    (cx - text_w // 2, cy - text_h // 2 - 1),
                    label_text,
                    fill=(255, 255, 255, 255)
                )

    # Composite layers
    combined = Image.alpha_composite(base_img, overlay_fill)
    final_img = Image.alpha_composite(combined, overlay_lines)
    return final_img.convert("RGB")
