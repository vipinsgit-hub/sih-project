"""
GIS Layered Visualization Module: Cadastral Reference vs. AI Candidate Parcels vs. Discrepancy Zones.
"""
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import Polygon, MultiPolygon


def render_gis_comparison_map(
    image: Union[Image.Image, np.ndarray],
    cadastral_parcels: List[Dict[str, Any]],
    candidate_parcels: List[Dict[str, Any]],
    comparison_results: Optional[List[Dict[str, Any]]] = None,
    show_cadastral: bool = True,
    show_candidates: bool = True,
    show_discrepancies: bool = True,
    selected_cadastral_id: Optional[str] = None
) -> Image.Image:
    """
    Render a multi-layer GIS visualization comparing cadastral records against AI candidate boundaries.
    
    Args:
        image: Base aerial survey photo (RGB PIL Image or numpy array).
        cadastral_parcels: List of reference cadastral parcel dictionaries.
        candidate_parcels: List of AI-derived candidate parcel dictionaries.
        comparison_results: List of spatial comparison results with discrepancy geometries.
        show_cadastral: Toggle Layer 1 (Cadastral boundaries).
        show_candidates: Toggle Layer 2 (AI candidate parcels).
        show_discrepancies: Toggle Layer 3 (Potential encroachment / discrepancy zones).
        selected_cadastral_id: Optional specific cadastral ID to highlight with high-visibility accent.
        
    Returns:
        Composited PIL.Image.Image.
    """
    if isinstance(image, np.ndarray):
        base_img = Image.fromarray(image).convert("RGBA")
    else:
        base_img = image.convert("RGBA")

    w, h = base_img.size

    # Layer for translucent polygon fills
    overlay_fill = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw_fill = ImageDraw.Draw(overlay_fill)

    # Layer for solid vector outlines & text labels
    overlay_lines = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw_lines = ImageDraw.Draw(overlay_lines)

    # Color Palette Definitions (RGBA)
    CAD_FILL = (30, 64, 175, 45)         # Navy/Blue subtle fill
    CAD_STROKE = (37, 99, 235, 255)       # Strong Royal Blue outline
    
    CAND_FILL = (6, 182, 212, 70)        # Bright Cyan fill
    CAND_STROKE = (8, 145, 178, 255)      # Cyan outline

    ENCROACH_FILL = (239, 68, 68, 160)    # Crimson Red highlight fill
    ENCROACH_STROKE = (220, 38, 38, 255)  # Crimson Red outline

    SELECTED_STROKE = (250, 204, 21, 255) # Bright Yellow highlight border

    # 1. Render Cadastral Reference Parcels (Layer 1)
    if show_cadastral:
        for cad in cadastral_parcels:
            poly: Polygon = cad["geometry"]
            is_selected = (selected_cadastral_id == cad["cadastral_id"])
            coords = [(int(x), int(y)) for x, y in poly.exterior.coords]

            if len(coords) >= 3:
                draw_fill.polygon(coords, fill=(30, 64, 175, 80) if is_selected else CAD_FILL)
                stroke_color = SELECTED_STROKE if is_selected else CAD_STROKE
                stroke_width = 3 if is_selected else 2
                draw_lines.line(coords, fill=stroke_color, width=stroke_width)

                # Cadastral Label (Top-Left corner of bounding box)
                minx, miny, _, _ = poly.bounds
                lbl = cad["cadastral_id"]
                draw_lines.rectangle(
                    [int(minx) + 2, int(miny) + 2, int(minx) + len(lbl) * 7 + 8, int(miny) + 16],
                    fill=(30, 58, 138, 230),
                    outline=(255, 255, 255, 220),
                    width=1
                )
                draw_lines.text((int(minx) + 5, int(miny) + 3), lbl, fill=(255, 255, 255, 255))

    # 2. Render AI Candidate Parcels (Layer 2)
    if show_candidates:
        for cand in candidate_parcels:
            poly: Polygon = cand["geometry"]
            coords = [(int(x), int(y)) for x, y in poly.exterior.coords]

            if len(coords) >= 3:
                draw_fill.polygon(coords, fill=CAND_FILL)
                draw_lines.line(coords, fill=CAND_STROKE, width=2)

                # Candidate Label at centroid
                cx, cy = int(poly.centroid.x), int(poly.centroid.y)
                lbl = cand["parcel_id"]
                draw_lines.rectangle(
                    [cx - len(lbl) * 4 - 4, cy - 7, cx + len(lbl) * 4 + 4, cy + 7],
                    fill=(15, 118, 110, 220),
                    outline=(255, 255, 255, 200),
                    width=1
                )
                draw_lines.text((cx - len(lbl) * 4, cy - 6), lbl, fill=(255, 255, 255, 255))

    # 3. Render Potential Encroachment / Discrepancy Zones (Layer 3)
    if show_discrepancies and comparison_results:
        for comp in comparison_results:
            enc_geom = comp.get("encroachment_geometry")
            if enc_geom is not None and not enc_geom.is_empty:
                geoms_to_draw = enc_geom.geoms if isinstance(enc_geom, MultiPolygon) else [enc_geom]

                for g in geoms_to_draw:
                    if isinstance(g, Polygon) and len(g.exterior.coords) >= 3:
                        coords = [(int(x), int(y)) for x, y in g.exterior.coords]
                        draw_fill.polygon(coords, fill=ENCROACH_FILL)
                        draw_lines.line(coords, fill=ENCROACH_STROKE, width=2)

                        # Encroachment alert badge at discrepancy center
                        if g.area > 300:
                            gx, gy = int(g.centroid.x), int(g.centroid.y)
                            alert_lbl = "⚠️ ENCROACH"
                            draw_lines.rectangle(
                                [gx - 32, gy - 8, gx + 32, gy + 8],
                                fill=(185, 28, 28, 240),
                                outline=(254, 202, 202, 255),
                                width=1
                            )
                            draw_lines.text((gx - 30, gy - 6), alert_lbl, fill=(255, 255, 255, 255))

    # Composite layers
    combined = Image.alpha_composite(base_img, overlay_fill)
    final_img = Image.alpha_composite(combined, overlay_lines)
    return final_img.convert("RGB")


def render_parcel_detail_comparison(
    image: Union[Image.Image, np.ndarray],
    cadastral_geom: Optional[Polygon],
    candidate_geom: Optional[Polygon],
    extension_geom: Optional[Polygon] = None,
    missing_geom: Optional[Polygon] = None,
    crop_to_parcel: bool = True,
    padding: int = 40
) -> Image.Image:
    """
    Render a focused deep-dive discrepancy comparison map for a single selected parcel.
    
    Args:
        image: Base aerial image.
        cadastral_geom: Shapely Polygon of reference cadastral parcel.
        candidate_geom: Shapely Polygon of AI candidate parcel.
        extension_geom: Shapely Polygon representing candidate area outside cadastral parcel.
        missing_geom: Shapely Polygon representing cadastral area not covered by candidate.
        crop_to_parcel: If True, crops the view closely around the parcel bounding box with padding.
        padding: Pixel padding around the bounding box if cropped.
        
    Returns:
        Rendered PIL.Image.Image.
    """
    if isinstance(image, np.ndarray):
        base_img = Image.fromarray(image).convert("RGBA")
    else:
        base_img = image.convert("RGBA")

    w, h = base_img.size

    overlay_fill = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw_fill = ImageDraw.Draw(overlay_fill)

    overlay_lines = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw_lines = ImageDraw.Draw(overlay_lines)

    # 1. Cadastral Boundary (Blue)
    if cadastral_geom is not None and not cadastral_geom.is_empty:
        coords = [(int(x), int(y)) for x, y in cadastral_geom.exterior.coords]
        draw_fill.polygon(coords, fill=(30, 64, 175, 60))
        draw_lines.line(coords, fill=(37, 99, 235, 255), width=3)

    # 2. Candidate Boundary (Cyan)
    if candidate_geom is not None and not candidate_geom.is_empty:
        coords = [(int(x), int(y)) for x, y in candidate_geom.exterior.coords]
        draw_fill.polygon(coords, fill=(6, 182, 212, 70))
        draw_lines.line(coords, fill=(8, 145, 178, 255), width=3)

    # 3. Missing Cadastral (Orange)
    if missing_geom is not None and not missing_geom.is_empty:
        geoms = missing_geom.geoms if isinstance(missing_geom, MultiPolygon) else [missing_geom]
        for g in geoms:
            if isinstance(g, Polygon) and len(g.exterior.coords) >= 3:
                coords = [(int(x), int(y)) for x, y in g.exterior.coords]
                draw_fill.polygon(coords, fill=(245, 158, 11, 140))
                draw_lines.line(coords, fill=(217, 119, 6, 255), width=2)

    # 4. Potential Extension Area (Crimson Red)
    if extension_geom is not None and not extension_geom.is_empty:
        geoms = extension_geom.geoms if isinstance(extension_geom, MultiPolygon) else [extension_geom]
        for g in geoms:
            if isinstance(g, Polygon) and len(g.exterior.coords) >= 3:
                coords = [(int(x), int(y)) for x, y in g.exterior.coords]
                draw_fill.polygon(coords, fill=(239, 68, 68, 180))
                draw_lines.line(coords, fill=(220, 38, 38, 255), width=3)

    combined = Image.alpha_composite(base_img, overlay_fill)
    final_img = Image.alpha_composite(combined, overlay_lines).convert("RGB")

    if crop_to_parcel:
        # Determine union bounds for cropping
        bounds = None
        for g in [cadastral_geom, candidate_geom, extension_geom]:
            if g is not None and not g.is_empty:
                b = g.bounds
                if bounds is None:
                    bounds = list(b)
                else:
                    bounds[0] = min(bounds[0], b[0])
                    bounds[1] = min(bounds[1], b[1])
                    bounds[2] = max(bounds[2], b[2])
                    bounds[3] = max(bounds[3], b[3])

        if bounds is not None:
            minx = max(0, int(bounds[0]) - padding)
            miny = max(0, int(bounds[1]) - padding)
            maxx = min(w, int(bounds[2]) + padding)
            maxy = min(h, int(bounds[3]) + padding)
            if maxx > minx and maxy > miny:
                return final_img.crop((minx, miny, maxx, maxy))

    return final_img

