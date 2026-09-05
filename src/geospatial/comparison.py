"""
Spatial comparison engine: Cadastral Reference Geometry vs. AI Candidate Parcels.
Computes IoU (Spatial Overlap Score), area differences, discrepancy classification, and potential encroachment alerts.
"""
from typing import Any, Dict, List, Optional, Tuple
from shapely.geometry import Polygon
from src.vectorization.polygons import validate_and_repair_geometry


def calculate_spatial_overlap(
    poly_a: Polygon,
    poly_b: Polygon
) -> Tuple[float, float, float]:
    """
    Calculate Intersection over Union (IoU), intersection area, and union area.
    
    Args:
        poly_a: First Shapely Polygon.
        poly_b: Second Shapely Polygon.
        
    Returns:
        Tuple of (iou, intersection_area, union_area).
    """
    if poly_a is None or poly_b is None or poly_a.is_empty or poly_b.is_empty:
        return 0.0, 0.0, 0.0

    try:
        inter = poly_a.intersection(poly_b)
        inter_area = float(inter.area) if not inter.is_empty else 0.0

        union_geom = poly_a.union(poly_b)
        union_area = float(union_geom.area) if not union_geom.is_empty else 0.0

        if union_area > 0:
            iou = inter_area / union_area
        else:
            iou = 0.0

        return round(iou, 4), round(inter_area, 2), round(union_area, 2)
    except Exception:
        return 0.0, 0.0, 0.0


def classify_discrepancy(
    iou: float,
    excess_ratio: float,
    match_iou_threshold: float = 0.85,
    minor_iou_threshold: float = 0.60,
    encroachment_threshold: float = 0.15
) -> Tuple[str, bool]:
    """
    Classify spatial discrepancy based on explicit geometric rules.
    
    Args:
        iou: Spatial overlap score (0.0 to 1.0).
        excess_ratio: Ratio of candidate polygon extending outside cadastral boundary.
        match_iou_threshold: Minimum IoU for MATCH. Default 0.85.
        minor_iou_threshold: Minimum IoU for MINOR MISMATCH. Default 0.60.
        encroachment_threshold: Minimum excess ratio to trigger POTENTIAL ENCROACHMENT. Default 0.15.
        
    Returns:
        Tuple of (discrepancy_category, is_potential_encroachment).
    """
    # Encroachment rule: Candidate structure extends significantly beyond legal parcel line
    if excess_ratio >= encroachment_threshold and iou > 0.20:
        return "POTENTIAL ENCROACHMENT", True

    if iou >= match_iou_threshold:
        return "MATCH", False
    elif iou >= minor_iou_threshold:
        return "MINOR MISMATCH", False
    else:
        return "BOUNDARY MISMATCH", False


def compare_cadastral_vs_candidate_parcels(
    cadastral_parcels: List[Dict[str, Any]],
    candidate_parcels: List[Dict[str, Any]],
    match_iou_threshold: float = 0.85,
    minor_iou_threshold: float = 0.60,
    encroachment_threshold: float = 0.15
) -> Dict[str, Any]:
    """
    Perform spatial comparison between existing cadastral survey records and AI-extracted candidate parcels.
    
    Args:
        cadastral_parcels: List of reference cadastral parcel records.
        candidate_parcels: List of AI-derived candidate parcel records.
        match_iou_threshold: Threshold for MATCH status.
        minor_iou_threshold: Threshold for MINOR MISMATCH status.
        encroachment_threshold: Minimum external protrusion ratio for POTENTIAL ENCROACHMENT.
        
    Returns:
        Dictionary containing comparison results and summary statistics.
    """
    comparison_results: List[Dict[str, Any]] = []
    matched_candidate_ids = set()

    for cad in cadastral_parcels:
        cad_id = cad["cadastral_id"]
        cad_geom: Polygon = cad["geometry"]
        cad_area = cad["pixel_area"]

        # Find best matching candidate parcel by maximum intersection area
        best_candidate = None
        best_inter_area = 0.0
        best_iou = 0.0
        best_union_area = 0.0

        for cand in candidate_parcels:
            cand_geom: Polygon = cand["geometry"]
            iou, inter_area, union_area = calculate_spatial_overlap(cad_geom, cand_geom)

            if inter_area > best_inter_area:
                best_inter_area = inter_area
                best_iou = iou
                best_union_area = union_area
                best_candidate = cand

        if best_candidate is not None and best_inter_area > 0:
            cand_id = best_candidate["parcel_id"]
            matched_candidate_ids.add(cand_id)
            cand_geom = best_candidate["geometry"]
            cand_area = best_candidate["pixel_area"]

            # Compute excess geometry outside cadastral bounds
            try:
                excess_diff = cand_geom.difference(cad_geom)
                excess_diff = validate_and_repair_geometry(excess_diff)
                excess_area = float(excess_diff.area) if excess_diff and not excess_diff.is_empty else 0.0
            except Exception:
                excess_diff = None
                excess_area = 0.0

            excess_ratio = excess_area / cad_area if cad_area > 0 else 0.0
            area_diff = round(cand_area - cad_area, 2)

            discrepancy_type, is_encroachment = classify_discrepancy(
                iou=best_iou,
                excess_ratio=excess_ratio,
                match_iou_threshold=match_iou_threshold,
                minor_iou_threshold=minor_iou_threshold,
                encroachment_threshold=encroachment_threshold,
            )

            comparison_results.append({
                "cadastral_id": cad_id,
                "candidate_id": cand_id,
                "cadastral_geometry": cad_geom,
                "candidate_geometry": cand_geom,
                "encroachment_geometry": excess_diff if is_encroachment else None,
                "cadastral_area_px": cad_area,
                "candidate_area_px": cand_area,
                "intersection_area_px": best_inter_area,
                "union_area_px": best_union_area,
                "area_difference_px": area_diff,
                "excess_area_px": round(excess_area, 2),
                "excess_ratio": round(excess_ratio, 3),
                "iou": best_iou,
                "overlap_percentage": round(best_iou * 100.0, 1),
                "discrepancy_type": discrepancy_type,
                "potential_encroachment": is_encroachment,
                "land_use": cad.get("land_use", "N/A"),
                "owner_reference": cad.get("owner_reference", "N/A"),
                "status": discrepancy_type,
            })
        else:
            # Cadastral parcel with no corresponding candidate structure detected
            comparison_results.append({
                "cadastral_id": cad_id,
                "candidate_id": "NONE (Unmatched)",
                "cadastral_geometry": cad_geom,
                "candidate_geometry": None,
                "encroachment_geometry": None,
                "cadastral_area_px": cad_area,
                "candidate_area_px": 0.0,
                "intersection_area_px": 0.0,
                "union_area_px": cad_area,
                "area_difference_px": -cad_area,
                "excess_area_px": 0.0,
                "excess_ratio": 0.0,
                "iou": 0.0,
                "overlap_percentage": 0.0,
                "discrepancy_type": "UNMATCHED (No Candidate)",
                "potential_encroachment": False,
                "land_use": cad.get("land_use", "N/A"),
                "owner_reference": cad.get("owner_reference", "N/A"),
                "status": "UNMATCHED",
            })

    # Summary Counts
    matches_count = sum(1 for r in comparison_results if r["discrepancy_type"] == "MATCH")
    minor_mismatches_count = sum(1 for r in comparison_results if r["discrepancy_type"] == "MINOR MISMATCH")
    boundary_mismatches_count = sum(1 for r in comparison_results if r["discrepancy_type"] == "BOUNDARY MISMATCH")
    encroachments_count = sum(1 for r in comparison_results if r["potential_encroachment"])
    unmatched_count = sum(1 for r in comparison_results if "UNMATCHED" in r["discrepancy_type"])

    return {
        "comparison_results": comparison_results,
        "total_cadastral": len(cadastral_parcels),
        "total_candidates": len(candidate_parcels),
        "matches_count": matches_count,
        "minor_mismatches_count": minor_mismatches_count,
        "boundary_mismatches_count": boundary_mismatches_count,
        "potential_encroachments_count": encroachments_count,
        "unmatched_count": unmatched_count,
    }
