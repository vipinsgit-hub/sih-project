"""
Cadastral Change Detection & Potential Encroachment Analysis Engine.
Analyzes geometric discrepancies between reference cadastral polygons and AI-derived candidate parcels.
Computes spatial metrics, area deltas, potential extension geometries, and assigns prototype risk tiers.
"""
from typing import Any, Dict, List, Optional, Tuple
from shapely.geometry import Polygon
from src.vectorization.polygons import validate_and_repair_geometry
from src.geospatial.comparison import calculate_spatial_overlap


def calculate_change_metrics(
    cadastral_geom: Optional[Polygon],
    candidate_geom: Optional[Polygon]
) -> Dict[str, Any]:
    """
    Calculate comprehensive geometric change metrics between cadastral reference and candidate parcel.
    
    Args:
        cadastral_geom: Shapely Polygon of reference cadastral parcel.
        candidate_geom: Shapely Polygon of AI candidate parcel.
        
    Returns:
        Dictionary of computed change metrics and difference geometries.
    """
    if cadastral_geom is None or cadastral_geom.is_empty:
        cand_area = float(candidate_geom.area) if candidate_geom and not candidate_geom.is_empty else 0.0
        return {
            "iou": 0.0,
            "overlap_percentage": 0.0,
            "intersection_area_px": 0.0,
            "union_area_px": cand_area,
            "cadastral_area_px": 0.0,
            "candidate_area_px": round(cand_area, 2),
            "area_difference_px": round(cand_area, 2),
            "area_difference_pct": 100.0 if cand_area > 0 else 0.0,
            "potential_extension_geom": candidate_geom,
            "potential_extension_area_px": round(cand_area, 2),
            "extension_ratio": 1.0 if cand_area > 0 else 0.0,
            "missing_cadastral_geom": None,
            "missing_cadastral_area_px": 0.0,
        }

    cad_area = float(cadastral_geom.area)

    if candidate_geom is None or candidate_geom.is_empty:
        return {
            "iou": 0.0,
            "overlap_percentage": 0.0,
            "intersection_area_px": 0.0,
            "union_area_px": round(cad_area, 2),
            "cadastral_area_px": round(cad_area, 2),
            "candidate_area_px": 0.0,
            "area_difference_px": round(-cad_area, 2),
            "area_difference_pct": 100.0,
            "potential_extension_geom": None,
            "potential_extension_area_px": 0.0,
            "extension_ratio": 0.0,
            "missing_cadastral_geom": cadastral_geom,
            "missing_cadastral_area_px": round(cad_area, 2),
        }

    cand_area = float(candidate_geom.area)
    iou, inter_area, union_area = calculate_spatial_overlap(cadastral_geom, candidate_geom)

    # Calculate Candidate Extension outside Cadastral boundary (Candidate - Cadastral)
    try:
        ext_geom = candidate_geom.difference(cadastral_geom)
        ext_geom = validate_and_repair_geometry(ext_geom)
        ext_area = float(ext_geom.area) if ext_geom and not ext_geom.is_empty else 0.0
    except Exception:
        ext_geom = None
        ext_area = 0.0

    # Calculate Missing Cadastral area not covered by Candidate (Cadastral - Candidate)
    try:
        missing_geom = cadastral_geom.difference(candidate_geom)
        missing_geom = validate_and_repair_geometry(missing_geom)
        missing_area = float(missing_geom.area) if missing_geom and not missing_geom.is_empty else 0.0
    except Exception:
        missing_geom = None
        missing_area = 0.0

    area_diff = cand_area - cad_area
    area_diff_pct = (abs(area_diff) / cad_area * 100.0) if cad_area > 0 else 0.0
    ext_ratio = (ext_area / cad_area) if cad_area > 0 else 0.0

    return {
        "iou": iou,
        "overlap_percentage": round(iou * 100.0, 1),
        "intersection_area_px": inter_area,
        "union_area_px": union_area,
        "cadastral_area_px": round(cad_area, 2),
        "candidate_area_px": round(cand_area, 2),
        "area_difference_px": round(area_diff, 2),
        "area_difference_pct": round(area_diff_pct, 1),
        "potential_extension_geom": ext_geom,
        "potential_extension_area_px": round(ext_area, 2),
        "extension_ratio": round(ext_ratio, 3),
        "missing_cadastral_geom": missing_geom,
        "missing_cadastral_area_px": round(missing_area, 2),
    }


def classify_discrepancy_and_risk(
    iou: float,
    extension_ratio: float,
    area_diff_pct: float,
    match_iou_threshold: float = 0.85,
    minor_iou_threshold: float = 0.60,
    encroachment_threshold: float = 0.10
) -> Tuple[str, str, bool]:
    """
    Classify spatial discrepancy type and prototype risk level using explicit geometric thresholds.
    
    Args:
        iou: Spatial overlap score (0.0 to 1.0).
        extension_ratio: Ratio of candidate protrusion area to cadastral area.
        area_diff_pct: Percentage area difference relative to cadastral area.
        match_iou_threshold: Minimum IoU for MATCH.
        minor_iou_threshold: Minimum IoU for MINOR BOUNDARY MISMATCH.
        encroachment_threshold: Minimum protrusion ratio for POTENTIAL ENCROACHMENT.
        
    Returns:
        Tuple of (discrepancy_type, risk_level, is_potential_encroachment).
    """
    # 1. Potential Encroachment Check (Lateral protrusion beyond legal parcel bounds)
    if extension_ratio >= encroachment_threshold and iou > 0.15:
        return "POTENTIAL ENCROACHMENT", "HIGH", True

    # 2. Match Check
    if iou >= match_iou_threshold:
        return "MATCH", "LOW", False

    # 3. Minor Boundary Mismatch
    if iou >= minor_iou_threshold:
        risk = "MEDIUM" if area_diff_pct > 15.0 or extension_ratio > 0.05 else "LOW"
        return "MINOR BOUNDARY MISMATCH", risk, False

    # 4. Significant Boundary Mismatch
    if iou > 0.0:
        return "SIGNIFICANT BOUNDARY MISMATCH", "HIGH", False

    # 5. Unmatched
    return "UNMATCHED", "MEDIUM", False


def detect_cadastral_changes(
    cadastral_parcels: List[Dict[str, Any]],
    candidate_parcels: List[Dict[str, Any]],
    match_iou_threshold: float = 0.85,
    minor_iou_threshold: float = 0.60,
    encroachment_threshold: float = 0.10
) -> Dict[str, Any]:
    """
    Execute end-to-end cadastral change and potential encroachment detection pipeline.
    
    Args:
        cadastral_parcels: List of reference cadastral parcels.
        candidate_parcels: List of AI-extracted candidate parcels.
        match_iou_threshold: Threshold for MATCH status.
        minor_iou_threshold: Threshold for MINOR BOUNDARY MISMATCH status.
        encroachment_threshold: Threshold for POTENTIAL ENCROACHMENT status.
        
    Returns:
        Dictionary containing detailed change records, surveyor verification queue, and summary statistics.
    """
    change_records: List[Dict[str, Any]] = []
    matched_candidate_ids = set()

    for cad in cadastral_parcels:
        cad_id = cad.get("cadastral_id", "UNKNOWN")
        cad_geom: Polygon = cad.get("geometry")

        # Find best matching candidate parcel
        best_candidate = None
        best_inter_area = 0.0

        for cand in candidate_parcels:
            cand_geom: Polygon = cand.get("geometry")
            if cad_geom is not None and cand_geom is not None and not cad_geom.is_empty and not cand_geom.is_empty:
                inter = cad_geom.intersection(cand_geom)
                inter_area = float(inter.area) if not inter.is_empty else 0.0
                if inter_area > best_inter_area:
                    best_inter_area = inter_area
                    best_candidate = cand

        if best_candidate is not None and best_inter_area > 0:
            cand_id = best_candidate.get("parcel_id", "UNKNOWN")
            matched_candidate_ids.add(cand_id)
            cand_geom = best_candidate.get("geometry")

            metrics = calculate_change_metrics(cad_geom, cand_geom)
            discrepancy_type, risk_level, is_encroachment = classify_discrepancy_and_risk(
                iou=metrics["iou"],
                extension_ratio=metrics["extension_ratio"],
                area_diff_pct=metrics["area_difference_pct"],
                match_iou_threshold=match_iou_threshold,
                minor_iou_threshold=minor_iou_threshold,
                encroachment_threshold=encroachment_threshold,
            )

            record = {
                "parcel_id": cad_id,
                "candidate_id": cand_id,
                "cadastral_geometry": cad_geom,
                "candidate_geometry": cand_geom,
                "potential_extension_geom": metrics["potential_extension_geom"],
                "missing_cadastral_geom": metrics["missing_cadastral_geom"],
                "cadastral_area_px": metrics["cadastral_area_px"],
                "candidate_area_px": metrics["candidate_area_px"],
                "intersection_area_px": metrics["intersection_area_px"],
                "union_area_px": metrics["union_area_px"],
                "area_difference_px": metrics["area_difference_px"],
                "area_difference_pct": metrics["area_difference_pct"],
                "potential_extension_area_px": metrics["potential_extension_area_px"],
                "extension_ratio": metrics["extension_ratio"],
                "iou": metrics["iou"],
                "overlap_percentage": metrics["overlap_percentage"],
                "discrepancy_type": discrepancy_type,
                "risk_level": risk_level,
                "potential_encroachment": is_encroachment,
                "verification_status": "PENDING SURVEYOR REVIEW",
                "land_use": cad.get("land_use", "Residential / Mixed"),
                "owner_reference": cad.get("owner_reference", "Municipal Registry"),
            }
            change_records.append(record)
        else:
            # Cadastral record with no detected candidate structure
            metrics = calculate_change_metrics(cad_geom, None)
            discrepancy_type, risk_level, is_encroachment = classify_discrepancy_and_risk(
                iou=0.0,
                extension_ratio=0.0,
                area_diff_pct=100.0,
                match_iou_threshold=match_iou_threshold,
                minor_iou_threshold=minor_iou_threshold,
                encroachment_threshold=encroachment_threshold,
            )

            record = {
                "parcel_id": cad_id,
                "candidate_id": "NONE (Unmatched)",
                "cadastral_geometry": cad_geom,
                "candidate_geometry": None,
                "potential_extension_geom": None,
                "missing_cadastral_geom": cad_geom,
                "cadastral_area_px": metrics["cadastral_area_px"],
                "candidate_area_px": 0.0,
                "intersection_area_px": 0.0,
                "union_area_px": metrics["union_area_px"],
                "area_difference_px": metrics["area_difference_px"],
                "area_difference_pct": 100.0,
                "potential_extension_area_px": 0.0,
                "extension_ratio": 0.0,
                "iou": 0.0,
                "overlap_percentage": 0.0,
                "discrepancy_type": "UNMATCHED (No Candidate Detected)",
                "risk_level": "MEDIUM",
                "potential_encroachment": False,
                "verification_status": "PENDING SURVEYOR REVIEW",
                "land_use": cad.get("land_use", "Residential / Mixed"),
                "owner_reference": cad.get("owner_reference", "Municipal Registry"),
            }
            change_records.append(record)

    # Sort surveyor verification queue by risk priority (HIGH -> MEDIUM -> LOW)
    risk_priority = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    sorted_queue = sorted(change_records, key=lambda x: (risk_priority.get(x["risk_level"], 3), -x["potential_extension_area_px"]))

    # Summary Statistics
    matches_count = sum(1 for r in change_records if r["discrepancy_type"] == "MATCH")
    minor_count = sum(1 for r in change_records if "MINOR" in r["discrepancy_type"])
    significant_count = sum(1 for r in change_records if "SIGNIFICANT" in r["discrepancy_type"])
    encroachments_count = sum(1 for r in change_records if r["potential_encroachment"])
    unmatched_count = sum(1 for r in change_records if "UNMATCHED" in r["discrepancy_type"])
    high_risk_count = sum(1 for r in change_records if r["risk_level"] == "HIGH")
    medium_risk_count = sum(1 for r in change_records if r["risk_level"] == "MEDIUM")
    low_risk_count = sum(1 for r in change_records if r["risk_level"] == "LOW")

    return {
        "change_records": change_records,
        "surveyor_queue": sorted_queue,
        "total_cadastral": len(cadastral_parcels),
        "total_candidates": len(candidate_parcels),
        "matches_count": matches_count,
        "minor_discrepancies_count": minor_count,
        "significant_discrepancies_count": significant_count,
        "potential_encroachments_count": encroachments_count,
        "unmatched_count": unmatched_count,
        "high_risk_count": high_risk_count,
        "medium_risk_count": medium_risk_count,
        "low_risk_count": low_risk_count,
        "total_flagged_for_review": len(change_records),
    }

