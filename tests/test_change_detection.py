"""
Unit tests for Cadastral Change Detection, Potential Encroachment Analysis, and Surveyor Verification Queue.
"""
import unittest
from shapely.geometry import Polygon
from src.change_detection.compare import (
    calculate_change_metrics,
    classify_discrepancy_and_risk,
    detect_cadastral_changes,
)


class TestCadastralChangeDetection(unittest.TestCase):
    """Test suite for Milestone 6 Change Detection & Encroachment Analysis."""

    def setUp(self):
        # Base cadastral parcel 100x100 square (Area = 10,000 px²)
        self.cadastral_poly = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])

    def test_exact_match(self):
        """Test exact identical polygon comparison."""
        cand_poly = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
        metrics = calculate_change_metrics(self.cadastral_poly, cand_poly)
        
        self.assertAlmostEqual(metrics["iou"], 1.0, places=2)
        self.assertAlmostEqual(metrics["area_difference_px"], 0.0, places=1)
        self.assertAlmostEqual(metrics["area_difference_pct"], 0.0, places=1)
        self.assertAlmostEqual(metrics["potential_extension_area_px"], 0.0, places=1)

        disc, risk, is_enc = classify_discrepancy_and_risk(
            iou=metrics["iou"],
            extension_ratio=metrics["extension_ratio"],
            area_diff_pct=metrics["area_difference_pct"]
        )
        self.assertEqual(disc, "MATCH")
        self.assertEqual(risk, "LOW")
        self.assertFalse(is_enc)

    def test_small_boundary_mismatch(self):
        """Test small boundary discrepancy (IoU around 0.75 - 0.80)."""
        cand_poly = Polygon([(0, 0), (110, 0), (110, 100), (0, 100)]) # 110x100 (Area 11,000)
        metrics = calculate_change_metrics(self.cadastral_poly, cand_poly)

        # Intersection = 10,000, Union = 11,000 => IoU = 10/11 ~ 0.909
        # Let's test moderate mismatch
        cand_poly2 = Polygon([(0, 0), (80, 0), (80, 100), (0, 100)]) # Area 8,000
        metrics2 = calculate_change_metrics(self.cadastral_poly, cand_poly2)
        # Intersection = 8,000, Union = 10,000 => IoU = 0.80

        disc, risk, is_enc = classify_discrepancy_and_risk(
            iou=metrics2["iou"],
            extension_ratio=metrics2["extension_ratio"],
            area_diff_pct=metrics2["area_difference_pct"]
        )
        self.assertEqual(disc, "MINOR BOUNDARY MISMATCH")
        self.assertIn(risk, ["LOW", "MEDIUM"])
        self.assertFalse(is_enc)

    def test_large_boundary_mismatch(self):
        """Test significant boundary mismatch (IoU < 0.60 without external protrusion)."""
        cand_poly = Polygon([(0, 0), (45, 0), (45, 100), (0, 100)]) # Inside cadastral, area 4,500
        metrics = calculate_change_metrics(self.cadastral_poly, cand_poly)

        self.assertLess(metrics["iou"], 0.60)
        self.assertEqual(metrics["potential_extension_area_px"], 0.0)

        disc, risk, is_enc = classify_discrepancy_and_risk(
            iou=metrics["iou"],
            extension_ratio=metrics["extension_ratio"],
            area_diff_pct=metrics["area_difference_pct"]
        )
        self.assertEqual(disc, "SIGNIFICANT BOUNDARY MISMATCH")
        self.assertEqual(risk, "HIGH")
        self.assertFalse(is_enc)

    def test_potential_encroachment_protrusion(self):
        """Test candidate polygon extending meaningfully outside cadastral boundary."""
        # Cadastral: [0, 0] to [100, 100] (Area 10,000)
        # Candidate: [20, 0] to [130, 100] (Area 11,000, extends by 30x100 = 3,000 outside)
        cand_poly = Polygon([(20, 0), (130, 0), (130, 100), (20, 100)])
        metrics = calculate_change_metrics(self.cadastral_poly, cand_poly)

        self.assertGreater(metrics["potential_extension_area_px"], 2500)
        self.assertGreater(metrics["extension_ratio"], 0.20)

        disc, risk, is_enc = classify_discrepancy_and_risk(
            iou=metrics["iou"],
            extension_ratio=metrics["extension_ratio"],
            area_diff_pct=metrics["area_difference_pct"],
            encroachment_threshold=0.10
        )
        self.assertEqual(disc, "POTENTIAL ENCROACHMENT")
        self.assertEqual(risk, "HIGH")
        self.assertTrue(is_enc)
        self.assertIsNotNone(metrics["potential_extension_geom"])

    def test_no_candidate_match(self):
        """Test when no candidate structure is detected for a cadastral record."""
        metrics = calculate_change_metrics(self.cadastral_poly, None)
        self.assertEqual(metrics["iou"], 0.0)
        self.assertEqual(metrics["candidate_area_px"], 0.0)
        self.assertEqual(metrics["potential_extension_area_px"], 0.0)

        disc, risk, is_enc = classify_discrepancy_and_risk(
            iou=0.0,
            extension_ratio=0.0,
            area_diff_pct=100.0
        )
        self.assertEqual(disc, "UNMATCHED")
        self.assertFalse(is_enc)

    def test_detect_cadastral_changes_pipeline_and_queue(self):
        """Test the complete change detection pipeline and surveyor queue generation."""
        cadastral_list = [
            {"cadastral_id": "P-001", "geometry": Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])}, # Match
            {"cadastral_id": "P-002", "geometry": Polygon([(200, 0), (300, 0), (300, 100), (200, 100)])}, # Encroachment
            {"cadastral_id": "P-003", "geometry": Polygon([(400, 0), (500, 0), (500, 100), (400, 100)])}, # Unmatched
        ]

        candidate_list = [
            {"parcel_id": "C-001", "geometry": Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])}, # Exact match with P-001
            {"parcel_id": "C-002", "geometry": Polygon([(220, 0), (330, 0), (330, 100), (220, 100)])}, # Protrusion with P-002
        ]

        result = detect_cadastral_changes(cadastral_list, candidate_list)

        self.assertEqual(result["total_cadastral"], 3)
        self.assertEqual(result["total_candidates"], 2)
        self.assertEqual(result["matches_count"], 1)
        self.assertEqual(result["potential_encroachments_count"], 1)
        self.assertEqual(result["unmatched_count"], 1)

        # Verify Surveyor Verification Queue priority sorting (HIGH risk at top)
        queue = result["surveyor_queue"]
        self.assertEqual(len(queue), 3)
        self.assertEqual(queue[0]["parcel_id"], "P-002")
        self.assertEqual(queue[0]["risk_level"], "HIGH")
        self.assertEqual(queue[0]["verification_status"], "PENDING SURVEYOR REVIEW")


if __name__ == "__main__":
    unittest.main()
