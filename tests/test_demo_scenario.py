"""
Verification test for the critical demo scenario and deterministic benchmark cases.
"""
import time
import unittest
from shapely.geometry import Polygon
from src.utils.image_processing import load_image, get_image_metadata
from src.geospatial.cadastral import load_cadastral_geojson
from src.change_detection.compare import detect_cadastral_changes


class TestDemoScenario(unittest.TestCase):
    """Verifies that the deterministic demo scenario produces exact expected benchmark results."""

    @classmethod
    def setUpClass(cls):
        cls.start_time = time.time()
        # Benchmark cadastral dataset: 5 plots (P-001 to P-005)
        cls.cad_parcels = [
            {"cadastral_id": "P-001", "geometry": Polygon([(50, 50), (200, 50), (200, 200), (50, 200)]), "pixel_area": 22500.0},
            {"cadastral_id": "P-002", "geometry": Polygon([(250, 50), (400, 50), (400, 200), (250, 200)]), "pixel_area": 22500.0},
            {"cadastral_id": "P-003", "geometry": Polygon([(50, 250), (200, 250), (200, 400), (50, 400)]), "pixel_area": 22500.0},
            {"cadastral_id": "P-004", "geometry": Polygon([(250, 250), (400, 250), (400, 400), (250, 400)]), "pixel_area": 22500.0},
            {"cadastral_id": "P-005", "geometry": Polygon([(450, 50), (530, 50), (530, 200), (450, 200)]), "pixel_area": 12000.0},
        ]

        # Benchmark candidate parcels:
        cls.candidate_parcels = [
            # C-001: Close Match with P-001 (extension < 5%, IoU > 90%)
            {"parcel_id": "C-001", "geometry": Polygon([(50, 50), (205, 50), (205, 200), (50, 200)]), "pixel_area": 23250.0},
            # C-002: Potential Encroachment with 4,500 px² protrusion
            {"parcel_id": "C-002", "geometry": Polygon([(220, 50), (400, 50), (400, 200), (220, 200)]), "pixel_area": 27000.0},
            # C-003: Potential Encroachment with 7,500 px² lateral protrusion
            {"parcel_id": "C-003", "geometry": Polygon([(50, 250), (250, 250), (250, 400), (50, 400)]), "pixel_area": 30000.0},
            # C-004: Significant boundary mismatch with half area
            {"parcel_id": "C-004", "geometry": Polygon([(250, 250), (325, 250), (325, 400), (250, 400)]), "pixel_area": 11250.0},
        ]

        cls.change_data = detect_cadastral_changes(cls.cad_parcels, cls.candidate_parcels)
        cls.total_time = time.time() - cls.start_time

    def test_pipeline_performance(self):
        """Change detection must execute instantly on CPU (< 0.5s)."""
        self.assertLess(self.total_time, 0.5)

    def test_p003_potential_encroachment(self):
        """P-003 must trigger POTENTIAL ENCROACHMENT with HIGH risk and 7,500 px² protrusion."""
        record = next((r for r in self.change_data["change_records"] if r["parcel_id"] == "P-003"), None)
        self.assertIsNotNone(record)
        self.assertEqual(record["discrepancy_type"], "POTENTIAL ENCROACHMENT")
        self.assertEqual(record["risk_level"], "HIGH")
        self.assertAlmostEqual(record["potential_extension_area_px"], 7500.0, delta=100.0)

    def test_p002_potential_encroachment(self):
        """P-002 must trigger POTENTIAL ENCROACHMENT with HIGH risk and 4,500 px² protrusion."""
        record = next((r for r in self.change_data["change_records"] if r["parcel_id"] == "P-002"), None)
        self.assertIsNotNone(record)
        self.assertEqual(record["discrepancy_type"], "POTENTIAL ENCROACHMENT")
        self.assertEqual(record["risk_level"], "HIGH")
        self.assertAlmostEqual(record["potential_extension_area_px"], 4500.0, delta=100.0)

    def test_p004_significant_boundary_mismatch(self):
        """P-004 must trigger SIGNIFICANT BOUNDARY MISMATCH with HIGH risk."""
        record = next((r for r in self.change_data["change_records"] if r["parcel_id"] == "P-004"), None)
        self.assertIsNotNone(record)
        self.assertEqual(record["discrepancy_type"], "SIGNIFICANT BOUNDARY MISMATCH")
        self.assertEqual(record["risk_level"], "HIGH")

    def test_p005_unmatched(self):
        """P-005 must be UNMATCHED with MEDIUM risk."""
        record = next((r for r in self.change_data["change_records"] if r["parcel_id"] == "P-005"), None)
        self.assertIsNotNone(record)
        self.assertIn("UNMATCHED", record["discrepancy_type"])
        self.assertEqual(record["risk_level"], "MEDIUM")

    def test_p001_match(self):
        """P-001 must be MATCH with LOW risk."""
        record = next((r for r in self.change_data["change_records"] if r["parcel_id"] == "P-001"), None)
        self.assertIsNotNone(record)
        self.assertEqual(record["discrepancy_type"], "MATCH")
        self.assertEqual(record["risk_level"], "LOW")
        self.assertGreaterEqual(record["overlap_percentage"], 85.0)



if __name__ == "__main__":
    unittest.main()
