"""
Smoke tests for project initialization and modules.
"""
import unittest
from src import __version__
from src.segmentation import run_segmentation_inference
from src.geometry import extract_boundaries
from src.vectorization import generate_parcel_polygons
from src.change_detection import detect_cadastral_changes
from src.utils import format_area_sqm, export_geojson


class TestCadastralPrototypeSmoke(unittest.TestCase):
    def test_version(self):
        self.assertEqual(__version__, "0.1.0")

    def test_segmentation_skeleton(self):
        res = run_segmentation_inference(image=None)
        self.assertIn("status", res)
        self.assertEqual(res["status"], "ready")

    def test_geometry_skeleton(self):
        res = extract_boundaries(mask=None)
        self.assertIsInstance(res, list)

    def test_vectorization_skeleton(self):
        res = generate_parcel_polygons(boundaries=[])
        self.assertIsInstance(res, list)

    def test_change_detection_skeleton(self):
        res = detect_cadastral_changes([], [])
        self.assertIn("matched", res)
        self.assertIn("encroachments", res)

    def test_utils(self):
        formatted = format_area_sqm(10500.5)
        self.assertIn("sq.m", formatted)
        self.assertIn("ha", formatted)


if __name__ == "__main__":
    unittest.main()
