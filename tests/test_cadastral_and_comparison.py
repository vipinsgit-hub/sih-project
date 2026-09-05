"""
Unit tests for Cadastral GeoJSON parsing, Spatial Comparison, Discrepancy Classification, and GIS Visualization.
"""
import os
import unittest
import numpy as np
from PIL import Image
from shapely.geometry import Polygon

from src.geospatial.cadastral import (
    load_cadastral_geojson,
    get_cadastral_statistics,
    CadastralDataError,
)
from src.geospatial.comparison import (
    calculate_spatial_overlap,
    classify_discrepancy,
    compare_cadastral_vs_candidate_parcels,
)
from src.visualization.map import render_gis_comparison_map


class TestCadastralAndComparison(unittest.TestCase):
    def setUp(self):
        self.demo_geojson_path = os.path.join("data", "cadastral", "demo_cadastral.geojson")

    def test_load_demo_cadastral_geojson(self):
        parcels = load_cadastral_geojson(self.demo_geojson_path)
        self.assertIsInstance(parcels, list)
        self.assertEqual(len(parcels), 5)

        first = parcels[0]
        self.assertIn(first["cadastral_id"], ["P-001", "CAD-001"])
        self.assertIsInstance(first["geometry"], Polygon)
        self.assertTrue(first["geometry"].is_valid)
        self.assertGreater(first["pixel_area"], 0)

    def test_cadastral_statistics(self):
        parcels = load_cadastral_geojson(self.demo_geojson_path)
        stats = get_cadastral_statistics(parcels)

        self.assertEqual(stats["total_parcels"], 5)
        self.assertGreater(stats["total_area_px"], 0)
        self.assertIn("land_use_breakdown", stats)

    def test_spatial_overlap_iou_cases(self):
        # Case 1: Identical Polygons (IoU = 1.0)
        poly_a = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
        poly_b = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
        iou, inter, union = calculate_spatial_overlap(poly_a, poly_b)
        self.assertEqual(iou, 1.0)
        self.assertEqual(inter, 10000.0)

        # Case 2: Disjoint / Non-overlapping (IoU = 0.0)
        poly_c = Polygon([(200, 200), (300, 200), (300, 300), (200, 300)])
        iou_disjoint, _, _ = calculate_spatial_overlap(poly_a, poly_c)
        self.assertEqual(iou_disjoint, 0.0)

        # Case 3: 50% shift overlap
        poly_d = Polygon([(50, 0), (150, 0), (150, 100), (50, 100)])
        iou_half, inter_half, _ = calculate_spatial_overlap(poly_a, poly_d)
        self.assertAlmostEqual(iou_half, 5000.0 / 15000.0, places=3)

    def test_discrepancy_classification_rules(self):
        # Match
        cat, is_enc = classify_discrepancy(iou=0.92, excess_ratio=0.05)
        self.assertEqual(cat, "MATCH")
        self.assertFalse(is_enc)

        # Minor Mismatch
        cat, is_enc = classify_discrepancy(iou=0.75, excess_ratio=0.08)
        self.assertEqual(cat, "MINOR MISMATCH")
        self.assertFalse(is_enc)

        # Boundary Mismatch
        cat, is_enc = classify_discrepancy(iou=0.45, excess_ratio=0.05)
        self.assertEqual(cat, "BOUNDARY MISMATCH")
        self.assertFalse(is_enc)

        # Potential Encroachment (excess ratio > 15%)
        cat, is_enc = classify_discrepancy(iou=0.55, excess_ratio=0.35)
        self.assertEqual(cat, "POTENTIAL ENCROACHMENT")
        self.assertTrue(is_enc)

    def test_compare_cadastral_vs_candidate_pipeline(self):
        cadastral_parcels = load_cadastral_geojson(self.demo_geojson_path)

        # Create candidate parcels simulating AI detections
        candidate_parcels = [
            # Candidate 1: Exact match with CAD-001 (80,80) to (240,180)
            {
                "parcel_id": "P-001",
                "geometry": Polygon([(80, 80), (240, 80), (240, 180), (80, 180)]),
                "pixel_area": 16000.0,
            },
            # Candidate 2: Encroaching outside CAD-003 (120,420)-(240,520) -> extending to (100,420)-(260,520)
            {
                "parcel_id": "P-003",
                "geometry": Polygon([(100, 420), (260, 420), (260, 520), (100, 520)]),
                "pixel_area": 16000.0,
            }
        ]

        comp_res = compare_cadastral_vs_candidate_parcels(cadastral_parcels, candidate_parcels)

        self.assertEqual(comp_res["total_cadastral"], 5)
        self.assertEqual(comp_res["total_candidates"], 2)
        self.assertGreater(comp_res["matches_count"], 0)
        self.assertGreater(comp_res["potential_encroachments_count"], 0)
        self.assertGreater(comp_res["unmatched_count"], 0)

    def test_gis_map_rendering(self):
        cadastral_parcels = load_cadastral_geojson(self.demo_geojson_path)
        candidate_parcels = [
            {
                "parcel_id": "P-001",
                "geometry": Polygon([(80, 80), (240, 80), (240, 180), (80, 180)]),
                "pixel_area": 16000.0,
            }
        ]
        comp_res = compare_cadastral_vs_candidate_parcels(cadastral_parcels, candidate_parcels)

        sample_img = Image.new("RGB", (800, 600), color=(140, 175, 120))
        gis_map = render_gis_comparison_map(
            sample_img,
            cadastral_parcels,
            candidate_parcels,
            comparison_results=comp_res["comparison_results"]
        )

        self.assertIsInstance(gis_map, Image.Image)
        self.assertEqual(gis_map.size, (800, 600))


if __name__ == "__main__":
    unittest.main()
