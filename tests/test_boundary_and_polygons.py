"""
Unit tests for boundary extraction, contour processing, polygon regularization, and parcel generation.
"""
import unittest
import numpy as np
from PIL import Image
from shapely.geometry import Polygon

from src.geometry.boundary import (
    extract_boundaries,
    extract_feature_binary_mask,
    clean_binary_mask,
    render_boundary_overlay,
)
from src.vectorization.polygons import (
    contour_to_shapely_polygon,
    validate_and_repair_geometry,
    simplify_and_regularize_polygon,
    generate_candidate_parcels,
    parcels_to_geojson_dict,
    render_parcel_overlay,
)


class TestBoundaryAndPolygons(unittest.TestCase):
    def setUp(self):
        # Create a synthetic 300x300 binary mask with:
        # - Two clean rectangular parcels
        # - One tiny 3x3 noise speckle (area = 9 px)
        self.mask = np.zeros((300, 300), dtype=np.uint8)

        # Parcel 1: 80x80 (area = 6400 px)
        self.mask[30:110, 30:110] = 255

        # Parcel 2: 60x100 (area = 6000 px)
        self.mask[150:250, 150:210] = 255

        # Noise speckle: 3x3 (area = 9 px)
        self.mask[10:13, 200:203] = 255

    def test_extract_feature_binary_mask(self):
        multi_class_mask = np.zeros((100, 100), dtype=np.int32)
        multi_class_mask[10:30, 10:30] = 1  # building
        multi_class_mask[40:60, 40:60] = 6  # road
        multi_class_mask[70:90, 70:90] = 0  # wall

        binary = extract_feature_binary_mask(multi_class_mask, target_class_ids=[1, 0])
        self.assertEqual(binary.dtype, np.uint8)
        self.assertEqual(binary[15, 15], 255)
        self.assertEqual(binary[45, 45], 0)
        self.assertEqual(binary[75, 75], 255)

    def test_boundary_extraction_and_noise_filtering(self):
        # With min_area=50, the 9px noise must be filtered out
        res = extract_boundaries(self.mask, min_area=50.0)

        self.assertIn("contours", res)
        self.assertEqual(res["total_regions_found"], 3)
        self.assertEqual(res["filtered_regions_count"], 2)
        self.assertEqual(res["rejected_count"], 1)
        self.assertEqual(len(res["contours"]), 2)

    def test_polygon_conversion_and_regularization(self):
        res = extract_boundaries(self.mask, min_area=50.0)
        contours = res["contours"]

        poly = contour_to_shapely_polygon(contours[0])
        self.assertIsInstance(poly, Polygon)
        self.assertTrue(poly.is_valid)

        simplified = simplify_and_regularize_polygon(poly, tolerance=2.0)
        self.assertIsInstance(simplified, Polygon)
        self.assertTrue(simplified.is_valid)
        self.assertGreater(simplified.area, 1000)

    def test_generate_candidate_parcels(self):
        res = extract_boundaries(self.mask, min_area=50.0)
        parcel_res = generate_candidate_parcels(res["contours"], min_area=50.0, id_prefix="P-")

        parcels = parcel_res["parcels"]
        self.assertEqual(len(parcels), 2)
        self.assertEqual(parcels[0]["parcel_id"], "P-001")
        self.assertEqual(parcels[1]["parcel_id"], "P-002")

        # Verify pixel measurements (strictly pixel units)
        self.assertGreater(parcels[0]["pixel_area"], 5000)
        self.assertGreater(parcels[0]["pixel_perimeter"], 100)
        self.assertEqual(parcels[0]["status"], "Candidate")
        self.assertEqual(parcels[0]["source"], "AI-Assisted Boundary Extraction")

    def test_geojson_serialization(self):
        res = extract_boundaries(self.mask, min_area=50.0)
        parcel_res = generate_candidate_parcels(res["contours"], min_area=50.0)
        geojson_doc = parcels_to_geojson_dict(parcel_res["parcels"])

        self.assertEqual(geojson_doc["type"], "FeatureCollection")
        self.assertEqual(len(geojson_doc["features"]), 2)
        feat = geojson_doc["features"][0]
        self.assertEqual(feat["id"], "P-001")
        self.assertIn("pixel_area", feat["properties"])
        self.assertEqual(feat["properties"]["unit"], "pixels")

    def test_visual_overlays(self):
        sample_img = Image.new("RGB", (300, 300), color=(150, 180, 140))
        res = extract_boundaries(self.mask, min_area=50.0)
        parcel_res = generate_candidate_parcels(res["contours"], min_area=50.0)

        # 1. Boundary overlay
        b_overlay = render_boundary_overlay(sample_img, res["contours"])
        self.assertIsInstance(b_overlay, Image.Image)
        self.assertEqual(b_overlay.size, (300, 300))

        # 2. Parcel overlay
        p_overlay = render_parcel_overlay(sample_img, parcel_res["parcels"])
        self.assertIsInstance(p_overlay, Image.Image)
        self.assertEqual(p_overlay.size, (300, 300))


if __name__ == "__main__":
    unittest.main()
