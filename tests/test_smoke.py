"""
Smoke tests for project initialization and modules.
"""
import unittest
from src import __version__
from src.segmentation import (
    load_segmentation_model,
    segment_image,
    get_class_statistics,
    create_colored_mask,
    create_segmentation_overlay,
    generate_color_palette,
    DEFAULT_MODEL_ID,
)
from src.geometry import extract_boundaries, extract_feature_binary_mask, render_boundary_overlay
from src.vectorization import (
    generate_candidate_parcels,
    generate_parcel_polygons,
    parcels_to_geojson_dict,
    render_parcel_overlay,
)
from src.change_detection import detect_cadastral_changes
from src.utils import format_area_sqm, export_geojson, load_image, get_image_metadata, preprocess_for_model


class TestCadastralPrototypeSmoke(unittest.TestCase):
    def test_version(self):
        self.assertEqual(__version__, "0.1.0")

    def test_segmentation_exports(self):
        self.assertTrue(callable(load_segmentation_model))
        self.assertTrue(callable(segment_image))
        self.assertTrue(callable(get_class_statistics))
        self.assertTrue(callable(create_colored_mask))
        self.assertTrue(callable(create_segmentation_overlay))
        self.assertTrue(callable(generate_color_palette))
        self.assertEqual(DEFAULT_MODEL_ID, "nvidia/segformer-b0-finetuned-ade-512-512")

    def test_geometry_exports(self):
        self.assertTrue(callable(extract_boundaries))
        self.assertTrue(callable(extract_feature_binary_mask))
        self.assertTrue(callable(render_boundary_overlay))

    def test_vectorization_exports(self):
        self.assertTrue(callable(generate_candidate_parcels))
        self.assertTrue(callable(generate_parcel_polygons))
        self.assertTrue(callable(parcels_to_geojson_dict))
        self.assertTrue(callable(render_parcel_overlay))

    def test_change_detection_skeleton(self):
        res = detect_cadastral_changes([], [])
        self.assertIn("matched", res)
        self.assertIn("encroachments", res)

    def test_utils(self):
        formatted = format_area_sqm(10500.5)
        self.assertIn("sq.m", formatted)
        self.assertIn("ha", formatted)

    def test_image_processing_exports(self):
        self.assertTrue(callable(load_image))
        self.assertTrue(callable(get_image_metadata))
        self.assertTrue(callable(preprocess_for_model))


if __name__ == "__main__":
    unittest.main()
