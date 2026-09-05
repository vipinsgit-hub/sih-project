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
from src.geospatial import (
    load_cadastral_geojson,
    get_cadastral_statistics,
    calculate_spatial_overlap,
    classify_discrepancy,
    compare_cadastral_vs_candidate_parcels,
)
from src.visualization import render_gis_comparison_map, render_parcel_detail_comparison
from src.change_detection import (
    detect_cadastral_changes,
    calculate_change_metrics,
    classify_discrepancy_and_risk,
)
from src.utils import format_area_sqm, export_geojson, encroachments_to_geojson_dict, load_image, get_image_metadata, preprocess_for_model


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

    def test_geospatial_and_gis_exports(self):
        self.assertTrue(callable(load_cadastral_geojson))
        self.assertTrue(callable(get_cadastral_statistics))
        self.assertTrue(callable(calculate_spatial_overlap))
        self.assertTrue(callable(classify_discrepancy))
        self.assertTrue(callable(compare_cadastral_vs_candidate_parcels))
        self.assertTrue(callable(render_gis_comparison_map))
        self.assertTrue(callable(render_parcel_detail_comparison))

    def test_change_detection_exports(self):
        self.assertTrue(callable(detect_cadastral_changes))
        self.assertTrue(callable(calculate_change_metrics))
        self.assertTrue(callable(classify_discrepancy_and_risk))
        res = detect_cadastral_changes([], [])
        self.assertIn("change_records", res)
        self.assertIn("surveyor_queue", res)

    def test_utils(self):
        formatted = format_area_sqm(10500.5)
        self.assertIn("sq.m", formatted)
        self.assertIn("ha", formatted)
        self.assertTrue(callable(encroachments_to_geojson_dict))


    def test_image_processing_exports(self):
        self.assertTrue(callable(load_image))
        self.assertTrue(callable(get_image_metadata))
        self.assertTrue(callable(preprocess_for_model))


if __name__ == "__main__":
    unittest.main()
