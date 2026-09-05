"""
Unit tests for AI semantic segmentation and feature extraction module.
"""
import unittest
import numpy as np
from PIL import Image

from src.segmentation.inference import (
    load_segmentation_model,
    segment_image,
    get_class_statistics,
    create_colored_mask,
    create_segmentation_overlay,
    generate_color_palette,
    DEFAULT_MODEL_ID,
)


class TestSegmentation(unittest.TestCase):
    def test_palette_generation(self):
        palette = generate_color_palette(150)
        self.assertEqual(palette.shape, (150, 3))
        self.assertEqual(palette.dtype, np.uint8)

    def test_class_statistics(self):
        # Create a mock 100x100 mask with known classes
        mask = np.zeros((100, 100), dtype=np.int32)
        mask[:50, :] = 1  # 50% building
        mask[50:80, :] = 6  # 30% road
        mask[80:, :] = 9  # 20% grass

        id2label = {0: "wall", 1: "building", 6: "road", 9: "grass"}
        stats = get_class_statistics(mask, id2label)

        self.assertEqual(len(stats), 3)
        self.assertEqual(stats[0]["class_name"], "building")
        self.assertEqual(stats[0]["percentage"], 50.0)
        self.assertEqual(stats[1]["class_name"], "road")
        self.assertEqual(stats[1]["percentage"], 30.0)
        self.assertEqual(stats[2]["class_name"], "grass")
        self.assertEqual(stats[2]["percentage"], 20.0)

    def test_colored_mask_and_overlay(self):
        mask = np.zeros((64, 64), dtype=np.int32)
        mask[:32, :] = 1
        mask[32:, :] = 6

        color_img = create_colored_mask(mask)
        self.assertIsInstance(color_img, Image.Image)
        self.assertEqual(color_img.size, (64, 64))

        orig = Image.new("RGB", (64, 64), color=(100, 100, 100))
        overlay = create_segmentation_overlay(orig, mask, alpha=0.5)
        self.assertIsInstance(overlay, Image.Image)
        self.assertEqual(overlay.size, (64, 64))

    def test_model_loading_and_inference(self):
        # Test real model loading on CPU
        bundle = load_segmentation_model(DEFAULT_MODEL_ID, device="cpu")
        self.assertIn("model", bundle)
        self.assertIn("processor", bundle)
        self.assertIn("id2label", bundle)
        self.assertEqual(len(bundle["id2label"]), 150)

        # Test inference with a sample test image
        test_img = Image.new("RGB", (128, 96), color=(140, 175, 120))
        result = segment_image(test_img, bundle)

        self.assertIn("mask", result)
        self.assertIn("colored_mask", result)
        self.assertIn("overlay_image", result)
        self.assertIn("class_statistics", result)

        # Output mask dimensions must match original image (96, 128)
        self.assertEqual(result["mask"].shape, (96, 128))
        self.assertEqual(result["colored_mask"].size, (128, 96))
        self.assertEqual(result["overlay_image"].size, (128, 96))
        self.assertGreater(result["num_classes_detected"], 0)


if __name__ == "__main__":
    unittest.main()
