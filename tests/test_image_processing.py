"""
Unit tests for aerial image loading, metadata extraction, and preprocessing.
"""
from io import BytesIO
import unittest
import numpy as np
from PIL import Image

from src.utils.image_processing import (
    load_image,
    get_image_metadata,
    preprocess_for_model,
    ImageProcessingError,
    SUPPORTED_FORMATS,
)


class TestImageProcessing(unittest.TestCase):
    def _create_sample_image(self, size=(100, 80), mode="RGB", fmt="PNG") -> BytesIO:
        """Helper to generate an in-memory sample test image."""
        img = Image.new(mode, size, color=(120, 180, 240) if mode == "RGB" else 128)
        buf = BytesIO()
        img.save(buf, format=fmt)
        buf.seek(0)
        return buf

    def test_valid_image_loading_png(self):
        buf = self._create_sample_image(size=(120, 90), mode="RGB", fmt="PNG")
        image = load_image(buf)
        self.assertIsInstance(image, Image.Image)
        self.assertEqual(image.size, (120, 90))
        self.assertEqual(image.mode, "RGB")

    def test_valid_image_loading_jpeg(self):
        buf = self._create_sample_image(size=(200, 150), mode="RGB", fmt="JPEG")
        image = load_image(buf)
        self.assertEqual(image.size, (200, 150))
        self.assertEqual(image.format, "JPEG")

    def test_valid_image_loading_tiff(self):
        buf = self._create_sample_image(size=(64, 64), mode="RGB", fmt="TIFF")
        image = load_image(buf)
        self.assertEqual(image.size, (64, 64))

    def test_rgb_conversion_from_rgba_and_grayscale(self):
        # Grayscale image
        gray_buf = self._create_sample_image(size=(50, 50), mode="L", fmt="PNG")
        gray_img = load_image(gray_buf)
        preprocessed = preprocess_for_model(gray_img)
        self.assertEqual(preprocessed["original_image"].mode, "RGB")
        self.assertEqual(preprocessed["original_np"].shape, (50, 50, 3))

        # RGBA image
        rgba_img = Image.new("RGBA", (40, 40), color=(255, 0, 0, 128))
        preprocessed_rgba = preprocess_for_model(rgba_img)
        self.assertEqual(preprocessed_rgba["original_image"].mode, "RGB")
        self.assertEqual(preprocessed_rgba["original_np"].shape, (40, 40, 3))

    def test_image_metadata_extraction(self):
        buf = self._create_sample_image(size=(300, 200), mode="RGB", fmt="PNG")
        raw_bytes = buf.getvalue()
        image = load_image(BytesIO(raw_bytes))
        meta = get_image_metadata(image, filename="drone_aerial_01.png", file_size_bytes=len(raw_bytes))

        self.assertEqual(meta["filename"], "drone_aerial_01.png")
        self.assertEqual(meta["width"], 300)
        self.assertEqual(meta["height"], 200)
        self.assertEqual(meta["resolution"], "300 × 200")
        self.assertEqual(meta["channels"], 3)
        self.assertEqual(meta["format"], "PNG")
        self.assertIn("file_size_formatted", meta)

    def test_preprocessing_pipeline_output(self):
        original_size = (640, 480)
        target_size = (512, 512)
        buf = self._create_sample_image(size=original_size, mode="RGB", fmt="PNG")
        image = load_image(buf)

        result = preprocess_for_model(image, target_size=target_size, normalize_imagenet=True)

        # 1. Original image preservation
        self.assertEqual(result["original_image"].size, original_size)
        self.assertEqual(result["original_np"].shape, (480, 640, 3))

        # 2. Resized copy
        self.assertEqual(result["resized_image"].size, target_size)
        self.assertEqual(result["resized_np"].shape, (512, 512, 3))

        # 3. Normalized tensor (C, H, W)
        tensor = result["normalized_tensor_np"]
        self.assertEqual(tensor.shape, (3, 512, 512))
        self.assertEqual(tensor.dtype, np.float32)

        # 4. Scale factors
        scale_x, scale_y = result["scale_factors"]
        self.assertAlmostEqual(scale_x, 640 / 512)
        self.assertAlmostEqual(scale_y, 480 / 512)

    def test_invalid_and_corrupted_input_handling(self):
        # Corrupted bytes
        corrupted_bytes = b"NOT_A_VALID_IMAGE_FILE_HEADER_XYZ123"
        with self.assertRaises(ImageProcessingError):
            load_image(corrupted_bytes)

        # Empty bytes
        with self.assertRaises(ImageProcessingError):
            load_image(b"")


if __name__ == "__main__":
    unittest.main()
