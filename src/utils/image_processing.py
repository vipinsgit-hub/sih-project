"""
Image loading, metadata extraction, and preprocessing module for aerial/drone imagery.
"""
from io import BytesIO
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
from PIL import Image, UnidentifiedImageError

SUPPORTED_FORMATS = {"JPEG", "JPG", "PNG", "TIFF", "TIF"}


class ImageProcessingError(Exception):
    """Custom exception for image processing errors."""
    pass


def load_image(file_source: Union[str, bytes, BytesIO]) -> Image.Image:
    """
    Safely load an image from a filepath, raw bytes, or BytesIO buffer.
    Converts and ensures standard RGB mode.
    
    Args:
        file_source: Path to file, bytes, or file-like object.
        
    Returns:
        PIL.Image.Image in RGB mode.
        
    Raises:
        ImageProcessingError: If the file is invalid, corrupted, or unsupported.
    """
    try:
        if isinstance(file_source, bytes):
            img_stream = BytesIO(file_source)
        elif isinstance(file_source, BytesIO):
            img_stream = file_source
        elif isinstance(file_source, str):
            with open(file_source, "rb") as f:
                img_stream = BytesIO(f.read())
        else:
            raise ImageProcessingError(f"Unsupported file source type: {type(file_source)}")

        img_stream.seek(0)
        image = Image.open(img_stream)
        image.load()  # Force load pixel data to catch corrupted files early

        # Verify format
        detected_format = (image.format or "").upper()
        if detected_format and detected_format not in SUPPORTED_FORMATS:
            raise ImageProcessingError(
                f"Unsupported image format: '{detected_format}'. "
                f"Supported formats are: {', '.join(sorted(SUPPORTED_FORMATS))}"
            )

        return image
    except UnidentifiedImageError as e:
        raise ImageProcessingError(f"Cannot identify or decode image file: {e}") from e
    except Exception as e:
        if isinstance(e, ImageProcessingError):
            raise
        raise ImageProcessingError(f"Failed to load image: {str(e)}") from e


def get_image_metadata(
    image: Image.Image,
    filename: Optional[str] = None,
    file_size_bytes: Optional[int] = None
) -> Dict[str, Any]:
    """
    Extract comprehensive metadata from a PIL Image object.
    
    Args:
        image: PIL Image object.
        filename: Optional filename of the uploaded image.
        file_size_bytes: Optional raw file size in bytes.
        
    Returns:
        Dictionary containing metadata properties.
    """
    width, height = image.size
    mode = image.mode
    format_name = image.format or "UNKNOWN"
    channels = len(image.getbands())

    meta: Dict[str, Any] = {
        "filename": filename or "unknown_image",
        "width": width,
        "height": height,
        "resolution": f"{width} × {height}",
        "channels": channels,
        "channel_names": list(image.getbands()),
        "mode": mode,
        "format": format_name,
        "aspect_ratio": round(width / height, 3) if height > 0 else 1.0,
    }

    if file_size_bytes is not None:
        meta["file_size_bytes"] = file_size_bytes
        if file_size_bytes < 1024 * 1024:
            meta["file_size_formatted"] = f"{file_size_bytes / 1024:.1f} KB"
        else:
            meta["file_size_formatted"] = f"{file_size_bytes / (1024 * 1024):.2f} MB"

    return meta


def preprocess_for_model(
    image: Image.Image,
    target_size: Tuple[int, int] = (512, 512),
    normalize_imagenet: bool = True
) -> Dict[str, Any]:
    """
    Preprocess an aerial image for AI model inference.
    Preserves original image unchanged and generates an AI-ready normalized copy.
    
    Args:
        image: Original PIL Image.
        target_size: (width, height) tuple for model input. Default is (512, 512).
        normalize_imagenet: If True, standardizes using ImageNet mean/std.
                            If False, normalizes to [0.0, 1.0].
                            
    Returns:
        Dictionary containing:
            - original_image: Original PIL Image (in RGB mode).
            - original_np: Original image as uint8 numpy array (H, W, 3).
            - resized_image: Resized PIL Image (target_size).
            - resized_np: Resized uint8 numpy array (target_H, target_W, 3).
            - normalized_tensor_np: Float32 normalized array shape (3, target_H, target_W).
            - target_size: (width, height).
            - scale_factors: (scale_x, scale_y) to map AI predictions back to original coordinates.
    """
    # 1. Ensure RGB mode without modifying original object in-place
    rgb_image = image.convert("RGB")
    original_w, original_h = rgb_image.size
    original_np = np.array(rgb_image, dtype=np.uint8)

    # 2. Resize AI-processing copy using high-quality resampling (BILINEAR / LANCZOS)
    target_w, target_h = target_size
    resized_image = rgb_image.resize((target_w, target_h), resample=Image.Resampling.BILINEAR)
    resized_np = np.array(resized_image, dtype=np.uint8)

    # 3. Convert to float32 and scale to [0, 1]
    float_array = resized_np.astype(np.float32) / 255.0

    # 4. Standardize / Normalize
    if normalize_imagenet:
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        norm_array = (float_array - mean) / std
    else:
        norm_array = float_array

    # 5. Transpose to Channel-First format (C, H, W) for PyTorch compatibility
    tensor_np = np.transpose(norm_array, (2, 0, 1))

    # 6. Coordinate scale factors (to project model predictions back to original image size)
    scale_x = original_w / target_w if target_w > 0 else 1.0
    scale_y = original_h / target_h if target_h > 0 else 1.0

    return {
        "original_image": rgb_image,
        "original_np": original_np,
        "resized_image": resized_image,
        "resized_np": resized_np,
        "normalized_tensor_np": tensor_np,
        "target_size": target_size,
        "scale_factors": (scale_x, scale_y),
    }
