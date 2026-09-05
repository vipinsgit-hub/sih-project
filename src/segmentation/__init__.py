"""Feature segmentation submodule."""
from .inference import (
    load_segmentation_model,
    segment_image,
    get_class_statistics,
    create_colored_mask,
    create_segmentation_overlay,
    generate_color_palette,
    DEFAULT_MODEL_ID,
)

__all__ = [
    "load_segmentation_model",
    "segment_image",
    "get_class_statistics",
    "create_colored_mask",
    "create_segmentation_overlay",
    "generate_color_palette",
    "DEFAULT_MODEL_ID",
]
