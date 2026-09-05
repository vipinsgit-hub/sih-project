"""
Boundary and contour extraction module from semantic feature masks.
"""
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image

# Default ADE20K semantic classes useful for physical structure & parcel boundary detection
# 1: building, 0: wall, 32: fence, 25: house/roof, 48: pole, 84: tower
DEFAULT_PARCEL_FEATURE_CLASSES = [1, 0, 32, 25, 84]


def extract_feature_binary_mask(
    mask: np.ndarray,
    target_class_ids: Optional[List[int]] = None
) -> np.ndarray:
    """
    Extract a binary mask (uint8, 0 or 255) for selected semantic feature classes.
    
    Args:
        mask: 2D integer semantic segmentation mask.
        target_class_ids: List of class IDs to include. If None, uses default structure classes.
        
    Returns:
        2D uint8 numpy array with values 0 or 255.
    """
    if target_class_ids is None:
        target_class_ids = DEFAULT_PARCEL_FEATURE_CLASSES

    binary_mask = np.isin(mask, target_class_ids).astype(np.uint8) * 255
    return binary_mask


def clean_binary_mask(
    binary_mask: np.ndarray,
    open_kernel_size: int = 3,
    close_kernel_size: int = 5
) -> np.ndarray:
    """
    Apply morphological filtering to remove speckle noise and close small boundary gaps.
    
    Args:
        binary_mask: 2D uint8 binary mask.
        open_kernel_size: Kernel dimension for morphological opening (removes small noise).
        close_kernel_size: Kernel dimension for morphological closing (bridges gaps).
        
    Returns:
        Cleaned 2D uint8 binary mask.
    """
    cleaned = binary_mask.copy()

    # 1. Opening: remove isolated noise pixels
    if open_kernel_size > 1:
        open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (open_kernel_size, open_kernel_size))
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, open_kernel)

    # 2. Closing: bridge small cracks and holes inside structures
    if close_kernel_size > 1:
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_kernel_size, close_kernel_size))
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, close_kernel)

    return cleaned


def extract_boundaries(
    mask_or_binary: np.ndarray,
    target_class_ids: Optional[List[int]] = None,
    min_area: float = 100.0,
    max_area: Optional[float] = None,
    apply_morphology: bool = True
) -> Dict[str, Any]:
    """
    Extract closed boundary contours from a semantic mask or binary mask.
    
    Args:
        mask_or_binary: 2D numpy array (multi-class or binary).
        target_class_ids: Target class IDs if multi-class mask is provided.
        min_area: Minimum contour area in pixels to filter out noise.
        max_area: Optional maximum contour area threshold.
        apply_morphology: Whether to clean mask with morphology before contour extraction.
        
    Returns:
        Dictionary containing:
            - binary_mask: 2D uint8 binary mask.
            - contours: List of filtered OpenCV contours.
            - total_regions_found: Raw contour count before filtering.
            - filtered_regions_count: Count of contours retained after area filtering.
            - rejected_count: Count of discarded small/noise regions.
    """
    # Convert to binary if not already uint8 binary
    unique_vals = np.unique(mask_or_binary)
    if len(unique_vals) > 2 or (len(unique_vals) == 2 and not set(unique_vals).issubset({0, 255})):
        binary_mask = extract_feature_binary_mask(mask_or_binary, target_class_ids=target_class_ids)
    else:
        binary_mask = (mask_or_binary > 0).astype(np.uint8) * 255

    if apply_morphology:
        binary_mask = clean_binary_mask(binary_mask)

    # Find external contours
    raw_contours, _ = cv2.findContours(
        binary_mask,
        mode=cv2.RETR_EXTERNAL,
        method=cv2.CHAIN_APPROX_SIMPLE
    )

    total_found = len(raw_contours)
    filtered_contours = []
    rejected_count = 0

    for cnt in raw_contours:
        # Minimum point count for valid polygon
        if len(cnt) < 3:
            rejected_count += 1
            continue

        area = cv2.contourArea(cnt)
        if area < min_area:
            rejected_count += 1
            continue
        if max_area is not None and area > max_area:
            rejected_count += 1
            continue

        filtered_contours.append(cnt)

    return {
        "binary_mask": binary_mask,
        "contours": filtered_contours,
        "total_regions_found": total_found,
        "filtered_regions_count": len(filtered_contours),
        "rejected_count": rejected_count,
    }


def render_boundary_overlay(
    image: Union[Image.Image, np.ndarray],
    contours: List[np.ndarray],
    line_color: Tuple[int, int, int] = (0, 255, 255),
    thickness: int = 2
) -> Image.Image:
    """
    Render extracted boundary contours over the original drone image.
    
    Args:
        image: Original PIL Image or numpy array.
        contours: List of OpenCV contours.
        line_color: RGB tuple (e.g., yellow = (0, 255, 255) in BGR/RGB).
        thickness: Line stroke width in pixels.
        
    Returns:
        PIL.Image.Image with boundary lines rendered.
    """
    if isinstance(image, Image.Image):
        img_np = np.array(image.convert("RGB"))
    else:
        img_np = image.copy()

    # Draw contours using OpenCV
    cv2.drawContours(img_np, contours, -1, line_color, thickness)
    return Image.fromarray(img_np)
