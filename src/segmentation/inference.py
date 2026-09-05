"""
AI Semantic Segmentation and Feature Extraction Module for Aerial/Drone Imagery.
Uses lightweight SegFormer (nvidia/segformer-b0-finetuned-ade-512-512) for pixel-level semantic classification.
"""
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image
import torch

DEFAULT_MODEL_ID = "nvidia/segformer-b0-finetuned-ade-512-512"


def get_default_device() -> str:
    """Return 'cuda' if GPU is available, else 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def generate_color_palette(num_classes: int = 150) -> np.ndarray:
    """
    Generate a deterministic, visually distinct RGB color palette for semantic classes.
    """
    rng = np.random.RandomState(42)
    # Pre-defined prominent colors for common cadastral/urban features
    palette = rng.randint(40, 240, size=(num_classes, 3), dtype=np.uint8)
    
    # Custom tailored colors for key urban classes in ADE20K:
    # 0: wall -> light blue/slate
    # 1: building -> brick red / orange
    # 3: floor / ground -> beige
    # 4: tree -> forest green
    # 6: road -> dark gray
    # 9: grass -> bright green
    # 11: sidewalk -> light gray
    # 12: earth / ground -> brown
    # 21: water -> ocean blue
    # 32: fence -> purple / magenta
    if num_classes > 35:
        palette[0] = [120, 140, 180]   # wall
        palette[1] = [220, 80, 60]     # building
        palette[3] = [200, 190, 160]   # floor
        palette[4] = [34, 139, 34]     # tree
        palette[6] = [80, 80, 90]      # road
        palette[9] = [50, 205, 50]     # grass
        palette[11] = [160, 160, 170]  # sidewalk
        palette[12] = [165, 115, 60]   # earth
        palette[21] = [30, 144, 255]   # water
        palette[32] = [186, 85, 211]   # fence

    return palette


def load_segmentation_model(
    model_name: str = DEFAULT_MODEL_ID,
    device: Optional[str] = None
) -> Dict[str, Any]:
    """
    Load SegFormer processor and model for semantic segmentation.
    
    Args:
        model_name: Hugging Face model identifier.
        device: Target execution device ('cpu' or 'cuda').
        
    Returns:
        Dictionary bundle containing processor, model, id2label, and device.
    """
    if device is None:
        device = get_default_device()

    try:
        from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

        processor = SegformerImageProcessor.from_pretrained(model_name)
        model = SegformerForSemanticSegmentation.from_pretrained(model_name)
        model.to(device)
        model.eval()

        id2label = {int(k): str(v) for k, v in model.config.id2label.items()}

        return {
            "model": model,
            "processor": processor,
            "id2label": id2label,
            "model_name": model_name,
            "device": device,
            "palette": generate_color_palette(len(id2label)),
        }
    except Exception as e:
        raise RuntimeError(f"Failed to load segmentation model '{model_name}': {str(e)}") from e


def get_class_statistics(
    mask: np.ndarray,
    id2label: Dict[int, str],
    min_pixel_threshold: int = 10
) -> List[Dict[str, Any]]:
    """
    Calculate pixel counts and area percentages for each detected class in the mask.
    
    Args:
        mask: 2D integer numpy array of class indices.
        id2label: Mapping from class ID to human-readable label.
        min_pixel_threshold: Minimum pixel count to include in stats.
        
    Returns:
        List of dictionaries sorted by area percentage in descending order.
    """
    total_pixels = mask.size
    if total_pixels == 0:
        return []

    unique_ids, counts = np.unique(mask, return_counts=True)
    stats = []

    for class_id, count in zip(unique_ids, counts):
        if count < min_pixel_threshold:
            continue
        c_id = int(class_id)
        label = id2label.get(c_id, f"class_{c_id}")
        percentage = (count / total_pixels) * 100.0
        stats.append({
            "class_id": c_id,
            "class_name": label,
            "pixel_count": int(count),
            "percentage": round(percentage, 2),
        })

    # Sort descending by pixel count / percentage
    stats.sort(key=lambda x: x["pixel_count"], reverse=True)
    return stats


def create_colored_mask(
    mask: np.ndarray,
    palette: Optional[np.ndarray] = None
) -> Image.Image:
    """
    Convert a 2D integer class mask into a color RGB PIL Image.
    
    Args:
        mask: 2D integer numpy array (H, W).
        palette: (N, 3) uint8 numpy array of RGB colors.
        
    Returns:
        PIL.Image.Image in RGB mode.
    """
    if palette is None:
        palette = generate_color_palette(int(mask.max()) + 1 if mask.size > 0 else 150)

    # Clip mask values to palette size
    clipped_mask = np.clip(mask, 0, len(palette) - 1)
    colored_array = palette[clipped_mask]
    return Image.fromarray(colored_array.astype(np.uint8))


def create_segmentation_overlay(
    original_image: Union[Image.Image, np.ndarray],
    mask: np.ndarray,
    alpha: float = 0.45,
    palette: Optional[np.ndarray] = None
) -> Image.Image:
    """
    Blend the semantic segmentation color mask over the original image.
    
    Args:
        original_image: Original RGB PIL Image or numpy array.
        mask: 2D integer mask matching original image dimensions.
        alpha: Blending transparency (0.0 = only original, 1.0 = only mask).
        palette: Color palette.
        
    Returns:
        Blended PIL.Image.Image.
    """
    if isinstance(original_image, np.ndarray):
        orig_img = Image.fromarray(original_image).convert("RGB")
    else:
        orig_img = original_image.convert("RGB")

    orig_w, orig_h = orig_img.size

    # Ensure mask dimensions match original image
    if mask.shape != (orig_h, orig_w):
        mask_img = Image.fromarray(mask.astype(np.uint32))
        resized_mask_img = mask_img.resize((orig_w, orig_h), resample=Image.Resampling.NEAREST)
        mask = np.array(resized_mask_img)

    colored_mask = create_colored_mask(mask, palette=palette)
    return Image.blend(orig_img, colored_mask, alpha=alpha)


def segment_image(
    image: Union[Image.Image, np.ndarray],
    model_bundle: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Perform semantic segmentation inference on an aerial/drone image.
    
    Args:
        image: PIL Image or numpy array.
        model_bundle: Output bundle from load_segmentation_model().
        
    Returns:
        Dictionary containing:
            - mask: 2D numpy array of class indices (matching original image H, W).
            - colored_mask: PIL Image of the color-coded semantic mask.
            - overlay_image: PIL Image of the mask blended onto the original image.
            - class_statistics: List of detected class stats (pixels, %).
            - detected_classes: List of detected class label strings.
            - num_classes_detected: Total number of distinct classes identified.
    """
    if isinstance(image, np.ndarray):
        pil_image = Image.fromarray(image).convert("RGB")
    else:
        pil_image = image.convert("RGB")

    original_w, original_h = pil_image.size
    model = model_bundle["model"]
    processor = model_bundle["processor"]
    id2label = model_bundle["id2label"]
    device = model_bundle["device"]
    palette = model_bundle.get("palette", generate_color_palette(len(id2label)))

    # 1. Preprocess with model processor
    inputs = processor(images=pil_image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 2. Forward pass
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits  # shape: (1, num_classes, target_h, target_w)

        # 3. Upsample logits to original image resolution (H, W)
        upsampled_logits = torch.nn.functional.interpolate(
            logits,
            size=(original_h, original_w),
            mode="bilinear",
            align_corners=False
        )

        # 4. Argmax to get predicted class index per pixel
        pred_mask = upsampled_logits.argmax(dim=1)[0].cpu().numpy().astype(np.int32)

    # 5. Extract statistics
    stats = get_class_statistics(pred_mask, id2label)
    detected_classes = [s["class_name"] for s in stats]

    # 6. Generate visual representations
    colored_mask = create_colored_mask(pred_mask, palette=palette)
    overlay_image = create_segmentation_overlay(pil_image, pred_mask, alpha=0.45, palette=palette)

    return {
        "mask": pred_mask,
        "colored_mask": colored_mask,
        "overlay_image": overlay_image,
        "class_statistics": stats,
        "detected_classes": detected_classes,
        "num_classes_detected": len(detected_classes),
        "id2label": id2label,
        "model_name": model_bundle.get("model_name", DEFAULT_MODEL_ID),
        "device": device,
    }
