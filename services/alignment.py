import cv2
import numpy as np
from typing import Dict, Tuple

def translate_band(
    img: np.ndarray,
    dx: float,
    dy: float,
) -> np.ndarray:
    """
    Translate an image by (dx, dy) pixels using affine warp.

    Args:
        img: 2D image array
        dx: Shift along X-axis (pixels)
        dy: Shift along Y-axis (pixels)

    Returns:
        Shifted image array with same shape as input
    """
    M = np.float32([[1, 0, -dx], [0, 1, -dy]])
    return cv2.warpAffine(
        img,
        M,
        (img.shape[1], img.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

def to_edges(img: np.ndarray) -> np.ndarray:
    """
    Compute an edge-enhanced representation of an image using Sobel gradients.

    Args:
        img: 2D image array

    Returns:
        Edge magnitude image
    """
    img = img.astype(np.float32)
    img = cv2.GaussianBlur(img, (7, 7), 1.8)
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    return np.abs(gx) + np.abs(gy)

def downscale(img: np.ndarray, scale: float) -> np.ndarray:
    """
    Downscale an image by a given factor.

    Args:
        img: 2D image array
        scale: Downsampling factor (<1.0)

    Returns:
        Resized image
    """
    h, w = img.shape
    return cv2.resize(
        img,
        (int(w * scale), int(h * scale)),
        interpolation=cv2.INTER_AREA,
    )


def upscale(img: np.ndarray, scale: float) -> np.ndarray:
    """
    Upscale an image by a given factor.

    Args:
        img: 2D image array
        scale: Scaling factor (>1.0)

    Returns:
        Resized image
    """
    h, w = img.shape
    return cv2.resize(
        img,
        (int(w * scale), int(h * scale)),
        interpolation=cv2.INTER_CUBIC,
    )

def align_image_ecc(
    base_image: np.ndarray,
    input_image: np.ndarray,
    downscale_factor: float = 0.5,
) -> np.ndarray:
    """
    Align an image to a base image using ECC translation-only alignment.

    Args:
        base_image: Reference image
        input_image: Image to align
        downscale_factor: Factor used to downscale during alignment (<1.0)

    Returns:
        Aligned version of input_image
    """
    base_edges = to_edges(base_image)
    input_edges = to_edges(input_image)

    base_small = downscale(base_edges, downscale_factor)
    input_small = downscale(input_edges, downscale_factor)

    base_small /= np.percentile(base_small, 95) + 1e-7
    input_small /= np.percentile(input_small, 95) + 1e-7

    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        400,
        1e-6,
    )

    _, warp_matrix = cv2.findTransformECC(
        base_small,
        input_small,
        warp_matrix,
        cv2.MOTION_TRANSLATION,
        criteria,
    )

    return cv2.warpAffine(
        input_image,
        warp_matrix,
        input_image.shape[::-1],
        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
    )

def apply_relative_center_shifts(
    sample: dict,
    relative_centers: Dict[str, Tuple[float, float]],
) -> dict:
    """
    Apply per-band relative optical center shifts to a multispectral sample.

    Args:
        sample: Multispectral image sample with schema
            {
                "fname_base": str,
                "class": str,
                "bands": {band_name: image_array}
            }

        relative_centers: Mapping
            {
                band_name: (dx, dy)
            }

    Returns:
        Shifted sample with the same schema
    """
    shifted_bands = {}

    for band, img in sample["bands"].items():
        dx, dy = relative_centers[band]
        shifted_bands[band] = translate_band(img, dx, dy)

    return {
        "fname_base": sample["fname_base"],
        "class": sample.get("class"),
        "bands": shifted_bands,
    }

def apply_relative_center_shifts(
    sample: dict,
    relative_centers: Dict[str, Tuple[float, float]],
) -> dict:
    """
    Apply per-band relative optical center shifts to a multispectral sample.

    Args:
        sample: Multispectral image sample with schema
            {
                "fname_base": str,
                "class": str,
                "bands": {band_name: image_array}
            }

        relative_centers: Mapping
            {
                band_name: (dx, dy)
            }

    Returns:
        Shifted sample with the same schema
    """
    shifted_bands = {}

    for band, img in sample["bands"].items():
        dx, dy = relative_centers[band]
        shifted_bands[band] = translate_band(img, dx, dy)

    return {
        "fname_base": sample["fname_base"],
        "class": sample.get("class"),
        "bands": shifted_bands,
    }

def shift_batch_by_relative_centers(
    samples: list[dict],
    relative_centers: Dict[str, Tuple[float, float]],
) -> list[dict]:
    """
    Apply relative optical center shifts to a batch of multispectral samples.

    Args:
        samples: List of samples with schema
            {
                "fname_base": str,
                "class": str,
                "bands": {band_name: image_array}
            }

        relative_centers: Mapping
            {
                band_name: (dx, dy)
            }

    Returns:
        List of shifted samples with identical schema
    """
    return [
        apply_relative_center_shifts(sample, relative_centers)
        for sample in samples
    ]

