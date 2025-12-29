import numpy as np
import cv2
import json
import os

def convert_band_to_pixel(img: np.ndarray, black_level: int) -> np.ndarray:
    """
    Converts raw band reflectance values to 8-bit pixel values. This follows an approximate conversion as per official DJI Documentation.
    
    Args:
        img: 2D numpy array of raw band reflectance values.
        black_level: Integer representing the black level for the band. To be retrieved from image metadata.
    
    Returns:
        arr: 2D numpy array of uint8 pixel values
    """
    arr = (img - black_level) / 65536.5
    arr = np.clip(arr, 0, 1)
    arr = (arr * 255).astype(np.uint8)
    return arr

def upscale_band(img: np.ndarray, scale: float) -> np.ndarray:
    """
    Upscales an image band by a scale factor using cubic interpolation
    
    Args:
        img: 2D numpy array representing image band
        scale: float scaling factor for upscaling (>1.0)
    
    Returns:
        A 2D numpy array of the upscaled image band
    """
    h, w = img.shape
    return cv2.resize(
        img,
        (int(w * scale), int(h * scale)),
        interpolation=cv2.INTER_CUBIC,
    )

def stack_bands(bands: dict[str, np.ndarray], order: list[str]) -> np.ndarray:
    """
    Stacks image bands into a single multi-channel image.

    Args:
        bands: Dictionary mapping band names to 2D numpy arrays.
            Should be in the form
            {
                "band_name_1": np.ndarray,
                "band_name_2": np.ndarray,
                ...
            }
        order: List of band names in the desired stacking order. These band names should match the required bands from keys of the `bands` dictionary.

    Returns:
        A 3D numpy array of the stacked image in the order specified.
    """
    return np.stack([bands[b] for b in order], axis=-1)

def tile_image(img: np.ndarray, tile_size: int) -> list[np.ndarray]:
    """
    Sequentially tiles an image into non overlapping square tiles of given size
    
    Args:
        img: 3D numpy array in H,W,C format representing the images to be tiled
        tile_size: int size of the square tiles to be extracted
    
    Returns:
        A list of 3D numpy arrays representing square tiles
    """
    h, w, c = img.shape
    tiles = []

    for y in range(0, h - tile_size + 1, tile_size):
        for x in range(0, w - tile_size + 1, tile_size):
            tiles.append(img[y:y+tile_size, x:x+tile_size])

    return tiles

def save_tiles_npz(tiles: list[np.ndarray], out_path: str):
    """
    Save a list of tiles into compressed NPZ file
    
    Args:
        tiles: List of 3D numpy ndarrays representing image tiles in H,W,C format
        out_path: Path to save NPZ file
        
    Returns:
        None. Image is saved in out_path in the array with key "tiles" in N,H,W,C format
    
    TODO: Verify if saving is done in N,H,W,C format or N,C,H,W format
    """
    arr = np.stack(tiles, axis=0)
    np.savez_compressed(os.path.join(os.getcwd(), out_path), tiles=arr)

def save_tile_metadata(metadata: dict, out_path: str):
    """
    Save tile metadata as a JSON file
    TODO: Make function to extract metadata from image file using exif and save metadata within the function instead of assuming metadata already defined.
    """
    with open(out_path, "w") as f:
        json.dump(metadata, f, indent=2)

def tile_multispectral_sample(
    bands: dict[str, np.ndarray],
    black_levels: dict[str, int],
    band_order: list[str],
    tile_size: int,
    upscale_scale: float,
) -> list[np.ndarray]:
    """
    Process and tile a multispectral image sample
    
    Args:
        bands: Dictionary mapping band names to 2D numpy arrays.
            Should be in the form
            {
                "band_name_1": np.ndarray,
                "band_name_2": np.ndarray,
                ...
            }
        black_levels: Dictionary mapping band names to their respective black levels.
            Should be in the form
            {
                "band_name_1": int,
                "band_name_2": int,
                ...
            }
        band_order: List of band names in the desired stacking order. These band names should match the required bands from keys of the `bands` dictionary.
        tile_size: int size of the square tiles to be extracted
        upscale_scale: float scaling factor for upscaling (>1.0)
    
    Returns:
        A list of 3D numpy arrays representing square tiles of the processed multispectral image
    
    TODO: Unit testing for tile_multispectral_sample() function in services/tiling.py
    """

    processed = {}

    for band, img in bands.items():
        px = convert_band_to_pixel(img, black_levels[band])
        px = upscale_band(px, upscale_scale)
        processed[band] = px

    stacked = stack_bands(processed, band_order)
    tiles = tile_image(stacked, tile_size)

    return tiles
