import numpy as np
import cv2
import os
from collections import defaultdict
import json

from concurrent.futures import ProcessPoolExecutor, as_completed

cwd = os.getcwd()

dataset_dir = 'dataset_aligned'
tiles_dir = 'dataset_tiles'

multispectral_bands = ['r', 'g', 'nir', 're']

metadata = {}
with open('r_params.json', 'r') as f:
    metadata['r'] = json.load(f)[0]
with open('g_params.json', 'r') as f:
    metadata['g'] = json.load(f)[0]
with open('nir_params.json', 'r') as f:
    metadata['nir'] = json.load(f)[0]
with open('re_params.json', 'r') as f:
    metadata['re'] = json.load(f)[0]
    
BLACK_LEVEL = {
    'r': metadata['r']['BlackLevel'],
    'g': metadata['g']['BlackLevel'],
    'nir': metadata['nir']['BlackLevel'],
    're': metadata['re']['BlackLevel'],
}

LOADING_UPSAMPLE_SCALE = 2.0

def stack_bands(sample):
    """Stack bands in order [r, g, nir, re] into a single 3D array (H x W x 4)"""
    return np.stack([sample['g'], sample['r'], sample['re'], sample['nir']], axis=-1)

def tile_image(img, tile_size=512):
    h,w,c = img.shape
    assert c==4 # must have 4 channels
    
    tiles = []
    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            if (y + tile_size > h) or (x + tile_size > w):
                continue
            
            tile = img[y:y+tile_size, x:x+tile_size, :]
            if tile.shape[0] == tile_size and tile.shape[1] == tile_size:
                tiles.append(tile)

    return tiles

def convert_band_to_pixel(img, band):
    arr = (img - BLACK_LEVEL[band]) / 65536.0
    arr = np.clip(arr, 0, 1).astype(np.float16)
    arr = (arr * 255).astype(np.uint8)
    
    return arr

def upscale(img, scale=2.0):
    h, w = img.shape
    return cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)

images = []

# loads all images from dataset directory, with classes and file base names appended
for class_name in sorted(os.listdir(dataset_dir)):
    class_dir = os.path.join(dataset_dir, class_name)
    if not os.path.isdir(class_dir):
        continue

    samples = defaultdict(dict)
    for fname in os.listdir(class_dir):
        if not fname.lower().endswith('.tif'):
            continue

        base, ext = os.path.splitext(fname)
        base_name = None
        for band in multispectral_bands:
            if base.endswith(f"_{band.upper()}"):
                base_name = base[:-len(f"_{band}")]
                image = cv2.imread(
                    os.path.join(class_dir, fname), cv2.IMREAD_UNCHANGED
                )

                samples[base_name][band] = image
                samples[base_name]['fname_base'] = base_name
                samples[base_name]['class'] = class_name

    for sample in samples.values():
        if all(b in sample for b in multispectral_bands):
            images.append(sample)

if not os.path.exists(os.path.join(cwd, tiles_dir)):
    os.makedirs(os.path.join(cwd, tiles_dir))

tiles_by_class = {}
data_description = {}
for image in images:
    cls = image['class']
    for band in ['g','r','re','nir']:
        image_band_px = convert_band_to_pixel(image[band], band)
        image_band_px = upscale(image_band_px, LOADING_UPSAMPLE_SCALE)
        image[band] = image_band_px

    stacked = stack_bands(image)    
    tiles = tile_image(stacked, tile_size=512)

    if cls not in tiles_by_class:
        tiles_by_class[cls] = []

    if cls not in data_description:
        data_description[cls] = []

    tiles_by_class[cls].extend(tiles)
    data_description[cls].extend([f"{image['fname_base']}_{i:04d}" for i in range(len(tiles))])

print("\n".join(
    [f"Shape of tiles for class '{cls}': {len(cls_tiles)}x{cls_tiles[0].shape}" for cls, cls_tiles in tiles_by_class.items()]
))
del images

def save_class_tiles(cls, tiles_list):
    arr = np.stack(tiles_list, axis=0).astype(np.uint8)   # [N, 512, 512, 4]
    out_path = os.path.join(cwd, tiles_dir, f"{cls}.npz")
    np.savez_compressed(
        out_path,
        tiles=arr
    )
    print(f"Saved {len(tiles_list)} tiles for class '{cls}' → {out_path}")

for cls, tiles_list in tiles_by_class.items():
    save_class_tiles(cls, tiles_list)

with open(os.path.join(cwd, tiles_dir, 'data.json'), 'w') as data_description_file:
    json.dump(data_description, data_description_file)
    print('Data Description is saved to data.json')