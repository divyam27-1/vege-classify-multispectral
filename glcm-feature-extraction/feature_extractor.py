import numpy as np
from skimage.feature.texture import graycomatrix, graycoprops
import matplotlib.pyplot as plt
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.stats import skew, kurtosis

cwd = os.getcwd()

dataset_partition = '/media/iittp/new volume'
tiles_dir = 'multispectral_tiles/'
dataset_dir = os.path.join(dataset_partition, tiles_dir)
features_dir = os.path.join(cwd, 'multispectral_glcm_vege_features/')
os.makedirs(features_dir, exist_ok=True)

classes = [os.path.splitext(f)[0] for f in os.listdir(dataset_dir) if f.endswith('.npz')]
print(classes)

# Function to load .npz file
def load_images(npz_file):
    data = np.load(npz_file)
    images = data['tiles']
    return images

# Function to extract GLCM features from a single image
def extract_glcm_features(image, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
    # image needs to be in uint8 format for graycomatrix

    glcm = graycomatrix(image, distances=distances, angles=angles, symmetric=True, normed=True)

    features = {
        'contrast': graycoprops(glcm, 'contrast'),
        'dissimilarity': graycoprops(glcm, 'dissimilarity'),
        'homogeneity': graycoprops(glcm, 'homogeneity'),
        'correlation': graycoprops(glcm, 'correlation'),
        'energy': graycoprops(glcm, 'energy'),
        'entropy': -np.sum(glcm * np.log2(glcm + 1e-10)),
        'variance': np.var(glcm),
        'max_prob': np.max(glcm),
    }

    # Flatten and return all feature values
    feature_values = []
    for prop in features.values():
        feature_values.extend(prop.flatten())

    return feature_values

def calculate_ndvi(NIR, Red):
    """Calculate NDVI from NIR and Red bands."""
    return (NIR - Red) / (NIR + Red + 1e-4)
def calculate_ndre(NIR, RE):
    """Calculate NDRE from NIR and Red Edge bands."""
    return (NIR - RE) / (NIR + RE + 1e-4)
def calculate_gndvi(NIR, Green):
    """Calculate GNDVI from NIR and Green bands."""
    return (NIR - Green) / (NIR + Green + 1e-4)
def calculate_savi(NIR, Red, L=0.5):
    """Calculate SAVI from NIR and Red bands with soil adjustment factor L."""
    return ((NIR - Red) * (1 + L)) / (NIR + Red + L)
def calculate_evi2(NIR, Red, G=2.5, C1=6, L=10000):
    """Calculate EVI2 from NIR and Red bands."""
    return G * (NIR - Red) / (NIR + C1 * Red + L + 1e-4)
def calculate_cvi(NIR, Green):
    """Calculate CVI from NIR and Green bands."""
    return NIR / (Green + 1e-4)

def summarize(matrix):
    """Convert a HxW vegetation index matrix into useful scalar features."""
    matrix = matrix.astype(float).flatten()
    return [
        np.nanmean(matrix),
        np.nanstd(matrix),
        np.nanmin(matrix),
        np.nanmax(matrix),
        np.nanmedian(matrix),
        np.nanpercentile(matrix, 25),
        np.nanpercentile(matrix, 75),
        skew(matrix, nan_policy='omit'),
        kurtosis(matrix, nan_policy='omit')
    ]

def calculate_vegetation_indices(g, r, re, nir):
    """Return scalar statistical features from each vegetation index."""
    
    indices = {
        'NDVI': calculate_ndvi(nir, r),
        'NDRE': calculate_ndre(nir, re),
        'GNDVI': calculate_gndvi(nir, g),
        'SAVI': calculate_savi(nir, r),
        'EVI2': calculate_evi2(nir, r),
        'CVI': calculate_cvi(nir, g),
    }

    feature_vector = []

    for name, vi_matrix in indices.items():
        feature_vector.extend(summarize(vi_matrix))

    return feature_vector

# Function to process all images in a .npz file and extract all features
def process_images(npz_file):
    # Load images from the .npz file
    images = load_images(npz_file)
    print(f"Loaded file {npz_file}")

    # Number of images in the dataset
    N = images.shape[0]

    all_features = []
    feature_length = None  # We will define a fixed feature length after the first image

    # Process each image
    for i in range(N):
        image = images[i]
        H, W, C = image.shape

        # Create a list to store GLCM features for all channels (G, R, RE, NIR)
        image_features = []

        # Loop through all 4 channels (G, R, RE, NIR) which are indices 0, 1, 2, 3
        for channel_idx in range(C):
            # Extract the current channel
            channel = image[:, :, channel_idx]

            # Extract GLCM features from this channel
            glcm_features = extract_glcm_features(channel)

            # Append the features for this channel to the image's feature list
            image_features.extend(glcm_features)  # Add the feature vector of this channel to the image's feature list

        num_glcm_features = len(image_features)

        g,r,re,nir = [image[:,:,i] for i in range(C)]
        vegetative_features = calculate_vegetation_indices(g,r,re,nir)
        image_features.extend(vegetative_features)

        num_vege_features = len(vegetative_features)

        #print(f"GLCM: {num_glcm_features}, VEGE: {num_vegetative_features}")

        # If it's the first image, set the feature length to compare against
        if feature_length is None:
            feature_length = len(image_features)

        # Ensure all feature vectors are the same length
        if len(image_features) != feature_length:
            print(f"Warning: Feature length mismatch for image {i}. Padding with zeros.")
            # If feature length is smaller, pad with zeros
            image_features.extend([0] * (feature_length - len(image_features)))

        # Append the features for this image to the all_features list
        all_features.append(image_features)

    # Convert to a numpy array (each row is a set of features for one image)
    return np.array(all_features)

with ProcessPoolExecutor(max_workers=len(classes)) as executor:
    futures = [
        executor.submit(process_images, os.path.join(dataset_dir, f"{cls}.npz")) 
        for cls in classes
    ] 
    
    for future in as_completed(futures): 
        features = future.result() 
        cls = classes[futures.index(future)] 
        outpath = os.path.join(features_dir, f'{cls}.npz') 
        np.savez_compressed(outpath, features=features)
        print(f"Class: {cls}, GLCM Features Shape:", features.shape)