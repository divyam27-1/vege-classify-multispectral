#!/bin/bash

# Define the source and target directories
SRC_DIR="/media/iittp/new volume/crop_classification_dataset"
TARGET_DIR="$HOME/Documents/divyam-btp/crop-classification/data-preparation/multispectral_sample"

# Create the target directory if it doesn't exist
mkdir -p "$TARGET_DIR"

# Define the classes and bands (excluding RGB)
classes=("banana" "elephant_foot" "paddy" "turmeric")
bands=("GREEN" "NIR" "RE" "RED")

# Loop over each class
for class in "${classes[@]}"; do
    echo "Processing class: $class"

    TARGET_CLASS_DIR="$TARGET_DIR/$class"
    mkdir -p "$TARGET_CLASS_DIR"

    # Loop over each band (excluding RGB)
    for band in "${bands[@]}"; do
        echo "  Processing band: $band"

        # Define the directory containing the images for this class and band
        IMAGE_DIR="$SRC_DIR/$class/$band"
        
        # Check if the directory exists
        if [ ! -d "$IMAGE_DIR" ]; then
            echo "    Directory does not exist: $IMAGE_DIR"
            continue
        fi

        # Select files in the band directory, using `ls` and `sed` to get the range 120-150
        files=$(ls "$IMAGE_DIR" | sed -n '120,150p')

        # Loop over each file and copy it to the target directory
        for file in $files; do
            echo "    Copying $file..."
            cp "$IMAGE_DIR/$file" "$TARGET_CLASS_DIR/"
        done
    done
done

echo "Image extraction and copying completed!"
