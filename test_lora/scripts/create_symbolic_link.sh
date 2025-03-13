#!/bin/bash

DATASET_PATH=$1
SUFFIX=$2

# Check if all required arguments are provided
if [ -z "$DATASET_PATH" ] || [ -z "$SUFFIX" ]; then
    echo "Error: DATASET_PATH, SCENE, SUFFIX are not specified."
    echo "Usage: ./create_symbolic_link.sh <DATASET_PATH> <SUFFIX>"
    exit 1
fi

# Create main symlink
if [ -L "data/mapfree_processed" ]; then
    rm "data/mapfree_processed"
fi
ln -s "$DATASET_PATH" data/mapfree_processed

# Create scene-specific pairs symlink
TARGET_FILE="mapfree_pairs_${SUFFIX}.npy"  # Fix variable interpolation

LINK_PATH="data/mapfree_processed/train/mapfree_pairs.npy"
if [ -L "$LINK_PATH" ]; then
    rm "$LINK_PATH"
fi
ln -s "$DATASET_PATH/finetune_duster_calib_pretrain/pairs/$TARGET_FILE" "$LINK_PATH"
echo "Created symbolic link: $LINK_PATH -> $DATASET_PATH/finetune_duster_calib_pretrain/pairs/$TARGET_FILE"
