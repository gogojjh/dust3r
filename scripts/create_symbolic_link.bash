#!/bin/bash

# Common variables
ROOT="/Rocket_ssd/dataset/data_litevloc/map_free_eval/hkust_aria/hkust_P000_N001/map_free_eval"
SUFFIX="10pdepth"

rm data/mapfree_processed
ln -s $ROOT data/mapfree_processed

rm data/mapfree_processed/train/mapfree_pairs.npy
ln -s $ROOT/train/mapfree_pairs_$SUFFIX.npy data/mapfree_processed/train/mapfree_pairs.npy

echo "Create symbolic link!"
