#!/bin/bash

# Common variables
PLACE="hkust_P000_N001"
ROOT="/Rocket_ssd/dataset/data_litevloc/map_free_eval/hkust_aria/$PLACE/map_free_eval"
MODEL_WEIGHTS_DIR="/Rocket_ssd/image_matching_model_weights"
PRETRAINED_MODEL="$MODEL_WEIGHTS_DIR/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
SUFFIXES=("10pdepth")

# Training hyperparameters (extracted for clarity)
TRAIN_ARGS=(
    "--train_dataset" "1_000 @ MapFree(split='train', ROOT='$ROOT/', aug_crop=16, resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], transform=ColorJitter)"
    "--test_dataset" "1_000 @ MapFree(split='train', ROOT='$ROOT/', resolution=(512, 384), seed=777)"
    "--model" "AsymmetricCroCo3DStereo(pos_embed='RoPE100', patch_embed_cls='ManyAR_PatchEmbed', img_size=(512, 512), head_type='dpt', output_mode='pts3d', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12)"
    "--train_criterion" "ConfLoss(Regr3D(L21, norm_mode='avg_dis'), alpha=0.2)"
    "--test_criterion" "Regr3D_ScaleShiftInv(L21, gt_scale=False)"
    "--pretrained" "$PRETRAINED_MODEL"
    "--lr" "0.0001"
    "--min_lr" "1e-06"
    "--warmup_epochs" "1"
    "--epochs" "10"
    "--batch_size" "1"
    "--accum_iter" "8"
    "--save_freq" "0"
    "--keep_freq" "0"
    "--eval_freq" "1"
    "--disable_cudnn_benchmark"
)

if [ -L "data/mapfree_processed" ]; then
    rm "data/mapfree_processed"
fi    
ln -s $ROOT data/mapfree_processed

for suffix in "${SUFFIXES[@]}"; do
    echo "Processing $suffix..."
    
    if [ -L "$ROOT/train/mapfree_pairs.npy" ]; then
        rm "$ROOT/train/mapfree_pairs.npy"
    fi    
    ln -s "$ROOT/train/mapfree_pairs_$suffix.npy" "$ROOT/train/mapfree_pairs.npy"
    
    # Training command
    OUTPUT_DIR="$MODEL_WEIGHTS_DIR/dust3r_"$PLACE"_512dpt_calib_ftlora_$suffix"
    python train.py "${TRAIN_ARGS[@]}" --output_dir "$OUTPUT_DIR"
    
    # Post-processing
    rm "$OUTPUT_DIR"/*.pth
    cp "$OUTPUT_DIR/lora.pt" "$ROOT/train/lora_$suffix.pt"
    rm "$ROOT/train/mapfree_pairs.npy"
    
    echo "Completed processing $suffix"
done

echo "All training runs completed!"
