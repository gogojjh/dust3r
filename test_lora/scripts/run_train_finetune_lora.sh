#!/bin/bash

# Check if all required arguments are provided
if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ]; then
    echo "Error: DATASET_PATH, DATASET_NAME, SCENE are not specified."
    echo "Usage: ./run_train_finetune_lora.sh <DATASET_PATH> <DATASET_NAME> <SCENE>"
    exit 1
fi

# Assign arguments to variables
DATASET_PATH=$1
DATASET_NAME=$2
SCENE=$3
SUFFIXES=("pdepth" "gtdepth")
MODEL_WEIGHTS_DIR="/Rocket_ssd/image_matching_model_weights"
PRETRAINED_MODEL="$MODEL_WEIGHTS_DIR/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"

# Training hyperparameters
TRAIN_ARGS=(
    "--train_dataset" "100 @ MapFree(split='test', ROOT='$DATASET_PATH/', aug_crop=16, resolution=[(512, 288)], transform=ColorJitter)"
    "--test_dataset" "1 @ MapFree(split='test', ROOT='$DATASET_PATH/', resolution=[(512, 288)], seed=777)"
    "--model" "AsymmetricCroCo3DStereo(pos_embed='RoPE100', patch_embed_cls='ManyAR_PatchEmbed', img_size=(512, 512), head_type='dpt', output_mode='pts3d', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12)"
    "--train_criterion" "ConfLoss(Regr3D(L21, norm_mode='avg_dis'), alpha=0.2)"
    "--test_criterion" "Regr3D_ScaleShiftInv(L21, gt_scale=False)"
    "--pretrained" "$PRETRAINED_MODEL"
    "--lr" "0.001"
    "--min_lr" "0.000001"
    "--warmup_epochs" "0"
    "--epochs" "20"
    "--batch_size" "2"
    "--accum_iter" "8"
    "--save_freq" "0"
    "--keep_freq" "0"
    "--eval_freq" "1"
    "--disable_cudnn_benchmark"
)

# Prepare data folder
if [ -L "data/mapfree_processed" ]; then
    rm "data/mapfree_processed"
fi    
ln -s "$DATASET_PATH" data/mapfree_processed

# Start training
for SUFFIX in "${SUFFIXES[@]}"; do
    echo "Processing ${SUFFIX}..."
    
    if [ -L "${DATASET_PATH}/finetune/mapfree_pairs.npy" ]; then
        rm "${DATASET_PATH}/finetune/mapfree_pairs.npy"
    fi    
    ln -s "${DATASET_PATH}/finetune/pairs/mapfree_pairs_${SCENE}_${SUFFIX}.npy" "${DATASET_PATH}/finetune/mapfree_pairs.npy"
    
    # Training command
    OUTPUT_DIR="${MODEL_WEIGHTS_DIR}/dust3r_${DATASET_NAME}_${SCENE}_512dpt_calib_ftlora_${SUFFIX}_lora4_lr0001_512"
    python train.py "${TRAIN_ARGS[@]}" --output_dir "$OUTPUT_DIR"
    
    # Post-processing
    rm "$OUTPUT_DIR"/*.pth
    cp "$OUTPUT_DIR/lora.pt" "${DATASET_PATH}/finetune/weights/duster_lora_${SCENE}_${SUFFIX}.pt"
    rm "${DATASET_PATH}/finetune/mapfree_pairs.npy"
    
    echo "Completed processing ${SUFFIX}"
done

echo "All training runs completed!"