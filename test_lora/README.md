# Fine-Tuning Dust3R with LoRA Technique 

This repository implements the **LoRA3D** technique (from [ICLR2025 paper](https://openreview.net/forum?id=LSp4KBhAom)) to efficiently fine-tune the Dust3R model using Low-Rank Adaptation (LoRA). 
Perfect for limited compute resources!

## Key Features ✨
- 🚀 **Efficient Fine-Tuning** - Only 0.1% of parameters trained
- 🎯 **Single-Image Support** - Works with 10-20 images
- 🔄 **Seamless Integration** - Works with original Dust3R weights
- 📊 **Visualization Tools** - Confidence maps & weight analysis

## Setup Guide 

### Prerequisites
```bash
# Clone repository
git clone https://github.com/yourusername/dust3r-lora.git
cd dust3r-lora

# Install requirements
pip install -r requirements.txt
```

### Directory Structure
```
dust3r-lora/
├── dust3r/
│   ├── lora.py        # LoRA layer implementation
│   └── training.py    # Training logic
├── test_lora/
│   └── run_duster_with_lora.py  # Inference script
│   └── assets/            # Sample data & pre-trained weights
```

## Training Process 🏋️

### Configuration (lora.py)
```python
# LoRA Hyperparameters
LORA_ALPHA = 4    # Scaling factor for LoRA weights
LORA_R = 16       # Rank of low-rank matrices
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

### Start Training
```bash
# Single GPU example (adjust paths accordingly)
python train.py \
  --train_dataset "100 @ MapFree(...)" \
  --test_dataset "1 @ MapFree(...)" \
  --pretrained "/path/to/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth" \
  --output_dir "./outputs/lora_finetuned" \
  --batch_size 2 \
  --epochs 20 \
  --lr 0.001
```

### Key Training Components
1. **LoRA Injection** - Modifies cross-attention layers
2. **Parameter Freezing** - Only trains LoRA matrices
3. **Checkpointing** - Saves weights every 5 epochs

## Inference 🚀

### Basic Usage
```bash
# Without LoRA
python test_lora/run_duster_with_lora.py \
  --img1 assets/rgb_00000.jpg \
  --img2 assets/rgb_00001.jpg \
  --img3 assets/rgb_00002.jpg \
  --model_path DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth

# With LoRA
python test_lora/run_duster_with_lora.py \
  --img1 assets/rgb_00000.jpg \
  --img2 assets/rgb_00001.jpg \
  --img3 assets/rgb_00002.jpg \
  --model_path DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth \  
  --lora_path assets/lora_pdepth.pt
```

### Web Demo 🌐
```bash
python test_lora/demo_duster_with_lora.py \
  --weights DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth \  
  --lora_path assets/lora_pdepth.pt
```

## Pre-trained Weights ⚖️

| Weight Type          | Path                      | Description |
|----------------------|---------------------------| ----------- |
| Base Model           | `pretrained/dust3r_base.pth` | |
| LoRA (GT Depth)      | `assets/lora_gtdepth.pt`  | Trained with GT depth with 10 images|
| LoRA (Pseudo Depth)  |`assets/lora_pdepth.pt`   | Trained with pseudo depth with 10 images|

## Troubleshooting 🛠️
- **Module Not Found**: Ensure parent directory is in Python path
- **CUDA Memory Errors**: Reduce batch size (`--batch_size 1`)
- **Weights Loading Issues**: Verify model architecture matches
