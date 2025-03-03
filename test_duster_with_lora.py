#!/usr/bin/env python3
import argparse
import torch
import logging
from pathlib import Path
from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import load_images, make_pairs
from dust3r.image_pairs import GlobalAlignerMode, global_aligner
from dust3r.utils import set_logging
from dust3r.lora import LoraLayer, inject_lora

class Dust3RPipeline:
    def __init__(self, model_path, lora_path=None, device='cuda'):
        self.device = device
        self.model = self._load_model(model_path)
        self._integrate_lora(lora_path)
        self.calib_params = None  # Add calibration params if needed
        self.verbose = True

    def _load_model(self, model_path):
        """Load base model weights"""
        model = AsymmetricCroCo3DStereo.from_pretrained(model_path)
        model = model.to(self.device)
        model.eval()
        print(f'Model Parameters: {sum(p.numel() for p in model.parameters()):,}')
        return model

    def _integrate_lora(self, lora_path):
        """Integrate LoRA weights if provided"""
        if lora_path:           
            # Phase 1: Inject LoRA adapters
            for name, module in self.model.named_modules():
                if any(n in name.split('.') for n in ['qkv']) and isinstance(module, torch.nn.Linear):
                    inject_lora(self.model, name, module)

            # Phase 2: Load LoRA weights
            try:
                lora_weights = torch.load(lora_path, map_location='cpu')
                self.model.load_state_dict(lora_weights, strict=False)
                print(f'LoRA Parameters: {sum(v.numel() for v in lora_weights.values()):,}')
            except Exception as e:
                raise RuntimeError(f"LoRA integration failed: {str(e)}")

    def process_images(self, img1_path, img2_path, resize=512):
        """Process pair of images and return aligned scene"""
        # Load and prepare images
        imgs_path = [str(img1_path), str(img2_path)]
        images = load_images(imgs_path, size=resize, verbose=self.verbose)
        pairs = make_pairs(images, scene_graph="complete", prefilter=None, symmetrize=True)
        
        # Run inference
        output = inference(pairs, self.model, self.device, batch_size=1, verbose=self.verbose)
        
        # Global alignment
        scene = global_aligner(
            output, 
            device=self.device,
            mode=GlobalAlignerMode.ModularPointCloudOptimizer,
            verbose=self.verbose,
            conf='log',
            calib_params=self.calib_params
        )
        
        return scene

def main():
    set_logging(level=logging.INFO)
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Dust3R with LoRA integration')
    parser.add_argument('--model_path', required=True, help='Path to base model weights')
    parser.add_argument('--lora_path', help='Path to LoRA weights (optional)')
    parser.add_argument('--img1', required=True, help='Path to first image')
    parser.add_argument('--img2', required=True, help='Path to second image')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (cuda/cpu)')
    args = parser.parse_args()

    # Initialize pipeline
    pipeline = Dust3RPipeline(
        model_path=args.model_path,
        lora_path=args.lora_path,
        device=args.device
    )

    # Process images
    try:
        scene = pipeline.process_images(args.img1, args.img2)
        
        # Example of using the aligned scene
        focals, im_poses = scene.get_focals(), scene.get_im_poses()
        print(f"Estimated focals: {focals}")
        print(f"Estimated poses: {im_poses}")
        
        # You could add visualization or saving logic here
        scene.show()
        
    except Exception as e:
        logging.error(f"Processing failed: {str(e)}")
        raise

if __name__ == '__main__':
    main()
