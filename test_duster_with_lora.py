#!/usr/bin/env python3

import argparse
import torch
import logging
from pathlib import Path
from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_images
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.lora import LoraLayer, inject_lora
import matplotlib.pyplot as plt

class Dust3RPipeline:
	def __init__(self, model_path, lora_path=None, device='cuda'):
		self.device = device
		self.model = self._load_model(model_path)

		if lora_path:           
			self._integrate_lora(lora_path)

		self.model = self.model.to(self.device)
		self.model.eval()

		self.calib_params = None  # Add calibration params if needed
		self.verbose = True

	def _load_model(self, model_path):
		"""Load base model weights"""
		model = AsymmetricCroCo3DStereo.from_pretrained(model_path)
		print(f'Model Parameters: {sum(p.numel() for p in model.parameters()):,}')
		return model

	def _integrate_lora(self, lora_path):
		"""Integrate LoRA weights if provided"""
		self.model = self.model.to('cpu')
		
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

		# Phase 3: Merge LoRA weights into base model
		for name, module in self.model.named_modules():
			if isinstance(module, LoraLayer):
				parent = self.model
				for component in name.split('.')[:-1]:
					parent = getattr(parent, component)
				
				# Mathematical merge: W' = W + (A*B)*(α/r)
				lora_weight = ((module.lora_a @ module.lora_b) * module.alpha / module.r).T
				merged_weight = module.raw_linear.weight + lora_weight
				module.raw_linear.weight.data.copy_(merged_weight)
				
				# Replace composite layer with merged linear layer
				setattr(parent, name.split('.')[-1], module.raw_linear)

	@torch.no_grad()
	def process_images(self, img1_path, img2_path, save_conf_path, resize=512):
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
		##### GlobalAlignerMode.ModularPointCloudOptimizer
		# loss = scene.compute_global_alignment(init="mst", niter=niter, schedule=schedule, lr=lr)
		##### GlobalAlignerMode.PairViewer
		scene = global_aligner(output, device=self.device, mode=GlobalAlignerMode.PairViewer)

		C_i = scene.conf_trf(scene.conf_i["0_1"]).cpu().numpy()
		C_j = scene.conf_trf(scene.conf_j["0_1"]).cpu().numpy()

		fig, axes = plt.subplots(1, 2, figsize=(10, 5))
		im = axes[0].imshow(C_i, cmap='jet')
		axes[0].set_title(f'Confidence Map {0}')
		fig.colorbar(im, ax=axes[0])
		im = axes[1].imshow(C_j, cmap='jet')
		axes[1].set_title(f'Confidence Map {0}')
		fig.colorbar(im, ax=axes[1])
		plt.savefig(save_conf_path)

		return scene

def main():
	# Parse arguments
	parser = argparse.ArgumentParser(description='Dust3R with LoRA integration')
	parser.add_argument('--model_path', required=True, help='Path to base model weights')
	parser.add_argument('--lora_path', help='Path to LoRA weights (optional)')
	parser.add_argument('--img1', required=True, help='Path to first image')
	parser.add_argument('--img2', required=True, help='Path to second image')
	parser.add_argument('--save_conf_path', required=True, help='Path to confidence map')
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
		scene = pipeline.process_images(args.img1, args.img2, args.save_conf_path)
		
		# Example of using the aligned scene
		focals, im_poses = scene.get_focals(), scene.get_im_poses()
		print(f"Estimated focals: {focals}")
		print(f"Estimated poses: {im_poses}")
		
		# You could add visualization or saving logic here
		scene.show(cam_size=0.1)
		
	except Exception as e:
		logging.error(f"Processing failed: {str(e)}")
		raise

if __name__ == '__main__':
	main()
