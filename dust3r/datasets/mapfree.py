# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Modified MapFree dataloader using predefined pairs
# --------------------------------------------------------
import os.path as osp
import numpy as np
from glob import glob
from scipy.spatial.transform import Rotation

import sys
sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), "../../")))

from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from dust3r.utils.image import imread_cv2, rgb
import cv2

class MapFree(BaseStereoViewDataset):
	""" Dataset for MapFree evaluation using predefined pairs """
	
	def __init__(self, *args, ROOT, split='test', **kwargs):
		self.ROOT = ROOT
		super().__init__(*args, **kwargs)
		self.split = split
		self._load_pairs_and_scenes()

	def _load_pairs_and_scenes(self):
		# Load predefined pairs and organize scene data
		self.pairs = np.load(osp.join(self.ROOT, 'finetune', 'mapfree_pairs.npy'))
		self.scene_paths = {scene: osp.join(self.ROOT, self.split, scene) 
							for scene in np.unique(self.pairs['scene_name'])}
		
		# Preload scene parameters for faster access
		self.scene_data = {}
		for scene_name, scene_path in self.scene_paths.items():
			self.scene_data[scene_name] = {
				'intrinsics': self._load_intrinsics(osp.join(scene_path, 'intrinsics.txt')),
				'poses': self._load_poses(osp.join(scene_path, 'poses.txt'))
			}

	def _load_intrinsics(self, path):
		intrinsics = {}
		with open(path) as f:
			for line in f:
				parts = line.strip().split()
				frame_path = osp.join(osp.dirname(path), parts[0])
				intrinsics[frame_path] = np.array(list(map(float, parts[1:5])), dtype=np.float32)
		return intrinsics

	def _load_poses(self, path):
		poses = {}
		with open(path) as f:
			for line in f:
				parts = line.strip().split()
				frame_path = osp.join(osp.dirname(path), parts[0])
				qw, qx, qy, qz, tx, ty, tz = map(float, parts[1:8])
				rot = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
				pose = np.eye(4, dtype=np.float32)
				pose[:3, :3] = rot
				pose[:3, 3] = [tx, ty, tz]
				poses[frame_path] = pose
		return poses

	def __len__(self):
		return len(self.pairs)

	def get_stats(self):
		return f'{len(self.pairs)} predefined pairs from {len(self.scene_paths)} scenes'

	def _get_views(self, pair_idx, resolution, rng):
		pair = self.pairs[pair_idx]
		scene_name = pair['scene_name']
		scene_path = self.scene_paths[scene_name]
		scene_params = self.scene_data[scene_name]
		
		views = []
		for img_name, depth_name in zip([pair['img0'], pair['img1']], [pair['depth0'], pair['depth1']]):

			frame_path = osp.join(scene_path, img_name)
			image = imread_cv2(frame_path)

			depth_path = osp.join(scene_path, depth_name)
			depth_map = imread_cv2(depth_path, cv2.IMREAD_ANYDEPTH).astype(np.float32) / 1000.0
			
			# Get camera parameters
			intrinsics = scene_params['intrinsics'][frame_path]
			pose_w2c = scene_params['poses'][frame_path]
			pose_c2w = np.linalg.inv(pose_w2c)
			
			# Create intrinsics matrix
			fx, fy, cx, cy = intrinsics
			intrinsics_mat = np.array([
				[fx, 0, cx],
				[0, fy, cy],
				[0,  0,  1]
			], dtype=np.float32)
			
			# Apply preprocessing
			image, depth_map, intrinsics_mat = self._crop_resize_if_necessary(
				image, depth_map, intrinsics_mat, resolution, rng, info=frame_path)
			
			views.append({
				'img': image,
				'depthmap': depth_map,
				'camera_pose': pose_c2w,
				'camera_intrinsics': intrinsics_mat,
				'dataset': 'MapFree',
				'label': scene_name,
				'instance': img_name
			})
			
		return views

if __name__ == '__main__':
	import numpy as np
	from dust3r.datasets.base.base_stereo_view_dataset import view_name
	from dust3r.viz import SceneViz, auto_cam_size
	from dust3r.utils.image import rgb

	# Initialize MapFree dataset
	# NOTE(gogojjh): Users should specify the split name for training
	dataset = MapFree(
		ROOT="data/mapfree_processed",  # Path to MapFree data
		split='test',                   # Use finetune split
		resolution=224,                 # Target resolution
		aug_crop=16                     # Augmentation crop size (if needed)
	)

	# Visualize random samples
	for idx in np.random.permutation(len(dataset)):
		views = dataset[idx]
		assert len(views) == 2, "Each sample should contain 2 views"
		# print(f"Sample {idx}: {view_name(views[0])} vs {view_name(views[1])}")
		viz = SceneViz()
		poses = [views[i]['camera_pose'] for i in [0, 1]]
		cam_size = max(auto_cam_size(poses), 0.001)
		for view_idx in [0, 1]:
			pts3d = views[view_idx].get('pts3d', None)  # Will be None until inference
			valid_mask = views[view_idx]['valid_mask']
			colors = rgb(views[view_idx]['img'])
			if pts3d is not None:
				viz.add_pointcloud(pts3d, colors, valid_mask)                
			viz.add_camera(
				pose_c2w=views[view_idx]['camera_pose'],
				focal=views[view_idx]['camera_intrinsics'][0, 0],
				color=(idx * 50 % 255, (1 - idx/len(dataset)) * 255, 0),
				image=colors,
				cam_size=cam_size
			)
		viz.show()