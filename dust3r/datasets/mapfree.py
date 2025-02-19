# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Dataloader for MapFree dataset
# --------------------------------------------------------
import os.path as osp
import numpy as np
import random
from glob import glob
from scipy.spatial.transform import Rotation

import sys
sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), "../../")))

from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from dust3r.utils.image import imread_cv2, rgb
import cv2

class MapFree(BaseStereoViewDataset):
	""" Dataset for MapFree evaluation - random pairs from scene sequences """
	
	def __init__(self, *args, ROOT, split='test', **kwargs):
		self.ROOT = ROOT
		super().__init__(*args, **kwargs)
		self.split = split
		self.num_data_each_scene = 10
		self._load_scenes()

	def _load_scenes(self):
		# Load all scenes from the specified split
		self.scenes = []
		scene_dirs = glob(osp.join(self.ROOT, self.split, 's*'))
		
		for scene_dir in scene_dirs:
			# Load all frames in seq1 (query frames)
			frames = sorted(glob(osp.join(scene_dir, 'seq1', 'frame_*.jpg')))
			if not frames:
				continue
				
			# Load camera parameters
			intrinsics = self._load_intrinsics(osp.join(scene_dir, 'intrinsics.txt'))
			poses = self._load_poses(osp.join(scene_dir, 'poses.txt'))
			
			self.scenes.append({
				'path': scene_dir,
				'frames': frames,
				'intrinsics': intrinsics,
				'poses': poses
			})

	def _load_intrinsics(self, path):
		# Load per-frame intrinsics: {frame_path: (fx, fy, cx, cy)}
		intrinsics = {}
		with open(path) as f:
			for line in f:
				parts = line.strip().split()
				frame_path = osp.join(osp.dirname(path), parts[0])
				intrinsics[frame_path] = np.array(list(map(float, parts[1:5])), dtype=np.float32)
		return intrinsics

	def _load_poses(self, path):
		# Load world-to-camera poses: {frame_path: (4x4 matrix)}
		poses = {}
		with open(path) as f:
			for line in f:
				parts = line.strip().split()
				frame_path = osp.join(osp.dirname(path), parts[0])
				qw, qx, qy, qz, tx, ty, tz = map(float, parts[1:8])
				
				# Convert quaternion to rotation matrix
				rot = self.quat_to_rotmat(qw, qx, qy, qz)
				pose = np.eye(4, dtype=np.float32)
				pose[:3, :3] = rot
				pose[:3, 3] = [tx, ty, tz]
				poses[frame_path] = pose
		return poses

	def quat_to_rotmat(self, qw, qx, qy, qz):
		return Rotation.from_quat(np.array([qx, qy, qz, qw])).as_matrix()

	# NOTE(gogojjh): Arbitrary multiplier for epoch size
	def __len__(self):
		return len(self.scenes) * self.num_data_each_scene

	def get_stats(self):
		return f'{len(self.scenes)} scenes with {sum(len(s["frames"]) for s in self.scenes)} total frames'

	def _get_views(self, scene_idx, resolution, rng):
		# Randomly select two frames from the same scene
		scene = self.scenes[scene_idx % len(self.scenes)]
		frame_paths = random.sample(scene['frames'], 2)
		
		views = []
		for path in frame_paths:
			# Load color image and depth map
			image = imread_cv2(path)
			depth_path = path.replace('.jpg', '.zed.png')
			depth_map = imread_cv2(depth_path, cv2.IMREAD_ANYDEPTH)
			depth_map = (depth_map / 1000).astype(np.float32)  # Convert mm to meters

			# Get camera parameters
			intrinsics = scene['intrinsics'][path]
			
			# world-to-camera pose: transform world point into camera coordinate
			pose_w2c = scene['poses'][path]
			
			# Convert to camera-to-world pose: transform camera point into world coordinate
			pose_c2w = np.linalg.inv(pose_w2c)
			
			# Create intrinsics matrix
			fx, fy, cx, cy = intrinsics
			intrinsics_mat = np.array([
				[fx, 0, cx],
				[0, fy, cy],
				[0,  0,  1]
			], dtype=np.float32)
			
			# Apply any necessary preprocessing
			image, depth_map, intrinsics_mat = self._crop_resize_if_necessary(
				image, depth_map, intrinsics_mat, resolution, rng, info=path)
			
			views.append({
				'img': image,
				'depthmap': depth_map,
				'camera_pose': pose_c2w,  # Converted to cam2world
				'camera_intrinsics': intrinsics_mat,
				'dataset': 'MapFree',
				'label': osp.basename(scene['path']),
				'instance': osp.basename(path)
			})
			
		return views

if __name__ == '__main__':
	import numpy as np
	from dust3r.datasets.base.base_stereo_view_dataset import view_name
	from dust3r.viz import SceneViz, auto_cam_size
	from dust3r.utils.image import rgb

	# Initialize MapFree dataset
	dataset = MapFree(
		ROOT="data/mapfree_processed",  # Path to MapFree data
		split='test',                   # Use test split
		resolution=224,                 # Target resolution
		aug_crop=16                     # Augmentation crop size (if needed)
	)

	# Visualize random samples
	for idx in np.random.permutation(len(dataset)):
		views = dataset[idx]
		assert len(views) == 2, "Each sample should contain 2 views"
		print(f"Sample {idx}: {view_name(views[0])} vs {view_name(views[1])}")
		viz = SceneViz()
		poses = [views[i]['camera_pose'] for i in [0, 1]]
		cam_size = max(auto_cam_size(poses), 0.001)
		cam_size = 2
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