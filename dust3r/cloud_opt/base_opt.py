# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Base class for the global alignement procedure
# --------------------------------------------------------
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import roma
from copy import deepcopy
import tqdm

from dust3r.utils.geometry import inv, geotrf
from dust3r.utils.device import to_numpy
from dust3r.utils.image import rgb
from dust3r.viz import SceneViz, segment_sky, auto_cam_size
from dust3r.optim_factory import adjust_learning_rate_by_lr

from dust3r.cloud_opt.commons import (edge_str, ALL_DISTS, NoGradParamDict, get_imshapes, signed_expm1, signed_log1p,
									  cosine_schedule, linear_schedule, get_conf_trf)
import dust3r.cloud_opt.init_im_poses as init_fun


class BasePCOptimizer (nn.Module):
	""" Optimize a global scene, given a list of pairwise observations.
	Graph node: images
	Graph edges: observations = (pred1, pred2)
	"""

	def __init__(self, *args, **kwargs):
		if len(args) == 1 and len(kwargs) == 0:
			other = deepcopy(args[0])
			attrs = '''edges is_symmetrized dist n_imgs pred_i pred_j imshapes 
						min_conf_thr conf_thr conf_i conf_j im_conf
						base_scale norm_pw_scale POSE_DIM pw_poses 
						pw_adaptors pw_adaptors has_im_poses rand_pose imgs verbose'''.split()
			self.__dict__.update({k: other[k] for k in attrs})
		else:
			self._init_from_views(*args, **kwargs)

	def _init_from_views(self, view1, view2, pred1, pred2,
						 dist='l1',
						 conf='log',
						 min_conf_thr=3,
						 base_scale=0.5,
						 allow_pw_adaptors=False,
						 pw_break=20,
						 rand_pose=torch.randn,
						 iterationsCount=None,
						 verbose=True):
		super().__init__()
		if not isinstance(view1['idx'], list):
			view1['idx'] = view1['idx'].tolist()
		if not isinstance(view2['idx'], list):
			view2['idx'] = view2['idx'].tolist()
		self.edges = [(int(i), int(j)) for i, j in zip(view1['idx'], view2['idx'])]
		# NOTE(gogojjh):
		# is_symmetrized = True: edges = [(1, 0), (2, 0), (2, 1), (0, 1), (0, 2), (1, 2), ...]
		# is_symmetrized = False: edges = [(1, 0), (2, 0), (2, 1)]
		self.is_symmetrized = set(self.edges) == {(j, i) for i, j in self.edges}
		self.dist = ALL_DISTS[dist]
		self.verbose = verbose

		self.n_imgs = self._check_edges()

		# input data
		pred1_pts = pred1['pts3d']
		pred2_pts = pred2['pts3d_in_other_view']
		self.pred_i = NoGradParamDict({ij: pred1_pts[n] for n, ij in enumerate(self.str_edges)})
		self.pred_j = NoGradParamDict({ij: pred2_pts[n] for n, ij in enumerate(self.str_edges)})
		self.imshapes = get_imshapes(self.edges, pred1_pts, pred2_pts)

		# work in log-scale with conf
		pred1_conf = pred1['conf']
		pred2_conf = pred2['conf']
		self.min_conf_thr = min_conf_thr
		self.conf_trf = get_conf_trf(conf)

		self.conf_i = NoGradParamDict({ij: pred1_conf[n] for n, ij in enumerate(self.str_edges)})
		self.conf_j = NoGradParamDict({ij: pred2_conf[n] for n, ij in enumerate(self.str_edges)})
		self.im_conf = self._compute_img_conf(pred1_conf, pred2_conf)
		for i in range(len(self.im_conf)):
			self.im_conf[i].requires_grad = False      

		# pairwise pose parameters
		self.base_scale = base_scale
		self.norm_pw_scale = True
		self.pw_break = pw_break
		self.POSE_DIM = 7
		self.pw_poses = nn.Parameter(rand_pose((self.n_edges, 1+self.POSE_DIM)))  # pairwise poses
		self.pw_adaptors = nn.Parameter(torch.zeros((self.n_edges, 2)))  # slight xy/z adaptation
		self.pw_adaptors.requires_grad_(allow_pw_adaptors)
		self.has_im_poses = False
		self.rand_pose = rand_pose

		# possibly store images for show_pointcloud
		self.imgs = None
		if 'img' in view1 and 'img' in view2:
			imgs = [torch.zeros((3,)+hw) for hw in self.imshapes]
			for v in range(len(self.edges)):
				idx = view1['idx'][v]
				imgs[idx] = view1['img'][v]
				idx = view2['idx'][v]
				imgs[idx] = view2['img'][v]
			self.imgs = rgb(imgs)

		##########################33 NOTE(gogojjh):
		# add learnable weights
		self.weight_i = nn.ParameterDict({ij: nn.Parameter(torch.ones_like(pred1_conf[n]), requires_grad=False) 
										  for n, ij in enumerate(self.str_edges)})
		self.weight_j = nn.ParameterDict({ij: nn.Parameter(torch.ones_like(pred2_conf[n]), requires_grad=False) 
										  for n, ij in enumerate(self.str_edges)})        

	@property
	def n_edges(self):
		return len(self.edges)

	@property
	def str_edges(self):
		return [edge_str(i, j) for i, j in self.edges]

	@property
	def imsizes(self):
		return [(w, h) for h, w in self.imshapes]

	@property
	def device(self):
		return next(iter(self.parameters())).device

	def state_dict(self, trainable=True):
		all_params = super().state_dict()
		return {k: v for k, v in all_params.items() if k.startswith(('_', 'pred_i.', 'pred_j.', 'conf_i.', 'conf_j.')) != trainable}

	def load_state_dict(self, data):
		return super().load_state_dict(self.state_dict(trainable=False) | data)

	def _check_edges(self):
		indices = sorted({i for edge in self.edges for i in edge})
		assert indices == list(range(len(indices))), 'bad pair indices: missing values '
		return len(indices)

	@torch.no_grad()
	def _compute_img_conf(self, pred1_conf, pred2_conf):
		im_conf = nn.ParameterList([torch.zeros(hw, device=self.device) for hw in self.imshapes])
		for e, (i, j) in enumerate(self.edges):
			im_conf[i] = torch.maximum(im_conf[i], pred1_conf[e])
			im_conf[j] = torch.maximum(im_conf[j], pred2_conf[e])
		return im_conf

	def get_adaptors(self):
		adapt = self.pw_adaptors
		adapt = torch.cat((adapt[:, 0:1], adapt), dim=-1)  # (scale_xy, scale_xy, scale_z)
		if self.norm_pw_scale:  # normalize so that the product == 1
			adapt = adapt - adapt.mean(dim=1, keepdim=True)
		return (adapt / self.pw_break).exp()

	def _get_poses(self, poses):
		# normalize rotation
		Q = poses[:, :4]
		T = signed_expm1(poses[:, 4:7])
		RT = roma.RigidUnitQuat(Q, T).normalize().to_homogeneous()
		return RT

	def _set_pose(self, poses, idx, R, T=None, scale=None, force=False):
		# all poses == cam-to-world
		pose = poses[idx]
		if not (pose.requires_grad or force):
			return pose

		if R.shape == (4, 4):
			assert T is None
			T = R[:3, 3]
			R = R[:3, :3]

		if R is not None:
			pose.data[0:4] = roma.rotmat_to_unitquat(R)
		if T is not None:
			pose.data[4:7] = signed_log1p(T / (scale or 1))  # translation is function of scale

		if scale is not None:
			assert poses.shape[-1] in (8, 13)
			pose.data[-1] = np.log(float(scale))
		return pose

	def get_pw_norm_scale_factor(self):
		if self.norm_pw_scale:
			# normalize scales so that things cannot go south
			# we want that exp(scale) ~= self.base_scale
			return (np.log(self.base_scale) - self.pw_poses[:, -1].mean()).exp()
		else:
			return 1  # don't norm scale for known poses

	def get_pw_scale(self):
		scale = self.pw_poses[:, -1].exp()  # (n_edges,)
		scale = scale * self.get_pw_norm_scale_factor()
		return scale

	def get_pw_poses(self):  # cam to world
		RT = self._get_poses(self.pw_poses)
		scaled_RT = RT.clone()
		scaled_RT[:, :3] *= self.get_pw_scale().view(-1, 1, 1)  # scale the rotation AND translation
		return scaled_RT

	def get_masks(self):
		return [(conf > self.min_conf_thr) for conf in self.im_conf]

	def depth_to_pts3d(self):
		raise NotImplementedError()

	def get_pts3d(self, raw=False):
		res = self.depth_to_pts3d()
		if not raw:
			res = [dm[:h*w].view(h, w, 3) for dm, (h, w) in zip(res, self.imshapes)]
		return res

	def _set_focal(self, idx, focal, force=False):
		raise NotImplementedError()

	def get_focals(self):
		raise NotImplementedError()

	def get_known_focal_mask(self):
		raise NotImplementedError()

	def get_principal_points(self):
		raise NotImplementedError()

	def get_conf(self, mode=None):
		trf = self.conf_trf if mode is None else get_conf_trf(mode)
		return [trf(c) for c in self.im_conf]

	def get_im_poses(self):
		raise NotImplementedError()

	def _set_depthmap(self, idx, depth, force=False):
		raise NotImplementedError()

	def get_depthmaps(self, raw=False):
		raise NotImplementedError()

	def clean_pointcloud(self, **kw):
		cams = inv(self.get_im_poses())
		K = self.get_intrinsics()
		depthmaps = self.get_depthmaps()
		all_pts3d = self.get_pts3d()

		new_im_confs = clean_pointcloud(self.im_conf, K, cams, depthmaps, all_pts3d, **kw)

		for i, new_conf in enumerate(new_im_confs):
			self.im_conf[i].data[:] = new_conf
		return self

	def forward(self, ret_details=False):
		# NOTE(gogojjh):
		"""
		Performs the forward pass of the optimization process.

		Args:
			ret_details (bool): Flag indicating whether to return the details of the optimization.

		Returns:
			float or tuple: The loss value if `ret_details` is False, otherwise a tuple containing the loss value and details.
		"""
		# Constant parameters
		mu = 0.01
		conf_thre = 0.5

		pw_poses = self.get_pw_poses()  # cam-to-world
		pw_adapt = self.get_adaptors()
		proj_pts3d = self.get_pts3d()                                           # optimized point in the global coordinate

		loss = 0
		if ret_details:
			details = -torch.ones((self.n_imgs, self.n_imgs))

		for e, (i, j) in enumerate(self.edges):
			i_j = edge_str(i, j)
			# distance in image i and j
			aligned_pred_i = geotrf(pw_poses[e], pw_adapt[e] * self.pred_i[i_j]) # predicted point in the global coordinate
			aligned_pred_j = geotrf(pw_poses[e], pw_adapt[e] * self.pred_j[i_j])
			res_i = proj_pts3d[i] - aligned_pred_i
			res_j = proj_pts3d[j] - aligned_pred_j
			
			C_i = self.conf_trf(self.conf_i[i_j])
			C_j = self.conf_trf(self.conf_j[i_j])

			# Weight
			self.weight_i[i_j] = C_i / (1 + res_i.norm(dim=-1) / mu) ** 2
			self.weight_j[i_j] = C_j / (1 + res_j.norm(dim=-1) / mu) ** 2

			mask_i = C_i > conf_thre
			mask_j = C_j > conf_thre
		
			# Regularization term (μ*(√w_p - √C_p)^2)
			reg_i = mu * (torch.sqrt(self.weight_i[i_j][mask_i]) - torch.sqrt(C_i[mask_i]))**2
			reg_j = mu * (torch.sqrt(self.weight_j[i_j][mask_j]) - torch.sqrt(C_j[mask_j]))**2

			# Weighted error term (w_p * ||e_p||)
			loss_i = (self.weight_i[i_j][mask_i] * res_i.norm(dim=-1)[mask_i]).mean() + reg_i.mean()
			loss_j = (self.weight_j[i_j][mask_j] * res_j.norm(dim=-1)[mask_j]).mean() + reg_j.mean()

			loss += loss_i + loss_j

			if ret_details:
				details[i, j] = loss_i + loss_j
		loss /= self.n_edges  # average over all pairs

		if ret_details:
			return loss, details
		return loss

	@torch.no_grad()
	def visualize_weights_errors(self, edge_str_key, cur_iter, vmin=0, vmax=1):
		"""
		Visualizes raw images, confidence maps, weights, and error maps for the ith and jth images in a 2x4 grid.

		Args:
			edge_str_key (str): Edge identifier (e.g., "0-1").
			vmin/vmax (float): Colorbar range for plotting.
		"""
		import matplotlib.pyplot as plt
		import torch

		mu = 0.01
		conf_thre = 1.5

		# Fetch data
		i, j = map(int, edge_str_key.split('_'))
		pw_poses = self.get_pw_poses()  # cam-to-world
		pw_adapt = self.get_adaptors()
		proj_pts3d = self.get_pts3d()  # optimized point in the global coordinate

		# Compute error maps
		aligned_pred_j = geotrf(pw_poses[self.str_edges.index(edge_str_key)], pw_adapt[self.str_edges.index(edge_str_key)] * self.pred_j[edge_str_key])
		res_j = proj_pts3d[j] - aligned_pred_j
		error_map_j = res_j.norm(dim=-1).cpu().numpy()

		aligned_pred_i = geotrf(pw_poses[self.str_edges.index(edge_str_key)], pw_adapt[self.str_edges.index(edge_str_key)] * self.pred_i[edge_str_key])
		res_i = proj_pts3d[i] - aligned_pred_i
		error_map_i = res_i.norm(dim=-1).cpu().numpy()

		# Fetch raw images, confidence maps, and weights
		raw_image_i = to_numpy(self.imgs[i])
		if np.issubdtype(raw_image_i.dtype, np.floating):
			raw_image_i = np.uint8(255*raw_image_i.clip(min=0, max=1))
		raw_image_j = to_numpy(self.imgs[j])
		if np.issubdtype(raw_image_j.dtype, np.floating):
			raw_image_j = np.uint8(255*raw_image_j.clip(min=0, max=1))
		C_i = self.conf_trf(self.conf_i[edge_str_key]).cpu().numpy()
		C_j = self.conf_trf(self.conf_j[edge_str_key]).cpu().numpy()
		weight_i = C_i / (1 + error_map_i / mu)**2
		weight_j = C_j / (1 + error_map_j / mu)**2

		mask_i = weight_i < conf_thre
		mask_j = weight_j < conf_thre

		# Extract values for the ith and jth maps
		random_coords_i = np.array([[30, 470], [80, 400]])
		C_i_values = np.array([C_i[y, x] for y, x in random_coords_i])
		weight_i_values = np.array([weight_i[y, x] for y, x in random_coords_i])
		error_i_values = np.array([error_map_i[y, x] for y, x in random_coords_i])

		random_coords_j = np.array([[150, 300], [100, 400]])
		C_j_values = np.array([C_j[y, x] for y, x in random_coords_j])
		weight_j_values = np.array([weight_j[y, x] for y, x in random_coords_j])
		error_j_values = np.array([error_map_j[y, x] for y, x in random_coords_j])

		# max_error = max(np.max(error_j_values), np.max(error_i_values))
		max_error = 0.125
		max_conf = 2.8

		opt_depthmaps = self.get_depthmaps()

		# Create figure
		fig, axes = plt.subplots(2, 7, figsize=(24, 8))

		# Plot ith row
		axes[0, 0].imshow(raw_image_i, cmap='gray')
		axes[0, 0].set_title(f'Raw Image {i}')
		for idx in range(len(random_coords_i)):
			axes[0, 0].plot([random_coords_i[idx][1]], [random_coords_i[idx][0]], color='r', marker='x', markersize=10)

		raw_image_i[mask_i] = [0, 0, 0]
		axes[0, 1].imshow(raw_image_i, cmap='gray')
		axes[0, 1].set_title(f'Filtered Image {i}')

		im = axes[0, 2].imshow(C_i, cmap='jet')
		axes[0, 2].set_title(f'Confidence Map {i}')
		fig.colorbar(im, ax=axes[0, 2])
		
		im = axes[0, 3].imshow(weight_i, cmap='jet')
		axes[0, 3].set_title(f'Weight {i}')
		fig.colorbar(im, ax=axes[0, 3])
		
		im = axes[0, 4].imshow(error_map_i, cmap='jet')
		axes[0, 4].set_title(f'Error Map {i}')
		fig.colorbar(im, ax=axes[0, 4])
		
		im = axes[0, 5].imshow(opt_depthmaps[i].detach().cpu().numpy(), cmap='jet')
		axes[0, 5].set_title(f'Depth Map {i}')
		fig.colorbar(im, ax=axes[0, 5], shrink=0.8)
		
		for idx in range(len(random_coords_i)):
			axes[0, 6].plot([C_i_values[idx]], [error_i_values[idx]], color='r', marker='x')
			axes[0, 6].plot([weight_i_values[idx]], [error_i_values[idx]], color='k', marker='x')
		axes[0, 6].set_xlim(-0.3, max_conf)
		axes[0, 6].set_ylim(-0.01, max_error)

		# Plot jth row
		axes[1, 0].imshow(raw_image_j, cmap='gray')
		axes[1, 0].set_title(f'Raw Image {j}')
		for idx in range(len(random_coords_j)):
			axes[1, 0].plot([random_coords_j[idx][1]], [random_coords_j[idx][0]], color='r', marker='x', markersize=10)

		raw_image_j[mask_j] = [0, 0, 0]
		axes[1, 1].imshow(raw_image_j, cmap='gray')
		axes[1, 1].set_title(f'Filtered Image {j}')

		im = axes[1, 2].imshow(C_j, cmap='jet')
		axes[1, 2].set_title(f'Confidence Map {j}')
		fig.colorbar(im, ax=axes[1, 2])
		
		im = axes[1, 3].imshow(weight_j, cmap='jet')
		axes[1, 3].set_title(f'Weight {j}')
		fig.colorbar(im, ax=axes[1, 3])
		
		im = axes[1, 4].imshow(error_map_j, cmap='jet')
		axes[1, 4].set_title(f'Error Map {j}')
		fig.colorbar(im, ax=axes[1, 4])
		
		im = axes[1, 5].imshow(opt_depthmaps[j].detach().cpu().numpy(), cmap='jet')
		axes[1, 5].set_title(f'Depth Map {j}')
		fig.colorbar(im, ax=axes[1, 5], shrink=0.8)
		
		for idx in range(len(random_coords_j)):
			axes[1, 6].plot([C_j_values[idx]], [error_j_values[idx]], color='r', marker='x')
			axes[1, 6].plot([weight_j_values[idx]], [error_j_values[idx]], color='k', marker='x')
		axes[1, 6].set_xlim(-0.3, max_conf)
		axes[1, 6].set_ylim(-0.01, max_error)			

		plt.tight_layout()
		plt.savefig(f'/Titan/code/robohike_ws/src/pose_estimation_models/outputs_duster/replica_img_weights_errors_{cur_iter}.jpg')
		plt.close()


	@torch.cuda.amp.autocast(enabled=False)
	def compute_global_alignment(self, init=None, niter_PnP=10, **kw):
		if init is None:
			pass
		elif init == 'msp' or init == 'mst':
			init_fun.init_minimum_spanning_tree(self, niter_PnP=niter_PnP)
		elif init == 'known_poses':
			init_fun.init_from_known_poses(self, min_conf_thr=self.min_conf_thr,
										   niter_PnP=niter_PnP)
		else:
			raise ValueError(f'bad value for {init=}')

		return global_alignment_loop(self, **kw)

	@torch.no_grad()
	def mask_sky(self):
		res = deepcopy(self)
		for i in range(self.n_imgs):
			sky = segment_sky(self.imgs[i])
			res.im_conf[i][sky] = 0
		return res

	def show(self, show_pw_cams=False, show_pw_pts3d=False, cam_size=None, **kw):
		viz = SceneViz()
		if self.imgs is None:
			colors = np.random.randint(0, 256, size=(self.n_imgs, 3))
			colors = list(map(tuple, colors.tolist()))
			for n in range(self.n_imgs):
				viz.add_pointcloud(self.get_pts3d()[n], colors[n], self.get_masks()[n])
		else:
			viz.add_pointcloud(self.get_pts3d(), self.imgs, self.get_masks())
			colors = np.random.randint(256, size=(self.n_imgs, 3))

		# camera poses
		im_poses = to_numpy(self.get_im_poses())
		if cam_size is None:
			cam_size = auto_cam_size(im_poses)
		viz.add_cameras(im_poses, self.get_focals(), colors=colors,
						images=self.imgs, imsizes=self.imsizes, cam_size=cam_size)
		if show_pw_cams:
			pw_poses = self.get_pw_poses()
			viz.add_cameras(pw_poses, color=(192, 0, 192), cam_size=cam_size)

			if show_pw_pts3d:
				pts = [geotrf(pw_poses[e], self.pred_i[edge_str(i, j)]) for e, (i, j) in enumerate(self.edges)]
				viz.add_pointcloud(pts, (128, 0, 128))

		viz.show(**kw)
		return viz

def global_alignment_loop(net, lr=0.01, niter=300, schedule='cosine', lr_min=1e-6):
	# NOTE(gogojjh):
	"""
	Performs global alignment optimization loop.

	Args:
		net (nn.Module): The neural network model.
		lr (float): The learning rate for optimization (default: 0.01).
		niter (int): The number of iterations for optimization (default: 300).
		schedule (str): The learning rate schedule (default: 'cosine').
		lr_min (float): The minimum learning rate (default: 1e-6).

	Returns:
		float: The final loss value after optimization.
	"""
	params = [p for p in net.parameters() if p.requires_grad]
	if not params:
		return net

	verbose = net.verbose
	if verbose:
		print('Global alignment - optimizing for:')
		print([name for name, value in net.named_parameters() if value.requires_grad])

	lr_base = lr
	optimizer = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.9))

	loss = float('inf')
	if verbose:
		with tqdm.tqdm(total=niter) as bar:
			while bar.n < bar.total:
				loss, lr = global_alignment_iter(net, bar.n, niter, lr_base, lr_min, optimizer, schedule)
				bar.set_postfix_str(f'{lr=:g} loss={loss:g}')
				bar.update()
	else:
		for n in range(niter):
			loss, _ = global_alignment_iter(net, n, niter, lr_base, lr_min, optimizer, schedule)
	return loss


def global_alignment_iter(net, cur_iter, niter, lr_base, lr_min, optimizer, schedule):
	# NOTE(gogojjh):
	"""
	Perform a single iteration of global alignment optimization.

	Args:
		net: The neural network model.
		cur_iter: The current iteration number.
		niter: The total number of iterations.
		lr_base: The base learning rate.
		lr_min: The minimum learning rate.
		optimizer: The optimizer used for gradient descent.
		schedule: The learning rate schedule ('cosine' or 'linear').

	Returns:
		loss: The loss value for the current iteration.
		lr: The learning rate used for the current iteration.
	"""
	t = cur_iter / niter
	if schedule == 'cosine':
		lr = cosine_schedule(t, lr_base, lr_min)
	elif schedule == 'linear':
		lr = linear_schedule(t, lr_base, lr_min)
	else:
		raise ValueError(f'bad lr {schedule=}')
	adjust_learning_rate_by_lr(optimizer, lr)
	optimizer.zero_grad()

	if cur_iter % 50 == 0:  # Visualize every 50 iterations
		net.visualize_weights_errors(edge_str(0, 1), cur_iter)

	loss = net()
	loss.backward()
	optimizer.step()

	return float(loss), lr


@torch.no_grad()
def clean_pointcloud( im_confs, K, cams, depthmaps, all_pts3d, 
					  tol=0.001, bad_conf=0, dbg=()):
	""" Method: 
	1) express all 3d points in each camera coordinate frame
	2) if they're in front of a depthmap --> then lower their confidence
	"""
	assert len(im_confs) == len(cams) == len(K) == len(depthmaps) == len(all_pts3d)
	assert 0 <= tol < 1
	res = [c.clone() for c in im_confs]

	# reshape appropriately
	all_pts3d = [p.view(*c.shape,3) for p,c in zip(all_pts3d, im_confs)]
	depthmaps = [d.view(*c.shape) for d,c in zip(depthmaps, im_confs)]
	
	for i, pts3d in enumerate(all_pts3d):
		for j in range(len(all_pts3d)):
			if i == j: continue

			# project 3dpts in other view
			proj = geotrf(cams[j], pts3d)
			proj_depth = proj[:,:,2]
			u,v = geotrf(K[j], proj, norm=1, ncol=2).round().long().unbind(-1)

			# check which points are actually in the visible cone
			H, W = im_confs[j].shape
			msk_i = (proj_depth > 0) & (0 <= u) & (u < W) & (0 <= v) & (v < H)
			msk_j = v[msk_i], u[msk_i]

			# find bad points = those in front but less confident
			bad_points = (proj_depth[msk_i] < (1-tol) * depthmaps[j][msk_j]) & (res[i][msk_i] < res[j][msk_j])

			bad_msk_i = msk_i.clone()
			bad_msk_i[msk_i] = bad_points
			res[i][bad_msk_i] = res[i][bad_msk_i].clip_(max=bad_conf)

	return res
