# Copyright (c) 2019-present, Royal Bank of Canada.
# Copyright (c) 2019-present, Michael Kelly.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

####################################################################################
# Code is based on the IODINE (https://arxiv.org/pdf/1903.00450.pdf) implementation 
# from https://github.com/MichaelKevinKelly/IODINE by Michael Kelly
####################################################################################

import torch
import math
import os
import numpy as np
from .utils.util import adjusted_rand_index, adjusted_rand_index_without_bg


class Model(torch.nn.Module):

	def __init__(self,
		opt,
		refine_net,
		decoder,
		T,
		K,
		z_dim,
		name='spatio_temporal',
		beta=100.,
		gamma = 0.1,
		psi = 10.,
		feature_extractor=None):
		super(Model, self).__init__()

		self.opt = opt
		save_path = opt.save_path
		model_name = opt.model_name
		self.save_dir = save_path+ model_name + '/'
		self.eps = 1e-9

		self.beta = beta
		self.gamma = gamma
		self.psi = psi
		self.T_non_fixed = T
		self.K_non_fixed = K

		self.lmbda0 = torch.nn.Parameter(torch.rand(1,2*z_dim)-0.5,requires_grad=True)
		self.decoder = decoder
		self.refine_net = refine_net
		self.layer_norms = torch.nn.ModuleList([
			torch.nn.LayerNorm((1,64,64),elementwise_affine=False),
			torch.nn.LayerNorm((3,64,64),elementwise_affine=False),
			torch.nn.LayerNorm((1,64,64),elementwise_affine=False),
			torch.nn.LayerNorm((2*z_dim,),elementwise_affine=False),
			torch.nn.LayerNorm((1,64,64),elementwise_affine=False)])


		self.feature_extractor = torch.nn.Sequential(
			feature_extractor,
			torch.nn.Conv2d(128,64,3,stride=1,padding=1),
			torch.nn.ELU(),
			torch.nn.Conv2d(64,32,3,stride=1,padding=1),
			torch.nn.ELU(),
			torch.nn.Conv2d(32,16,3,stride=1,padding=1),
			torch.nn.ELU())
		for param in self.feature_extractor[0]:
			param.requires_grad = False

		#2 layers MLP for learning the Prior parameters based on the temporal hidden state
		self.prior = torch.nn.Sequential(
			torch.nn.Linear(128, 128),
			torch.nn.ReLU())
		self.prior_mean = torch.nn.Linear(128, z_dim)
		self.prior_var = torch.nn.Sequential(
			torch.nn.Linear(128, z_dim),
			torch.nn.Softplus())

		self.name = name
		self.register_buffer('T', torch.tensor(T))
		self.register_buffer('K', torch.tensor(K))
		self.register_buffer('z_dim', torch.tensor(z_dim))
		self.register_buffer('var_x', torch.tensor(0.3))
		self.register_buffer('h0',torch.zeros((1,128)))
		self.register_buffer('base_loss',torch.zeros(1,1))
		self._create_meshgrid()
		self._setup_debug()

	"""
	Forward pass through the model.
	Two loops traverse the grid of time and refinements. The outer loop iterates
	along the frames and the inner along the iterative refinements.
	Additionally after the inference is done, if number of predict_frames 
	is not zero a new frames will be simulated.
	"""
	def forward(self, img, gt):
		img = img.permute(1,0,2,3,4)

		F,N,C,H,W = img.shape
		F -= self.opt.predict_frames

		K, T, z_dim = self.K_non_fixed, self.T_non_fixed, self.z_dim
		assert not torch.isnan(self.lmbda0).any().item(), 'lmbda0 has nan'

		## Initialize parameters for latents' distribution
		lmbda_frames = self.lmbda0.expand((F+self.opt.predict_frames, N*K,)+self.lmbda0.shape[1:])
		lmbda_first = self.lmbda0.expand((N*K,)+self.lmbda0.shape[1:])
		total_loss, losses = torch.zeros_like(self.base_loss.expand((N,1))), []

		## Initialize LSTMCell hidden states
		h_hor = self.h0.expand((N*K,)+self.h0.shape[1:]).clone().detach() ##TODO
		c_hor = torch.zeros_like(h_hor[0])
		_x_hor = torch.zeros_like(img[0]).unsqueeze(dim=1).expand((N,K,C,H,W))
		assert h_hor.max().item()==0. and h_hor.min().item()==0.



		final_masks = []
		final_mu_x = []
		mu_z, logvar_z = lmbda_first.chunk(2,dim=1)
		mu_z, logvar_z = mu_z.contiguous(), logvar_z.contiguous()
		z_first = self._sample(mu_z,logvar_z) ## (N*K,z_dim)
		r = 0
		for f in range(F):
			x = img[f]
			lmbda = lmbda_frames[f]
			h_vert = self.h0.expand((N*K,)+self.h0.shape[1:]).clone().detach() ##TODO
			c_vert = torch.zeros_like(h_vert)
			counter = 0
			prior_t = self.prior(h_hor)
			prior_mean_t = self.prior_mean(prior_t)
			prior_var_t = self.prior_var(prior_t)
			for it in range(T - r):
				## Sample latent code
				mu_z, logvar_z = lmbda.chunk(2,dim=1)
				mu_z, logvar_z = mu_z.contiguous(), logvar_z.contiguous()
				if it == 0:
					z = z_first
				else:
					z = self._sample(mu_z,logvar_z) ## (N*K,z_dim)

				## Get means and masks
				dec_out = self.decoder(z, h_hor) ## (N*K,C+1,H,W)
				mu_x, mask_logits = dec_out[:,:C,:,:], dec_out[:,C,:,:] ## (N*K,C,H,W), (N*K,H,W)
				mask_logits = mask_logits.view((N,K,)+mask_logits.shape[1:]) ## (N,K,H,W)
				mu_x = mu_x.view((N,K,)+mu_x.shape[1:]) ##(N,K,C,H,W)

				## Process masks
				masks = torch.nn.functional.softmax(mask_logits,dim=1).unsqueeze(dim=2) ##(N,K,1,H,W)
				mask_logits = mask_logits.unsqueeze(dim=2) ##(N,K,1,H,W)

				## Calculate loss: reconstruction (nll) & KL divergence
				_x = x.unsqueeze(dim=1).expand((N,K,)+x.shape[1:]) ## (N,K,C,H,W)
				deviation = -1.*(mu_x - _x)**2
				ll_pxl_channels = ((masks*(deviation/(2.*self.var_x)).exp()).sum(dim=1,keepdim=True)).log()
				assert ll_pxl_channels.min().item()>-math.inf
				ll_pxl = ll_pxl_channels.sum(dim=2,keepdim=True) ## (N,1,1,H,W)
				ll_pxl_flat = ll_pxl.view(N,-1)
				nll = -1.*(ll_pxl_flat.sum(dim=-1).mean())


				#Calculate the entropy loss
				e = -(masks*torch.log(masks+self.eps)).sum(dim = 1)
				flat_e = e.view(N,-1).sum(dim=-1).mean()

				if self.opt.cond_prior:
					div = self._get_div_gauss(mu_z, logvar_z, prior_mean_t, prior_var_t, N, K)
				else:
					div = self._get_div(mu_z,logvar_z,N,K)


				if self.opt.use_entropy:
					loss = self.beta * nll + self.psi * div + self.gamma * flat_e
				else:
					loss = self.beta * nll + self.psi * div

				## Accumulate loss 
				scaled_loss = ((float(it)+1)/float(T)) * loss
				losses.append(scaled_loss)

				total_loss += scaled_loss
				
				assert not torch.isnan(loss).any().item(), 'Loss at t={} is nan. (nll,div): ({},{},{})'.format(nll,div, flat_e)
				if it==T-1: continue

				## Refine lambda
				if self.opt.additional_input:
					refine_inp = self.get_refine_inputs(_x,mu_x,masks,mask_logits,ll_pxl,lmbda,loss,deviation, _x_hor)
				else: 
					refine_inp = self.get_refine_inputs(_x,mu_x,masks,mask_logits,ll_pxl,lmbda,loss,deviation)

				## Potentially add additional features from pretrained model (scaled down to appropriate size)
				x_resized = torch.nn.functional.interpolate(x,257) ## Upscale to desired input size for squeezenet
				additional_features = self.feature_extractor(x_resized).unsqueeze(dim=1)
				additional_features = additional_features.expand((N,K,16,64,64)).contiguous()
				additional_features = additional_features.view((N*K,16,64,64))
				refine_inp['img'] = torch.cat((refine_inp['img'],additional_features),dim=1)

				delta, h_vert, c_vert = self.refine_net(refine_inp, h_hor, c_hor, h_vert, c_vert)


				assert not torch.isnan(lmbda).any().item(), 'Lmbda at t={} has nan: {}'.format(it,lmbda)
				assert not torch.isnan(delta).any().item(), 'Delta at t={} has nan: {}'.format(it,delta)
				lmbda = lmbda + delta
				assert not torch.isnan(lmbda).any().item(), 'Lmbda at t={} has nan: {}'.format(it,lmbda)
				counter += 1

			if not r == T - 2:
				r += 1

			h_hor = h_vert
			c_hor = c_vert
			_x_hor = _x
			final_masks.append(masks)
			final_mu_x.append(mu_x)

		with torch.no_grad():
			for f_predict in range(F, F + self.opt.predict_frames):

				lmbda = lmbda_frames[f_predict]
				h_vert = self.h0.expand((N*K,)+self.h0.shape[1:]).clone().detach() ##TODO
				c_vert = torch.zeros_like(h_vert) 
				prior_t = self.prior(h_hor)
				prior_mean_t = self.prior_mean(prior_t)
				prior_var_t = self.prior_var(prior_t)
				mu_z, logvar_z = lmbda.chunk(2,dim=1)
				mu_z, logvar_z = mu_z.contiguous(), logvar_z.contiguous()
				z = z_first

				## Get means and masks
				dec_out = self.decoder(z, h_hor) ## (N*K,C+1,H,W)
				mu_x, mask_logits = dec_out[:,:C,:,:], dec_out[:,C,:,:] ## (N*K,C,H,W), (N*K,H,W)
				mask_logits = mask_logits.view((N,K,)+mask_logits.shape[1:]) ## (N,K,H,W)
				mu_x = mu_x.view((N,K,)+mu_x.shape[1:]) ##(N,K,C,H,W)

				## Process masks
				masks = torch.nn.functional.softmax(mask_logits,dim=1).unsqueeze(dim=2) ##(N,K,1,H,W)
				mask_logits = mask_logits.unsqueeze(dim=2) ##(N,K,1,H,W)


				x = (mu_x * masks).sum(dim=1)
				#print("x dimention:    ", x.shape)

				## Calculate loss: reconstruction (nll) & KL divergence
				_x = x.unsqueeze(dim=1).expand((N,K,)+x.shape[1:]) ## (N,K,C,H,W)
				deviation = -1.*(mu_x - _x)**2
				ll_pxl_channels = ((masks*(deviation/(2.*self.var_x)).exp()).sum(dim=1,keepdim=True)).log()
				assert ll_pxl_channels.min().item()>-math.inf
				ll_pxl = ll_pxl_channels.sum(dim=2,keepdim=True) ## (N,1,1,H,W)
				ll_pxl_flat = ll_pxl.view(N,-1)

				nll = -1.*(ll_pxl_flat.sum(dim=-1).mean())

				if self.opt.cond_prior:
					div = self._get_div_gauss(mu_z, logvar_z, prior_mean_t, prior_var_t, N, K)
				else:
					div = self._get_div(mu_z,logvar_z,N,K)
				
				loss = self.beta * nll + self.psi * div


				refine_inp = self.get_refine_inputs(_x,mu_x,masks,mask_logits,ll_pxl,lmbda,loss,deviation, predict= True)

				## Potentially add additional features from pretrained model (scaled down to appropriate size)
				x_resized = torch.nn.functional.interpolate(x,257) ## Upscale to desired input size for squeezenet
				additional_features = self.feature_extractor(x_resized).unsqueeze(dim=1)
				additional_features = additional_features.expand((N,K,16,64,64)).contiguous()
				additional_features = additional_features.view((N*K,16,64,64))
				refine_inp['img'] = torch.cat((refine_inp['img'],additional_features),dim=1)

				delta, h_vert, c_vert = self.refine_net(refine_inp, h_hor, c_hor, h_vert, c_vert)
				lmbda = lmbda + delta
				h_hor = h_vert
				c_hor = c_vert
				_x_hor = _x
				final_masks.append(masks)
				final_mu_x.append(mu_x)


		final_masks = torch.stack(final_masks).permute(1,0,2,3,4,5)
		final_mu_x = torch.stack(final_mu_x).permute(1,0,2,3,4,5)
		neg_kl = (1.+logvar_z-logvar_z.exp()-mu_z.pow(2))
		neg_kl = neg_kl.view((N,K,)+neg_kl.shape[1:])


		if not self.opt.isTrain and not self.opt.no_scores:
			gt = torch.round(100*gt)/100
			gt = (gt[:,:,0]*100 + gt[:,:,1]*10 + gt[:,:,2])
			gt = torch.round(gt)
			gt = torch.unsqueeze(gt, dim = 2)
			gt = torch.unsqueeze(gt, dim = 2)
			ari = adjusted_rand_index(gt, final_masks)
			ari_no_bg = adjusted_rand_index_without_bg(gt, final_masks)
		else:
			ari = []
			ari_no_bg = []
		z = self._sample(mu_z,logvar_z)
		return total_loss, nll, div, flat_e, final_mu_x, final_masks, neg_kl, z, h_hor, ari, ari_no_bg

	"""
	Generate inputs to refinement network
	"""
	def get_refine_inputs(self,_x,mu_x,masks,mask_logits,ll_pxl,lmbda,loss,deviation, _x_hor = None, predict = False ):
		N,K,C,H,W = mu_x.shape

		## Calculate additional non-gradient inputs
		ll_pxl = ll_pxl.expand((N,K,) + ll_pxl.shape[2:]) ## (N,K,1,H,W)
		p_mask_individual = (deviation/(2.*self.var_x)).exp().prod(dim=2,keepdim=True) ## (N,K,1,H,W)
		p_masks = torch.nn.functional.softmax(p_mask_individual, dim=1) ## (N,K,1,H,W)

		## Calculate gradient inputs. If prediction mode then set the gradients to zeros
		if not predict:
			dmu_x = torch.autograd.grad(loss,mu_x,retain_graph=True,only_inputs=True)[0] ## (N,K,C,H,W)
			dmasks = torch.autograd.grad(loss,masks,retain_graph=True,only_inputs=True)[0] ## (N,K,1,H,W)
			dlmbda = torch.autograd.grad(loss,lmbda,retain_graph=True,only_inputs=True)[0] ## (N*K,2*z_dim)
		else:
			dmu_x = torch.zeros_like(mu_x)
			dmasks = torch.zeros_like(masks)
			dlmbda = torch.zeros_like(lmbda)

		## Apply layer norm
		ll_pxl_stable = self.layer_norms[0](ll_pxl).detach()
		dmu_x_stable = self.layer_norms[1](dmu_x).detach()
		dmasks_stable = self.layer_norms[2](dmasks).detach()
		dlmbda_stable = self.layer_norms[3](dlmbda).detach()

		## Generate coordinate channels
		x_mesh = self.x_grid.expand(N,K,-1,-1,-1).contiguous()
		y_mesh = self.y_grid.expand(N,K,-1,-1,-1).contiguous()

		if self.opt.additional_input:
			x_hor_mesh = self.x_grid_hor.expand(N,K,-1,-1,-1).contiguous()
			y_hor_mesh = self.y_grid_hor.expand(N,K,-1,-1,-1).contiguous()

		## Concatenate into vec and mat inputs
		if self.opt.additional_input:
			img_args = (_x,_x_hor, mu_x,masks,mask_logits,dmu_x_stable,dmasks_stable,
				p_masks, ll_pxl_stable, x_mesh, y_mesh, x_hor_mesh, y_hor_mesh)
		else:
			img_args = (_x, mu_x,masks,mask_logits,dmu_x_stable,dmasks_stable,
				p_masks, ll_pxl_stable, x_mesh, y_mesh)

		vec_args = (lmbda, dlmbda_stable)

		img_inp = torch.cat(img_args,dim=2)
		vec_inp = torch.cat(vec_args,dim=1)

		## Reshape
		img_inp = img_inp.view((N*K,)+img_inp.shape[2:])

		return {'img':img_inp, 'vec':vec_inp}

	"""
	Computes the KL-divergence between an isotropic Gaussian distribution over latents
	parameterized by mu_z and logvar_z and the standard normal
	"""
	def _get_div(self,mu_z,logvar_z,N,K):
		kl = ( -0.5*((1.+logvar_z-logvar_z.exp()-mu_z.pow(2)).sum(dim=1)) ).view((N,K))
		return (kl.sum(dim=1)).mean()


	"""
	Computes the KL-divergence between 2 Gaussians
	"""
	def _get_div_gauss(self, mean_1, logvar_1, mean_2, logvar_2, N, K):
		kl = (0.5 * ((logvar_2 - logvar_1 + (logvar_1.exp() + (mean_1 - mean_2).pow(2)) / logvar_2.exp() - 1).sum(dim=1))).view((N,K))
		return (kl.sum(dim=1)).mean()


	"""
	Implements the reparameterization trick
	Samples from standard normal and then scales and shifts by var and mu
	"""
	def _sample(self,mu,logvar):
		std = torch.exp(0.5*logvar)
		return mu + torch.randn_like(std)*std

	"""
	Generates coordinate channels inputs for refinemet network
	"""
	def _create_meshgrid(self):
		H,W = (64,64)
		x_range = torch.linspace(-1.,1.,W)
		y_range = torch.linspace(-1.,1.,H)
		x_grid, y_grid = torch.meshgrid([x_range,y_range])
		x_grid_hor, y_grid_hor = torch.meshgrid([x_range,y_range])
		self.register_buffer('x_grid', x_grid.view((1, 1, 1) + x_grid.shape))
		self.register_buffer('y_grid', y_grid.view((1, 1, 1) + y_grid.shape))
		if self.opt.additional_input:
			self.register_buffer('x_grid_hor', x_grid_hor.view((1, 1, 1) + x_grid_hor.shape))
			self.register_buffer('y_grid_hor', y_grid_hor.view((1, 1, 1) + y_grid_hor.shape))

	"""
	Enable post mortem debugging
	"""
	def _setup_debug(self):
		import sys
		old_hook = sys.excepthook

		def new_hook(typ, value, tb):
			old_hook(typ, value, tb)
			if typ != KeyboardInterrupt:
				import ipdb
				ipdb.post_mortem(tb)

		sys.excepthook = new_hook

	"""
	Checks if any of the model's weights are NaN
	"""
	def has_nan(self):
		for name,param in self.named_parameters():
			if torch.isnan(param).any().item():
				print(param)
				assert False, '{} has nan'.format(name)

	"""
	Checks if any of the model's weight's gradients are NaNs
	"""
	def grad_has_nan(self):
		for name,param in self.named_parameters():
			if torch.isnan(param.grad).any().item():
				print(param)
				print('---------')
				print(param.grad)
				assert False, '{}.grad has nan'.format(name)


	"""
	Save the current model
	"""
	def save(self, epoch_label, device):
		save_filename = '%s_net.pth' % (epoch_label)
		save_path = os.path.join(self.save_dir, save_filename)
		torch.save(self.state_dict(), save_path)

	"""
	Helper loading function that can be used by subclasses
	"""
	def load_network(self, epoch_label, save_dir=''):
		save_filename = '%s_net.pth' % (epoch_label)
		if not save_dir:
			save_dir = self.save_dir
		save_path = os.path.join(save_dir, save_filename)
		if not os.path.isfile(save_path):
			print('%s not exists yet!' % save_path)
		else:
			#network.load_state_dict(torch.load(save_path))
			try:
				self.load_state_dict(torch.load(save_path))
			except:
				pretrained_dict = torch.load(save_path)
				model_dict = self.state_dict()
				try:
					pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
					self.load_state_dict(pretrained_dict)
					if self.opt.verbose:
						print('Pretrained network %s has excessive layers; Only loading layers that are used' % network_label)
				except:
					print('Pretrained network %s has fewer layers; The following are not initialized:' % network_label)
					for k, v in pretrained_dict.items():
						if v.size() == model_dict[k].size():
							model_dict[k] = v

					if sys.version_info >= (3,0):
						not_initialized = set()
					else:
						from sets import Set
						not_initialized = Set()

					for k, v in model_dict.items():
						if k not in pretrained_dict or v.size() != pretrained_dict[k].size():
							not_initialized.add(k.split('.')[0])

					print(sorted(not_initialized))
					self.load_state_dict(model_dict)

