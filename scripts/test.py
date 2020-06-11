# Copyright (c) 2019-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import time
import torchvision.models as models
import torchvision
from torchvision import transforms
import torch
import os
import colorsys
import random
import numpy as np
import math
from tensorboardX import SummaryWriter
from sklearn.metrics.cluster import adjusted_rand_score

from src.model import Model
from src.networks.refine_net import RefineNetLSTM
from src.networks.sbd import SBD
from src.datasets.datasets import ClevrerDataset, FloatBallsVideoDataset
from src.utils.util import latent_walks
from src.utils.util import gif
from src.utils.util import mkdir
from src.utils.test_options import TestOptions
from src.utils.util import adjusted_rand_index

from PIL import ImageFile
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 

ImageFile.LOAD_TRUNCATED_IMAGES = True

opt = TestOptions().parse(save=False)


## Paths for saving models and loading data
save_path = opt.save_path
datapath = opt.datapath
model_name = opt.model_name
save_path += model_name + '/'

## Test Parameters
device = opt.device
batch_size = opt.batch_size
lr = opt.lr
regularization = opt.regularization
parallel = not opt.not_parallel
num_workers = opt.n_workers

## Data Parameters
max_num_samples = opt.ntest
crop_sz = opt.crop_sz  ## Crop initial image down to square image with this dimension
down_sz = opt.down_sz ## Rescale cropped image down to this dimension

## Model Hyperparameters
T = opt.T  ## Number of steps of iterative inference
K =  opt.K  ## Number of slots
z_dim =  opt.z_dim  ## Dimensionality of latent codes
channels_in =  opt.channels_in ## Number of inputs to refinement network (16, + 16 additional if using feature extractor)
if opt.additional_input:
    channels_in += 5
out_channels =  opt.out_channels  ## Number of output channels for spatial broadcast decoder (RGB + mask logits channel)
img_dim = ( opt.img_height,  opt.img_width)  ## Input image dimension
beta =  opt.beta ## Weighting on nll term in VAE loss
gamma = opt.gamma
psi = opt.psi
use_feature_extractor =  opt.use_feature_extractor
mkdir(save_path + "test/")
mode = "test/"
results_dir = save_path + mode
if not opt.predict_frames == 0:
    results_dir = save_path + "predictions_"+str(opt.predict_frames)+"/"
    mkdir(results_dir)
    mode = "predictions/"
else:
    results_dir+="K_"+ str(K)+"T_"+ str(T)+"number_of_frames_"+str(opt.max_num_frames)+"_datapath_"+datapath.replace("/","") +"/"
    mkdir(results_dir)

# Add numbers of the batches that you want to visualize and do the latent walks on. 
# Note: Creating visualization takes up a lot of GPU memory.
batch_to_print = []
batch_to_print_latent = []

if "bb" in opt.datapath:
    test_data = torch.utils.data.DataLoader(
                FloatBallsVideoDataset(datapath, data_type = "test", max_num_samples=max_num_samples, crop_sz=crop_sz, down_sz=down_sz,max_num_frames= opt.max_num_frames + opt.predict_frames, normalize = opt.normalize, no_color = opt.no_color, gt_datapath = opt.gt_datapath),
                batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
elif "clevrer" in opt.datapath:
    test_data = torch.utils.data.DataLoader(
                ClevrerDataset(datapath, data_type = "test", max_num_samples=max_num_samples, down_sz=down_sz, max_num_frames= opt.max_num_frames + opt.predict_frames, normalize = opt.normalize, gt_datapath = opt.gt_datapath),
                batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)   
else:
    print("wrong dataset")
    raise SystemExit

## Create refinement network, decoder and an additional feature extractor
feature_extractor = models.squeezenet1_1(pretrained=True).features[:5]
refine_net = RefineNetLSTM(z_dim, channels_in)
decoder = SBD(z_dim, img_dim, out_channels=out_channels, cond = opt.cond_prior)


## Create the model
v = Model(opt, refine_net, decoder, T, K, z_dim, name=model_name,
           feature_extractor=feature_extractor, beta=beta, gamma= gamma, psi = psi)

## Will use all visible GPUs if parallel=True
pretrained_path = opt.load_pretrain
## Load the network
v.load_network(opt.which_epoch, pretrained_path)

if parallel and torch.cuda.device_count() > 1:
    print('Using {} GPUs'.format(torch.cuda.device_count()))
    v = torch.nn.DataParallel(v)
    v_module = v.module
else:
    parallel = False
    v_module = v


def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def test(model, dataloader, device='cpu', beta=10, gamma=0.1, psi=1.0):
    v = model.to(device)
    colors = random_colors(K)
    ari = 0
    mbatch_cnt = 0
    dataset_size = len(dataloader)*opt.batch_size
    print("Dataset size!!!!:     ", dataset_size)
    total_steps = 0
    print("Total steps: ", total_steps)

    epoch_start_time = time.time()
    ari_score = []
    ari_no_bg_score = []
    mse_score = []
    for i, mbatch in enumerate(dataloader, start=mbatch_cnt):

        total_steps += opt.batch_size

        x = mbatch.to(device)

        if opt.no_color or not opt.gt_datapath == opt.datapath:
            gt = x[:,:,3:6]
            x = x[:,:,:3]
        else:
            gt = x
        N, F, C, H, W = x.shape

        
        ## Forward pass
        loss, nll, div, entropy, mu_x, masks, neg_kl, z, h, ari, ari_no_bg = v.forward(x, gt)
        mse = 0.0
        output_means = (mu_x * masks).sum(dim=2)
        if not opt.no_scores:
            ari = ari.mean()
            ari_no_bg = ari_no_bg.mean()
            mse = torch.nn.functional.mse_loss(output_means, x)
            if i % 10 == 0:
                print('Ari score: %f (batch %d, total_batches %d)' % (ari, i, len(dataloader)))
                print('Arino bg score: %f (batch %d, total_batches %d)' % (ari_no_bg, i, len(dataloader)))
                print('Mse score: %f (batch %d, total_batches %d)' % (mse, i, len(dataloader)))
            ari_score.append(ari.data.cpu())
            ari_no_bg_score.append(ari_no_bg.data.cpu())
            mse_score.append(mse.data.cpu())
        del loss, nll, div, entropy, ari, ari_no_bg, neg_kl,  mse


        ## Visualization part. Builds GIFs and images with masks and separate reconstructions.
        if i in batch_to_print:

            single_mask_colors =  mu_x.clone()
            single_mask_colors = single_mask_colors.expand(-1,-1,-1,3,-1,-1)
            c = 0
            for k in range(K):
                for c in range(3):
                    single_mask_colors[:,:,k,c,:,:] = colors[k][c] 

            mask_image_debug = (single_mask_colors*masks).sum(dim=2)

            if opt.predict_frames == 0:
                gif_masks = torch.unbind(mask_image_debug, dim = 0)
                gif_masks = torch.cat(gif_masks, dim = 2)
                gif_means = torch.unbind(output_means, dim = 0)
                gif_means = torch.cat(gif_means, dim = 2)
                final_gif = torch.cat([gif_masks, gif_means], dim = 3)
                single_mus = []
                for slot in range(K):
                    mu_gif = mu_x[:,:,slot]
                    mu_gif = torch.unbind(mu_gif, dim = 0)
                    mu_gif = torch.cat(mu_gif, dim = 2)
                    single_mus.append(mu_gif)
                single_mus = torch.cat(single_mus, dim = 3)
                final_gif = torch.cat([final_gif, single_mus], dim=3)
                del mu_gif, single_mus
            else: 
                no_sim_masks = mask_image_debug[:, :opt.max_num_frames]
                sim_masks = mask_image_debug[:, opt.max_num_frames:]
                no_sim_gt = gt[:, :opt.max_num_frames]
                sim_gt = gt[:, opt.max_num_frames:]
                new_sim_masks = []
                new_sim_gt = []
                for b in range(opt.batch_size): 
                    batch_masks = []
                    batch_gt = []
                    for sim_num in range(opt.predict_frames):
                        im = transforms.ToPILImage()(sim_masks[b,sim_num].data.cpu()).convert("RGB")
                        im_gt = transforms.ToPILImage()(sim_gt[b,sim_num].data.cpu()).convert("RGB")
                        draw = ImageDraw.Draw(im)
                        font = ImageFont.truetype("./arial.ttf", 7)
                        draw.text((0, 0),"Prediction",(0,0,0),font=font)
                        draw = ImageDraw.Draw(im_gt)
                        draw.text((0, 0),"Ground truth",(255,255,255),font=font)
                        transform_list = [transforms.ToTensor()]
                        im = transforms.Compose(transform_list)(im) 
                        im_gt = transforms.Compose(transform_list)(im_gt) 
                        batch_masks.append(im.to(device))
                        batch_gt.append(im_gt.to(device))
                    new_sim_masks.append(torch.stack(batch_masks, dim =0))
                    new_sim_gt.append(torch.stack(batch_gt, dim =0))
                sim_masks = torch.stack(new_sim_masks, dim =0)
                sim_gt = torch.stack(new_sim_gt, dim =0)
                mask_image_debug = torch.cat([no_sim_masks, sim_masks], dim = 1)
                gt = torch.cat([no_sim_gt, sim_gt], dim = 1)
                gif_masks = torch.unbind(mask_image_debug, dim = 0)
                gif_masks = torch.cat(gif_masks, dim = 2)
                gif_gt = torch.unbind(gt, dim = 0)
                gif_gt = torch.cat(gif_gt, dim = 2)
                final_gif = torch.cat([gif_masks,gif_gt], dim = 3)
                del gif_gt, draw
            gif(results_dir+'gif_{}'.format(i) + '.png', final_gif.mul(255).add_(0.5).clamp_(0, 255).permute(0,2,3,1).to('cpu', torch.uint8).detach().numpy())
            del final_gif, gif_masks, gif_means
            if opt.predict_frames == 0:
                Y = torch.zeros_like(masks)
                Y.scatter_(2,torch.argmax(masks,dim=2,keepdim=True), 1)
                for batch in range(batch_size):
                    just_masks = []
                    just_mus = []
                    for slot in range(K):
                        just_mask = Y[batch,int(np.ceil(F/2.0)),slot]*single_mask_colors[batch,int(np.ceil(F/2.0)),slot]
                        just_mask[(just_mask.sum(dim=0)<0.0009).unsqueeze(dim=0).expand(3,64,64)]=1.0
                        just_masks.append(just_mask)
                        just_mu = mu_x[batch,F-1,slot]
                        just_mus.append(just_mu)
                    just_masks = torch.stack(just_masks, dim = 0)
                    just_mus = torch.stack(just_mus, dim = 0)
                    image = torch.cat([mask_image_debug[batch], output_means[batch], x[batch]], dim = 2)
                    grid_img = torchvision.utils.save_image(image, results_dir +'results_{}_{}'.format(batch, i) + '.png', nrow= opt.max_num_frames )
                    grid_img = torchvision.utils.save_image(just_masks, results_dir +'results_just_masks_{}_{}'.format(batch, i) + '.png', nrow=K )
                    grid_img = torchvision.utils.save_image(just_mus, results_dir +'results_just_mus_{}_{}'.format(batch, i) + '.png', nrow= K )
                del Y, just_mus, just_masks, image
            else:
                for batch in range(batch_size):
                    image = torch.cat([mask_image_debug[batch], output_means[batch], x[batch]], dim = 2)
                    grid_img = torchvision.utils.save_image(image, results_dir +'results_{}_{}'.format(batch, i) + '.png', nrow= opt.max_num_frames + opt.predict_frames )
                for batch in range(batch_size):
                    image = torch.cat([mask_image_debug[batch], gt[batch]], dim = 2)
                    grid_img = torchvision.utils.save_image(image, results_dir +'results_only_masks_{}_{}'.format(batch, i) + '.png', nrow= opt.max_num_frames + opt.predict_frames )
                 
            del grid_img, gt, x, mask_image_debug, single_mask_colors

        # Printing the latent walks
        if i in batch_to_print_latent:
            xs, masks = latent_walks(decoder, z, h, N, K)
            print("Final shape XS and masks:", xs.shape, masks.shape)
            for j in range(K):
                full_image = (xs[j] * masks[j]).sum(dim=2)
                grid_img = torchvision.utils.save_image(torch.flatten(full_image, end_dim = 1), save_path + 'images/'+'results_latent_walk_{}_slot_{}'.format(i, j) + '.png', nrow = 7)   
                del full_image
            grid_img = torchvision.utils.save_image(x[0], save_path + 'images/'+'latent_walk_gt_{}'.format(i) + '.png', nrow= opt.max_num_frames)

        del mu_x, masks, output_means, z, h
        torch.cuda.empty_cache()
    if not opt.no_scores:    
        print("ARI: ", torch.mean(torch.stack(ari_score, 0)))
        print("ARI without BG: ", torch.mean(torch.stack(ari_no_bg_score, 0)))
        print("MSE: ", torch.mean(torch.stack(mse_score, 0)))
        
    print('Time Taken: %d sec' % (time.time() - epoch_start_time))

## Run training function
test(v, test_data, device=device, beta = beta, gamma = gamma, psi = psi)
