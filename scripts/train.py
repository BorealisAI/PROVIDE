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

import time
import torchvision.models as models
import torchvision
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
from src.datasets.datasets import FloatBallsVideoDatasetMoreData,  ClevrerDatasetMoreData
from src.utils.util import display_samples
from src.utils.util import gif
from src.utils.util import adjusted_rand_index
from src.utils.train_options import TrainOptions

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

opt = TrainOptions().parse()

## Paths for saving models and loading data
save_path = opt.save_path
datapath = opt.datapath
gt_datapath = opt.gt_datapath
model_name = opt.model_name
save_path += model_name + '/'

## Training Parameters
device = opt.device
batch_size = opt.batch_size
lr = opt.lr
regularization = opt.regularization
n_epochs = opt.n_epochs
parallel = not opt.not_parallel
num_workers = opt.n_workers


## Data Parameters
max_num_samples = opt.max_num_samples
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

if "bb" in opt.datapath:
    train_data = torch.utils.data.DataLoader(
        FloatBallsVideoDatasetMoreData(datapath, max_num_samples=max_num_samples, crop_sz=crop_sz, down_sz=down_sz,max_num_frames= opt.max_num_frames, normalize = opt.normalize, no_color = opt.no_color, gt_datapath = opt.gt_datapath),
        batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
elif "clevrer" in opt.datapath:
    train_data = torch.utils.data.DataLoader(
                ClevrerDatasetMoreData(datapath, max_num_samples=max_num_samples, down_sz=down_sz,max_num_frames= opt.max_num_frames, normalize = opt.normalize),
                batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)   
else:
    print("wrong dataset")
    raise SystemExit

## Create refinement network, decoder, and (optionally) feature extractor
## 		Could speed up training by pre-computing squeezenet outputs since we just use this as a feature extractor
## 		Could also do this as a pre-processing step in dataset class
feature_extractor = models.squeezenet1_1(pretrained=True).features[:5] if use_feature_extractor else None
refine_net = RefineNetLSTM(z_dim, channels_in)
decoder = SBD(z_dim, img_dim, out_channels=out_channels, cond = opt.cond_prior)

## Create the  model
v = Model(opt, refine_net, decoder, T, K, z_dim, name=model_name,
               feature_extractor=feature_extractor, beta=beta, gamma= gamma)

## Will use all visible GPUs if parallel=True
    # load the network
if opt.continue_train or opt.load_pretrain:
    pretrained_path = opt.load_pretrain
    v.load_network(opt.which_epoch, pretrained_path)

v.train()
if parallel and torch.cuda.device_count() > 1:
    print('Using {} GPUs'.format(torch.cuda.device_count()))
    v = torch.nn.DataParallel(v)
    v_module = v.module
else:
    parallel = False
    v_module = v



## Set up optimizer and data logger
optimizer = torch.optim.Adam(v.parameters(), lr=lr, weight_decay=regularization)
writer = SummaryWriter(save_path + 'logs/')

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


def train(model, dataloader, n_epochs=10, device='cpu', beta=10, gamma=0.1, psi=1.0):
    v = model.to(device)
    colors = random_colors(K)
    iter_path = os.path.join(save_path, 'iter.txt')
    ari = 0

    if opt.continue_train:
        try:
            start_epoch, epoch_iter = np.loadtxt(iter_path , delimiter=',', dtype=int)
        except:
            start_epoch, epoch_iter = 1, 0
        print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))
    else:
        start_epoch, epoch_iter = 1, 0

    mbatch_cnt = int(epoch_iter/batch_size)

    dataset_size = len(dataloader)*opt.batch_size
    print("Dataset size!!!!:     ", dataset_size)
    total_steps = (start_epoch-1) * dataset_size + epoch_iter
    print("Total steps: ", total_steps)
    save_delta = total_steps % opt.save_latest_freq
    display_delta = total_steps % opt.display_freq
    print_delta = total_steps % opt.print_freq 

    for epoch in range(start_epoch, n_epochs):
        epoch_start_time = time.time()
        print('On epoch {}'.format(epoch))
        if epoch != start_epoch:
            epoch_iter = epoch_iter % dataset_size
            mbatch_cnt = int(epoch_iter/batch_size)
        for i, mbatch in enumerate(dataloader, start=mbatch_cnt):

            epoch_iter += opt.batch_size
            total_steps += opt.batch_size
            x = mbatch.to(device)

            if opt.no_color:
                gt = x[:,:,3:6]
                x = x[:,:,:3]
            else:
                gt = x
            N, F, C, H, W = x.shape

            ## Forward pass
            loss, nll, div, entropy, mu_x, masks, neg_kl, z, h, _, _ = v.forward(x, gt)

            ## Process Outputs
            nll = nll.mean()
            div = div.mean()
            entropy = entropy.mean()
            loss = loss.mean()

            if not opt.no_color:
                if nll.item() > 600.:
                    beta = 300.
                elif nll.item() > 200.:
                    beta= 100.

            v_module.gamma = gamma
            v_module.beta = beta

            assert not torch.isnan(loss).item(), 'Nan loss: loss / div / nll: {}/{}/{}/{}'.format(loss, div, nll, entropy)
            output_means = (mu_x * masks).sum(dim=2)

            mse = torch.nn.functional.mse_loss(output_means, x)

            ## Backwards Pass 
            optimizer.zero_grad()
            loss.backward(retain_graph=False)
            torch.nn.utils.clip_grad_norm_(v.parameters(), 5.0, norm_type=2)
            v_module.grad_has_nan()

            ## Update model
            assert not v_module.has_nan(), 'Model has nan pre-opt step'
            optimizer.step()
            assert not v_module.has_nan(), 'Model has nan post-opt step'

        
            ## Print and log outputs
            if i % opt.print_freq == 0 or ari == 0:

                single_mask  = torch.zeros_like(masks)
                _, idx = torch.max(masks, dim = 2)
                s = single_mask.shape
                idx = idx.unsqueeze(dim = 2).expand(s)
                r =  torch.from_numpy(np.arange(K)).unsqueeze(dim=0).unsqueeze(dim=0).unsqueeze(dim=3).unsqueeze(dim=3).unsqueeze(dim=5).to(device)
                r = r.expand(s)
                single_mask = (idx == r).to(device, dtype=torch.float32)


                single_mask_colors =  mu_x.clone()
                single_mask_colors = single_mask_colors.expand(-1,-1,-1,3,-1,-1)
                c = 0
                for k in range(K):
                    for c in range(3):
                        single_mask_colors[:,:,k,c,:,:] = colors[k][c] 
                ##DEBUGGING 
                mask_image_debug = (single_mask_colors*single_mask).sum(dim=2)
                grid_img = torchvision.utils.save_image(torch.flatten(mask_image_debug, end_dim = 1), save_path + 'images/'+'results_singel_masks_epoch_{}'.format(epoch) + '.png', nrow= opt.max_num_frames)
                #if not opt.dataset == "clevrer":
                gif_masks = torch.unbind(mask_image_debug, dim = 0)
                gif_masks = torch.cat(gif_masks, dim = 2)
                gif(save_path +'images/'+'gif_{}'.format(epoch) + '.png', gif_masks.mul(255).add_(0.5).clamp_(0, 255).permute(0,2,3,1).to('cpu', torch.uint8).detach().numpy())
                new_mask = torch.round(100*mask_image_debug)/100
                new_mask = (new_mask[:,:,0]*100 + new_mask[:,:,1]*10 + new_mask[:,:,2])
                new_mask = torch.round(new_mask) 
                new_mask = torch.flatten(new_mask, start_dim = 1)
                new_mask = new_mask.detach().cpu().numpy()


                new_x = torch.round(100*gt)/100
                new_x = (new_x[:,:,0]*100 + new_x[:,:,1]*10 + new_x[:,:,2])
                new_x = torch.round(new_x)
                new_x = torch.unsqueeze(new_x, dim = 2)
                new_x = torch.unsqueeze(new_x, dim = 2)
                ari = adjusted_rand_index(new_x, masks)

    
            output_means = output_means[0]


            if i % 50 == 0:    
                grid_img = torchvision.utils.save_image(output_means, save_path + 'images/'+'results_epoch_{}'.format(epoch) + '.png', nrow= opt.max_num_frames)
                grid_img = torchvision.utils.save_image(x[0], save_path + 'images/'+'gt_epoch_{}'.format(epoch) + '.png', nrow= opt.max_num_frames)
                if opt.no_color:
                    grid_img = torchvision.utils.save_image(gt[0], save_path + 'images/'+'gt_masks_epoch_{}'.format(epoch) + '.png', nrow= opt.max_num_frames)
                print('\nOn mbatch {}:'.format(i*batch_size))
                print('\nTotal steps{}:'.format(total_steps))
                print('model.beta = {}'.format(beta))
                print('model.gamma = {}'.format(gamma))
                print('model.psi = {}'.format(psi))
                print('Curr loss: {}'.format(loss.item()))
                print('Curr final nll: {}'.format(nll.item()))
                print('Curr final div: {}'.format(div.item()))
                print('Curr final entropy: {}'.format(entropy.item()))
                print('Curr mse: {}'.format(mse.item()))
                print('Curr ari: {}'.format(ari))


            writer.add_scalar('loss', loss.item(), total_steps)
            writer.add_scalar('final nll', nll.item(), total_steps)
            writer.add_scalar('final div', div.item(), total_steps)
            writer.add_scalar('final entropy', entropy.item(), total_steps)
            writer.add_scalar('final mse', mse.item(), total_steps)
            writer.add_scalar('ARI', ari.item(), total_steps)

            ### save latest model
            if total_steps % opt.save_latest_freq == save_delta:
                print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
                v_module.save('latest', device)
                np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')

             ### save model for this iteration
            if total_steps % opt.save_iter_freq == 0:
                print('saving the model at the epoch %d, iters %d' % (epoch, total_steps))
                print('-------------------------------------------------')
                print('-------------------------------------------------')
                v_module.save('latest', device)
                v_module.save(str(epoch)+"_"+str(epoch_iter), device)
                np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')
                
        # end of epoch
        iter_end_time = time.time()
        print('End of epoch %d / %d \t Time Taken: %d sec' %
            (epoch, n_epochs, time.time() - epoch_start_time))



## Run training function
train(v, train_data, n_epochs=n_epochs, device=device, beta = beta, gamma = gamma, psi = psi)
