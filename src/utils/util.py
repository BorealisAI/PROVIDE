# Copyright (c) 2019-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import time
import math
from numbers import Number
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import visdom
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
from moviepy.editor import ImageSequenceClip



def gif(filename, array, fps=10, scale=1.0):
    """Creates a gif given a stack of images using moviepy
    Usage
    -----
    >>> X = randn(100, 64, 64)
    >>> gif('test.gif', X)
    """

    # ensure that the file has the .gif extension
    fname, _ = os.path.splitext(filename)
    filename = fname + '.gif'

    # copy into the color dimension if the images are black and white
    if array.ndim == 3:
        array = array[..., np.newaxis] * np.ones(3)

    # make the moviepy clip
    clip = ImageSequenceClip(list(array), fps=fps).resize(scale)
    clip.write_gif(filename, fps=fps)
    return clip


def latent_walks(model, zs, h, N, K):
    with torch.no_grad():
        zs = zs[0:K]
        batch_size, z_dim = zs.size()
        _, h_dim = h.size()
        xs = []
        masks = []
        delta = torch.autograd.Variable(torch.linspace(-2.5, 2.5, 7), volatile=True).type_as(zs)

        for k in range(batch_size):
            for i in range(z_dim):
                vec = Variable(torch.zeros(z_dim)).view(1, z_dim).expand(batch_size, 7, z_dim).contiguous().type_as(zs)
                vec[k, :, i] = 1
                vec = vec * delta[None,:,None]
                zs_delta = zs.clone().view(batch_size, 1, z_dim)
                zs_delta[k, :, i] = 0
                zs_walk = zs_delta + vec
                zs_walk= torch.squeeze(zs_walk).permute(1,0,2).contiguous() 

                vec_2 = Variable(torch.zeros(h_dim)).view(1, h_dim).expand(batch_size, 7, h_dim).contiguous().type_as(h)
                vec_2[k, :, i] = 1
                vec_2 = vec_2 * delta[None, :, None]
                h_delta = h.clone().view(batch_size, 1, h_dim)
                h_delta[k, :, i] = 0
                h_walk = h_delta + vec_2
                h_walk= h_walk.permute(1,0,2).contiguous() 

                dec_out = model(zs_walk.view(-1, z_dim), h_walk.view(-1, h_dim))
                xs_walk, masks_walk = dec_out[:,:3,:,:], dec_out[:,3,:,:]
                xs_walk = xs_walk.view((7,K)+xs_walk.shape[1:]) ##(7,K,C,H,W)
                masks_walk = masks_walk.view((7,K)+masks_walk.shape[1:]) ##(7,K,H,W)
                masks_walk = torch.nn.functional.softmax(masks_walk ,dim=1).unsqueeze(dim=2)
                masks.append(masks_walk)
                xs.append(xs_walk)

        xs = torch.stack(xs, dim = 0)
        xs = xs.view((batch_size, z_dim) + xs.shape[1:]).data.cpu()
        masks = torch.stack(masks, dim = 0)
        masks = masks.view((batch_size, z_dim) + masks.shape[1:]).data.cpu()
        return xs, masks


def adjusted_rand_index(groups, gammas):
    """
    Inputs:
        groups: shape=(N, F, 1, 1, W, H)
            These are the masks
        gammas: shape=(N, F, K, 1, W, H)
            These are the gammas as predicted by the network
    """
    # reshape gammas and convert to one-hot
    yshape = list(gammas.size())
    gammas = gammas.contiguous().view(yshape[0]*yshape[1], yshape[2], yshape[3] * yshape[4] * yshape[5])
    tensor = torch.LongTensor
    if torch.cuda.is_available():
        tensor = torch.cuda.LongTensor
    Y = tensor(yshape[0] * yshape[1], yshape[2], yshape[3] * yshape[4] * yshape[5]).zero_()
    Y.scatter_(1,torch.argmax(gammas,dim=1,keepdim=True), 1)
    # reshape masks
    gshape = list(groups.size())
    groups = groups.view(gshape[0] * gshape[1], 1, gshape[3] * gshape[4] * gshape[5])
    G = tensor(gshape[0] * gshape[1], torch.max(groups).int()+1, gshape[3] * gshape[4] * gshape[5]).zero_()
    G.scatter_(1,groups.long(), 1)
    # now Y and G both have dim (B*T, K, N) where N=W*H*C
    M = torch.ge(groups, -0.5).float()
    n = torch.sum(torch.sum(M, 2), 1)
    DM = G.float()
    YM = Y.float()
    # contingency table for overlap between G and Y
    nij = torch.einsum('bij,bkj->bki', (YM, DM))
    a = torch.sum(nij, dim=1)
    b = torch.sum(nij, dim=2)
    # rand index
    rindex = torch.sum(torch.sum(nij * (nij-1), 2),1).float()
    aindex = torch.sum(a * (a-1), dim=1).float()
    bindex = torch.sum(b * (b-1), dim=1).float()
    expected_rindex = aindex * bindex / (n*(n-1))
    max_rindex = (aindex + bindex) / 2
    ARI = (rindex - expected_rindex)/(max_rindex - expected_rindex)
    mean_ARI = torch.mean(ARI)
    del yshape, Y, gshape, G, M, n, DM, YM, nij, a, b, rindex, bindex, expected_rindex, max_rindex, ARI
    return mean_ARI

def adjusted_rand_index_without_bg(groups, gammas):
    """
    Inputs:
        groups: shape=(N, F, 1, 1, W, H)
            These are the masks
        gammas: shape=(N, F, K, 1, W, H)
            These are the gammas as predicted by the network
    """
    # reshape gammas and convert to one-hot
    yshape = list(gammas.size())
    gammas = gammas.contiguous().view(yshape[0]*yshape[1], yshape[2], yshape[3] * yshape[4] * yshape[5])
    tensor = torch.LongTensor
    if torch.cuda.is_available():
        tensor = torch.cuda.LongTensor
    Y = tensor(yshape[0] * yshape[1], yshape[2], yshape[3] * yshape[4] * yshape[5]).zero_()
    Y.scatter_(1,torch.argmax(gammas,dim=1,keepdim=True), 1)
    # reshape masks
    gshape = list(groups.size())
    groups = groups.view(gshape[0] * gshape[1], 1, gshape[3] * gshape[4] * gshape[5])
    G = tensor(gshape[0] * gshape[1], torch.max(groups).int()+1, gshape[3] * gshape[4] * gshape[5]).zero_()
    G.scatter_(1,groups.long(), 1)
    # now Y and G both have dim (B*T, K, N) where N=W*H*C
    # mask entries with group 0
    M = torch.ge(groups, 0.5).float()
    n = torch.sum(torch.sum(M, 2), 1)
    DM = G.float() * M
    YM = Y.float() * M
    # contingency table for overlap between G and Y
    nij = torch.einsum('bij,bkj->bki', (YM, DM))
    a = torch.sum(nij, dim=1)
    b = torch.sum(nij, dim=2)
    # rand index
    rindex = torch.sum(torch.sum(nij * (nij-1), 2),1).float()
    aindex = torch.sum(a * (a-1), dim=1).float()
    bindex = torch.sum(b * (b-1), dim=1).float()
    expected_rindex = aindex * bindex / (n*(n-1))
    max_rindex = (aindex + bindex) / 2
    ARI = (rindex - expected_rindex)/(max_rindex - expected_rindex)
    ## If have some nan values then replace then with 1.0. This happens whenether 
    #  ther is one object on the image and it was correctly classified in every pixel. 
    ARI[ARI!=ARI] = 1.0
    mean_ARI = torch.mean(ARI)
    del yshape, Y, gshape, G, M, n, DM, YM, nij, a, b, rindex, bindex, expected_rindex, max_rindex, ARI
    return mean_ARI

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)