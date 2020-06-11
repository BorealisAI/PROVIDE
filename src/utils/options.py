# Copyright (c) 2019-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import os
from . import util
import torch

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        # experiment specifics
        self.parser.add_argument('--datapath', type=str, default='../bb_binary/', help='name of the experiment. It decides where to store samples and models, e.g ./data/CLEVR_v1.0/images/')
        self.parser.add_argument('--gt_datapath', type = str, default='../bb_color/', help='ground truth masks in case of the no color dataset')
        self.parser.add_argument('--dataset', type=str, default='balls', help='name of the dataset, either "balls" or "svqa" ')
        self.parser.add_argument('--save_path', type=str, default='./trained_models/', help='models are saved here')
        self.parser.add_argument('--model_name', type=str, default='clevrer', help='which model to use')
        self.parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        self.parser.add_argument('--load_pretrain', type=str, default='', help='load the pretrained model from the specified location')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--device', type=str, default='cuda:0', help='which device to use: cuda:0 or cpu')
        self.parser.add_argument('--batch_size', type=int, default=16, help='batch size')
        self.parser.add_argument('--lr', type=float, default=3e-4, help='initial learning rate for adam')
        self.parser.add_argument('--regularization', type=float, default=0., help='regularization')
        self.parser.add_argument('--not_parallel', action='store_true', help='if we parallelize')
        self.parser.add_argument('--n_workers', type=int, default=4, help='number of workers')
        self.parser.add_argument('--max_num_samples', type=int, default=50000, help='total number of smaples we use per epoch')
        self.parser.add_argument('--crop_sz', type=int, default=400, help='crop initial image down to square image with this dimension')
        self.parser.add_argument('--down_sz', type=int, default=64, help='rescale cropped image down to this dimension')
        self.parser.add_argument('--T', type=int, default=5, help='number of steps of iterative inference')
        self.parser.add_argument('--K', type=int, default=5, help='number of slots')
        self.parser.add_argument('--z_dim', type=int, default=64, help='dimensionality of latent codes')
        self.parser.add_argument('--channels_in', type=int, default=32, help='number of inputs to refinement network (16, + 16 additional if using feature extractor)')
        self.parser.add_argument('--out_channels', type=int, default=4, help='number of output channels for spatial broadcast decoder (RGB + mask logits channel)')
        self.parser.add_argument('--img_height', type = int, default= 64, help='height of the image after the rescaling')
        self.parser.add_argument('--img_width', type = int, default= 64, help='width of the image after the rescaling')
        self.parser.add_argument('--beta', type = float, default= 100., help='reconstraction loss weight')
        self.parser.add_argument('--gamma', type = float, default= 0.1, help='entropy loss weight')
        self.parser.add_argument('--psi', type = float, default= 10., help='kl weight')
        self.parser.add_argument('--use_feature_extractor', action='store_false', help='number of inputs to refinement network (16, + 16 additional if using feature extractor)')
        self.parser.add_argument('--display_freq', type=int, default=100, help='frequency of showing training results on screen')
        self.parser.add_argument('--print_freq', type=int, default=500, help='frequency of showing training results on console')
        self.parser.add_argument('--save_latest_freq', type=int, default=500, help='frequency of saving the latest results')
        self.parser.add_argument('--save_iter_freq', type=int, default=5000, help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--normalize', action='store_true', help='normalization of the images to be between [-1, 1]')
        self.parser.add_argument('--no_color', action='store_true', help='set to true if the dataset is grey scale')
        self.parser.add_argument('--additional_input', action='store_true', help='set to true if want to use an additional input to the refinement network')
        self.parser.add_argument('--use_entropy', action='store_true', help='set to true if want to use additional entropy loss which separets the masks')
        self.parser.add_argument('--max_num_frames', type=int, default=30, help='number of frames per video')
        self.parser.add_argument('--no_cond_prior', action='store_true',  help='not using the conditional prior')
        self.parser.add_argument('--param_schedule', action='store_true',  help='use parameter schedule if set to true')
        self.parser.add_argument('--predict_frames', type=int, default=0, help='number of frames to predict')
        self.parser.add_argument('--no_scores', action='store_true',  help='if set to true do not compute the score ')
        
        self.initialized = True

    def parse(self, save=True):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain 
        self.opt.cond_prior = not self.opt.no_cond_prior
        args = vars(self.opt)
        if "binary" in self.opt.datapath:
            self.opt.no_color = True
            self.opt.use_entropy = True

        if self.opt.predict_frames > 0:
            self.opt.predict = True 

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk
        expr_dir = os.path.join(self.opt.save_path, self.opt.model_name)
        util.mkdirs(expr_dir)
        util.mkdirs(os.path.join(expr_dir, "images"))
        if save and not self.opt.continue_train:
            file_name = os.path.join(expr_dir, 'opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('------------ Options -------------\n')
                for k, v in sorted(args.items()):
                    opt_file.write('%s: %s\n' % (str(k), str(v)))
                opt_file.write('-------------- End ----------------\n')
        return self.opt