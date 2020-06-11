# Copyright (c) 2019-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from torch.utils.data import Dataset
from skimage.transform import rescale, rotate
import torchvision.transforms as transforms
from skimage import io
import numpy as np
import torch
import os

"""
Pytorch dataset class for loading the CLEVRER sequences fof frames
"""
class ClevrerDatasetMoreData(Dataset):

    def __init__(self,datapath,data_type='train',max_num_samples=50000, crop_sz=320, down_sz=64, max_num_frames=10, normalize = False):
        suffix = data_type
        self.datapath = datapath + suffix + '/'
        self.max_num_samples = max_num_samples
        self.max_num_frames = max_num_frames
        self.normalize = normalize
        self.crop_sz = crop_sz
        self.down_scale = down_sz/crop_sz
        self.folders = [ f.path for f in os.scandir(self.datapath) if f.is_dir() ]
        print("Number of folders: ", len(self.folders))
        if len(self.folders) < self.max_num_samples:
            self.max_num_samples = len(self.folders)

    def __len__(self):
        return self.max_num_samples * (128 // self.max_num_frames)

    def __getitem__(self,idx):
        folder = self.folders[idx // (128 // self.max_num_frames)]
        start = (idx % (128 // self.max_num_frames)) * self.max_num_frames 
        all_frames = []
        for frame_idx in range(start, start + self.max_num_frames):
            imgname = str(folder) + "/" + str(frame_idx)
            imgpath = imgname + '.png'
            scaled_img = self.rescale_img(io.imread(imgpath))
            img = torch.tensor(scaled_img,dtype=torch.float32).permute((2,0,1))
            all_frames.append(img)
        return torch.stack(all_frames,dim=0)

    def rescale_img(self,img):
        H,W,C = img.shape  
        dW = abs(W-self.crop_sz)//2
        crop = img[:,dW:-dW,:3]
        down = rescale(crop,self.down_scale, order = 3, mode='reflect', multichannel=True)
        return down


"""
Pytorch dataset class for loading the CLEVRER sequences fof frames
"""
class ClevrerDataset(Dataset):

    def __init__(self,datapath,data_type='train',max_num_samples=50000, crop_sz=320, down_sz=64, max_num_frames=10, normalize = False, gt_datapath = ""):
        suffix = data_type
        self.datapath = datapath + suffix + '/'
        self.gt_datapath = gt_datapath + suffix + '/'
        self.max_num_samples = max_num_samples
        self.max_num_frames = max_num_frames
        self.normalize = normalize
        self.crop_sz = crop_sz
        self.down_scale = down_sz/crop_sz
        self.folders = [ f.path for f in os.scandir(self.datapath) if f.is_dir() ]
        print("Number of folders: ", len(self.folders))
        if len(self.folders) < self.max_num_samples:
            self.max_num_samples = len(self.folders)

    def __len__(self):
        return self.max_num_samples

    def __getitem__(self,idx):
        folder = self.folders[idx]
        all_frames = []
        all_frames_gt = []
        for frame_idx in range(self.max_num_frames):
            imgname = str(folder) + "/" + str(frame_idx)
            imgpath = imgname + '.png'
            scaled_img = self.rescale_img(io.imread(imgpath))
            img = torch.tensor(scaled_img,dtype=torch.float32).permute((2,0,1))
            if not (self.gt_datapath == self.datapath):
                imgname_gt = self.gt_datapath + "video_" + str(folder).split("/")[-1] + "/" + str(frame_idx)
                imgpath_gt = imgname_gt + '.png'
                scaled_img_gt = self.rescale_img(io.imread(imgpath_gt),  order = 0, anti_aliasing = False)
                img_gt = torch.tensor(scaled_img_gt,dtype=torch.float32).permute((2,0,1))
                all_frames_gt.append(img_gt)
            all_frames.append(img)
        result = torch.cat((torch.stack(all_frames,dim=0), torch.stack(all_frames_gt,dim=0)), dim=1) if ( not self.gt_datapath == self.datapath) else torch.stack(all_frames,dim=0)
        return result

    def rescale_img(self,img, order = 3, anti_aliasing= True):
        H,W,C = img.shape  
        dW = abs(W-self.crop_sz)//2
        crop = img[:,dW:-dW,:3]
        down = rescale(crop,self.down_scale, order = order, mode="reflect", multichannel=True, anti_aliasing = anti_aliasing)
        return down
    def rescale_img_gt(self, img):
        H,W,C = img.shape  
        dW = abs(W-self.crop_sz)//2
        crop = img[:,dW:-dW,:3]
        crop = torch.tensor(crop ,dtype=torch.float32).permute((2,0,1))
        down = torch.nn.functional.interpolate(crop.unsqueeze(0), scale_factor = self.down_scale, mode='nearest')
        down = down.squeeze(0)
        return down
        
"""
Pytorch dataset class for loading pre-generated images from the Floating Balls dataset
"""
class FloatBallsVideoDatasetMoreData(Dataset):

    def __init__(self,datapath,data_type='train',max_num_samples=50000,crop_sz=256,down_sz=64, max_num_frames=10, normalize = False, no_color = True, gt_datapath = ""):
        suffix = data_type
        self.datapath = datapath + suffix + '/'
        self.gt_datapath = gt_datapath + suffix + '/'
        self.max_num_samples = max_num_samples
        self.max_num_frames = max_num_frames
        self.normalize = normalize
        self.no_color = no_color
        self.folders = [ f.path for f in os.scandir(self.datapath) if f.is_dir() ]
        print(len(self.folders))
        if len(self.folders) < self.max_num_samples:
            self.max_num_samples = len(self.folders)

    def __len__(self):
        return self.max_num_samples * (51 // self.max_num_frames)

    def __getitem__(self,idx):
        folder = idx // (51 // self.max_num_frames)
        start = (idx % (51 // self.max_num_frames)) * self.max_num_frames 
        all_frames = []
        all_frames_gt = []
        for frame_idx in range(start, start + self.max_num_frames):
            imgname = self.datapath + str(folder) + "/" + str(frame_idx)
            imgpath = imgname + '.png'
            img = np.array(io.imread(imgpath))
            transform_list = [transforms.ToTensor()]
            if self.normalize:
                if self.no_color:
                    transform_list.append(transforms.Normalize((0.5), (0.5)))
                else: 
                    transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
            img = transforms.Compose(transform_list)(img) 
            if self.no_color:
                img = img.repeat(3,1,1)
                imgname_gt = self.gt_datapath + str(folder) + "/" + str(frame_idx)
                imgpath_gt = imgname_gt + '.png'
                img_gt = np.array(io.imread(imgpath_gt))
                img_gt = transforms.Compose(transform_list)(img_gt) 
                all_frames_gt.append(img_gt)
            all_frames.append(img)
        result = torch.cat((torch.stack(all_frames,dim=0), torch.stack(all_frames_gt,dim=0)), dim=1) if self.no_color else torch.stack(all_frames,dim=0)
        return result

"""
Pytorch dataset class for loading pre-generated images from the Floating Balls dataset
"""
class FloatBallsVideoDataset(Dataset):

    def __init__(self,datapath,data_type='train',max_num_samples=60000,crop_sz=256,down_sz=64, max_num_frames=10, normalize = False, no_color = True, gt_datapath = ""):
        suffix = data_type
        self.datapath = datapath + suffix + '/'
        self.gt_datapath = gt_datapath + suffix + '/'
        self.max_num_samples = max_num_samples
        self.max_num_frames = max_num_frames
        self.normalize = normalize
        self.no_color = no_color
        self.folders = [ f.path for f in os.scandir(self.datapath) if f.is_dir() ]
        if len(self.folders) < self.max_num_samples:
            self.max_num_samples = len(self.folders)

    def __len__(self):
        return self.max_num_samples

    def __getitem__(self,idx):
        folder = idx
        all_frames = []
        all_frames_gt = []        
        for frame_idx in range(self.max_num_frames):
            imgname = self.datapath + str(folder) + "/" + str(frame_idx)
            imgpath = imgname + '.png'
            img = np.array(io.imread(imgpath))
            transform_list = [transforms.ToTensor()]
            if self.normalize:
                if self.no_color:
                    transform_list.append(transforms.Normalize((0.5), (0.5)))
                else: 
                    transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
            img = transforms.Compose(transform_list)(img) 
            if self.no_color:
                img = img.repeat(3,1,1)
            if self.no_color or not (self.gt_datapath == self.datapath):
                imgname_gt = self.gt_datapath + str(folder) + "/" + str(frame_idx)
                imgpath_gt = imgname_gt + '.png'
                img_gt = np.array(io.imread(imgpath_gt))
                img_gt = transforms.Compose(transform_list)(img_gt) 
                all_frames_gt.append(img_gt)
            all_frames.append(img)
        result = torch.cat((torch.stack(all_frames,dim=0), torch.stack(all_frames_gt,dim=0)), dim=1) if (self.no_color or not self.gt_datapath == self.datapath) else torch.stack(all_frames,dim=0)
        return result



if __name__=='__main__':
    main()
