# Copyright (c) 2019-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
import torch
import h5py


f = h5py.File('balls678mass64.h5', 'r')

out_list = ['features']
out_list.append('groups') 
out_list.append('collisions') 

data_in_file_testing =  {
            data_name: f['test'][data_name] for data_name in out_list}
      
random_colors = [(166,206,227),(31,120,180),(178,223,138),(51,160,44),(166,206,227),(31,120,180),(178,223,138),(51,160,44)]

os.mkdir("bb_color678_4_colors/")
os.mkdir("bb_color678_4_colors/test")

b = np.array(data_in_file_testing["groups"])

        
n_frames, n_samples, _, _, _= np.shape(b)
columns = 1
rows = 1
for i in range(n_samples):
    os.mkdir("bb_color678_4_colors/test/"+ str(i))
    for j in range(n_frames):
        img = np.array(b[j][i])
        img = np.repeat(img, 3, axis = 2)
        new_img = np.zeros(img.shape)
        uni = np.unique(img[np.nonzero(img)])
        for l in range(len(uni)):
            for c in range(3):
                new_img[:,:,c][np.where(img==uni[l])[0:2]] = random_colors[l][c]
        
        new_img = new_img.astype(np.uint8)
        cv2.imwrite('bb_color678_4_colors/test/'+ str(i)+'/'+str(j) + '.png',new_img)


