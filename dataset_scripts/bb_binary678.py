# Copyright (c) 2019-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import h5py
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import os
import cv2


f = h5py.File('balls678mass64.h5', 'r')

out_list = ['features']
out_list.append('groups') 
out_list.append('collisions') 


data_in_file = {
            data_name: f['test'][data_name] for data_name in out_list
}

a = np.array(data_in_file["features"])
os.mkdir("bb_binary678")
os.mkdir("bb_binary678/test")
n_frames, n_samples, _, _, _= np.shape(a)
columns = 1
rows = 1
for i in range(n_samples):
    os.mkdir("bb_binary678/test/"+ str(i))
    for j in range(n_frames):
        new_img = np.array(a[j][i])
        new_img = new_img.astype(np.uint8)
        new_img*=255
        cv2.imwrite('bb_binary678/test/'+ str(i)+'/'+str(j) + '.png', new_img)
