# Copyright (c) 2019-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import imageio
import os
from os import listdir
from os.path import isfile, join 
from PIL import Image
import PIL   

os.mkdir("clevrer")
os.mkdir("clevrer/train")
mypath = 'video_train/'

outerfolders = [f for f in listdir(mypath)]

for folder_1 in outerfolders:
	new_path = mypath  + folder_1 + "/"
	files = [f for f in listdir(new_path) if isfile(join(new_path, f))]
	for file in files:
	    print(file)
	    path = new_path+file
	    vid = imageio.get_reader(path,  'ffmpeg')
	    new_new_path = "clevrer/train/"+file.split(".")[0].split("_")[1]+"/"
	    os.mkdir(new_new_path)
	    for i, im in enumerate(vid):
	        im = Image.fromarray(im)
	        im = im.save(new_new_path+"/"+str(i)+".png") 
