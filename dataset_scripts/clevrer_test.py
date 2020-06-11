# Copyright (c) 2019-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import imageio
import os
os.mkdir("clevrer345")
os.mkdir("clevrer6")
os.mkdir("clevrer345/test")
os.mkdir("clevrer6/test")
from os import listdir
from os.path import isfile, join 
from PIL import Image
import PIL   
mypath = 'sliced_videos_validation_mp4/'

objects_num = [3,4,5,6]

for num_obj in objects_num:
	new_path = mypath  + str(num_obj) + "/"
	outerfolders = [f for f in listdir(new_path)]
	for folder_1 in outerfolders:
		path = new_path + folder_1 + "/"+"video.mp4"
		vid = imageio.get_reader(path,  'ffmpeg')
		if num_obj == 6:
			new_new_path = "clevrer6/test/"+folder_1.split("_")[1]+"/"
		else:
			new_new_path = "clevrer345/test/"+folder_1.split("_")[1]+"/"
		os.mkdir(new_new_path)
		for i, im in enumerate(vid):
			im = Image.fromarray(im)
			im = im.save(new_new_path+"/"+str(i)+".png") 
