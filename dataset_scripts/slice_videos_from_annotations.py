# Copyright (c) 2019-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import json, glob, os, sys
from os.path import join
import subprocess
import cv2

ANNOTATIONS_DIR = 'annotation_validation'
VIDEO_DIR = 'video_validation'

OUTPUT_CONFIG_DIR = 'sliced_annotations_validation'
OUTPUT_VIDEO_DIR = 'sliced_video_validation'

if not os.path.exists(OUTPUT_CONFIG_DIR):
    os.makedirs(OUTPUT_CONFIG_DIR)

if not os.path.exists(OUTPUT_VIDEO_DIR):
    os.makedirs(OUTPUT_VIDEO_DIR)

MIN_FRAMES_ALL_OBJECTS_VISIBLE = 40

annotation_files = []
for annotation_dir in glob.glob(join(ANNOTATIONS_DIR, '**/')):
	for ann in glob.glob(join(annotation_dir, 'annotation_*.json')):
		annotation_files.append(ann)

annotations_by_nobjects = {}
total_visible_frames = 0
n_annotations = 0
n_annotations_above_1s = 0
n_annotations_above_2s = 0
n_annotations_above_3s = 0
n_annotations_above_4s = 0
n_frames_total = 0
for ann_idx, ann_file in enumerate(annotation_files):
	with open(ann_file) as f:
		ann = json.load(f)

	try:
		n_objects = len(ann['object_property'])
	except TypeError as e:
		print('TypeError for annotation file %s : %s'%(ann_file, str(e)))
		continue
	except KeyError as e:
		print('KeyError for annotation file %s : %s'%(ann_file, str(e)))
		continue

	n_frames = len(ann['motion_trajectory'])
	#print(f'Number of frames {n_frames}')

	# Which one is the first frame in which all the objects are visible
	frame_begin = -1
	# Which one is the last frame in which all the objects are visible
	frame_end = -1

	for frame in ann['motion_trajectory']:
		all_visible = True
		for o in frame['objects']:
			all_visible = all_visible and o['inside_camera_view']

		if all_visible and frame_begin == -1:
			frame_begin = frame['frame_id']

		if not all_visible and frame_end == -1 and frame_begin != -1:
			frame_end = frame['frame_id']

		if frame_begin != -1 and frame_end != -1:
			break

	if frame_begin != -1 and frame_end == -1:
		frame_end = n_frames - 1

	if frame_begin == -1:
		continue

	n_frames_visible = frame_end - frame_begin

	total_visible_frames += n_frames_visible
	n_annotations += 1
	n_frames_total += n_frames

	if n_frames_visible >= 25:
		n_annotations_above_1s += 1

	if n_frames_visible >= 50:
		n_annotations_above_2s += 1

	if n_frames_visible >= 75:
		n_annotations_above_3s += 1

	if n_frames_visible >= 100:
		n_annotations_above_4s += 1

	if n_frames_visible < MIN_FRAMES_ALL_OBJECTS_VISIBLE:
		print('Skipping %s: too short!'%(ann_file))
		continue

	# Splitting the annotations json into scene configuration and motion
	base_dir = os.path.dirname(ann_file).replace(ANNOTATIONS_DIR, OUTPUT_CONFIG_DIR + '/' + str(n_objects))
	if not os.path.exists(base_dir):
		os.makedirs(base_dir)

	annotation_id = os.path.splitext(os.path.basename(ann_file))[0].split('_')[1]
	motion_json_file = join(base_dir, 'motion_%s.json'%(annotation_id))#.replace('/', '\\')
	config_json_file = join(base_dir, 'config_%s.json'%(annotation_id))#.replace('/', '\\')
	print('Annotation %s -> %s, %s'%(ann_file, motion_json_file, config_json_file))

	# The configuration file contains a list of initial shapes, materials and physical parameters
	# print(config_json_file)
	# print(motion_json_file)

	config_json = []
	for i, o in enumerate(ann['object_property']):
		o_json = {}
		o_json['shape'] = o['shape']
		o_json['color'] = o['color']
		o_json['material'] = o['material']
		o_json['mass'] = 1 # Wrong
		o_json['scale'] = 0.2 # Is it really the same for all the motions?
		o_json['init_pos'] = ann['motion_trajectory'][frame_begin]['objects'][i]['location']
		o_json['init_orn'] = ann['motion_trajectory'][frame_begin]['objects'][i]['orientation']
		o_json['init_v'] = ann['motion_trajectory'][frame_begin]['objects'][i]['velocity']
		config_json.append(o_json)

	motion_json = {
		'timestep' : 0.01,
		'motion': []
	}
	for frame in range(frame_begin, frame_end):
		frame_json = []
		for o in range(n_objects):
			object_json = {}
			object_json['location'] = ann['motion_trajectory'][frame]['objects'][o]['location']
			object_json['orientation'] = ann['motion_trajectory'][frame]['objects'][o]['orientation']
			object_json['velocity'] = ann['motion_trajectory'][frame]['objects'][o]['velocity']
			object_json['angular_velocity'] = ann['motion_trajectory'][frame]['objects'][o]['angular_velocity']
			frame_json.append(object_json)
		#print(frame)
		motion_json['motion'].append(frame_json)


	with open(config_json_file, 'w') as f:
		json.dump(config_json, f)

	with open(motion_json_file, 'w') as f:
		json.dump(motion_json, f)

	# Export the frame within the being/end slice
	video_filename = ann['video_filename']

	video_file = glob.glob(join(VIDEO_DIR, '**/' + video_filename))[0]
	video_output_dir = join(join(OUTPUT_VIDEO_DIR, str(n_objects)), os.path.splitext(video_filename)[0])
	if not os.path.exists(video_output_dir):
		os.makedirs(video_output_dir)

	cap = cv2.VideoCapture(video_file)
	success, frame_image = cap.read()
	frame = 1
	while success:
		frame_file = join(video_output_dir, '%d.png'%(frame - frame_begin))
		success, frame_image = cap.read()
		if frame >= frame_begin and frame < frame_end:
			cv2.imwrite(frame_file, frame_image)
		frame += 1

	print('Generated sliced video for %d/%d'%(ann_idx, len(annotation_files)))



#================================

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software 
# and associated documentation files (the "Software"), to deal in the Software without restriction, 
# including without limitation the rights to use, copy, modify, merge, publish, distribute, 
# sublicense, and/or sell copies of the Software, and to permit persons to whom the Software 
# is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies 
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A 
# PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT 
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION 
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH 
# THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#===========================
