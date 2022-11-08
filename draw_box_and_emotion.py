# -*- coding: utf-8 -*-

import cv2
import os
import numpy as np
import statistics
import numpy as np
import json
import copy
import time
import math
from moviepy.editor import VideoFileClip

"""
Date: 09/06/2021

画图：
- 人框 （头上不带情绪值）
- 人脸框
- 保存视频： 为了加快视频保存时间，将每帧图片的长和宽除了2，再进行保存。

"""



def draw_box_emotion(video_path, frame_hd_sep,frame_fd_sep, sv_data,fd_data,hd_data):
	# read video configuration
	cap = cv2.VideoCapture(video_path)
	fps_ori = cap.get(cv2.CAP_PROP_FPS)
	total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	# save video wirh only half of width and half of length pixels
	w = int(cap.get(3)*0.5)
	h = int(cap.get(4)*0.5)


	# Define the codec and create VideoWriter object
	# Save Path for Video
	new_video_path = '/'.join(video_path.split('/')[:-2])
	output_video_name = os.path.join(new_video_path +'/output_video/', video_path.split('/')[-1])
	out = cv2.VideoWriter(output_video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps_ori, (w, h))

	start_time = time.time()
	key_access = set()
	frame_num = 0
	hd_current_frame = 0

	while True:
		ret, frame = cap.read()
		if ret != True:
			break

		# 每x帧画一次图
		frame_next = math.ceil(frame_num / frame_hd_sep) * frame_hd_sep
		current_frame = math.floor(frame_num / frame_fd_sep) * frame_fd_sep
		# if frame greater than total frame, end while loop
		if frame_next > total_frame:
			break

		# #!# 1. Plot Human Tracking Detection box in frame
		# if len(sv_data) > 0:
		# 	for key in sv_data:
		# 		# 1. Draw Human Detection Box in Frame
		# 		bbox = sv_data[key]['box'][frame_next]
		# 		cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), [0, 255, 0], 2)

		#!# 1. plot human detection frame
		if frame_next < total_frame and frame_next in hd_data:
			hdbox = hd_data[frame_next]
			if len(hdbox) > 0:
				for p_cord in hdbox:
					cv2.rectangle(frame, (int(p_cord[0]), int(p_cord[1])), \
										 (int(p_cord[2]), int(p_cord[3])), [0, 255, 0], 2)

		#!# 2. Draw Face Detection Box in Frame
		if current_frame > 0 and current_frame < total_frame: # ignore frame 0 and last few frames
			fbox = fd_data[current_frame]
			if len(fbox) > 0:
				for pid in fbox:
					cv2.rectangle(frame, (int(fbox[pid][0]), int(fbox[pid][1])), \
										 (int(fbox[pid][2]), int(fbox[pid][3])), [0, 255, 255], 2)

		frame = cv2.resize(frame, (w,h),interpolation = cv2.INTER_AREA)
		out.write(frame)
		frame_num += 1

	# release video
	out.release()

	print('### Video has been Saved!!! #### ')
	print('Time to save video:', time.time() - start_time)

	return output_video_name
