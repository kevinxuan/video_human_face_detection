import os
import cv2
import numpy as np
import pandas as pd

'''
Author: Kevin Xuan
Date: 09/26/2022

FUNCTION:
This file reads in CAER-S video, read frames from the videos. The objective is to generate training & Testing data

'''
# Set Directory
os.chdir('/data/kevin/paper1_swinTr_tt/CAER_S/CAER')

### 1. 读取视频+视频基本信息Read Video & Video Description
vc = cv2.VideoCapture('test/Disgust/0100.avi')
# Total Frame
length = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))

print(length)
# FPS
fps = vc.get(cv2.CAP_PROP_FPS)

print(fps)


ret, frame = vc.read()  # frame shape 640*480*3

print('Frame shape:',frame.shape)



# '''
# 一个视频差不多3秒，标注为人的心情
# '''
#
# ### 2. 视频处理
# path = 'train/'
# # emo_path = 'test/Angry'
# final = []
#
# # Import Object Detection
# det = detection()
#
# ## 进入所有文件夹读取视频
# i = 0
# for emo in os.listdir(path): # emo: Anger
#     emo_path = os.path.join(path,emo)
#     for vid in os.listdir(emo_path): # video: 0001.avi
#         vid_path = os.path.join(emo_path,vid) # img_path: test/Anger/0001.avi
#
#         # Read Video & Video Frame
#         video = cv2.VideoCapture(vid_path)
#
#         while True:
#             ret, frame = video_capture.read()  # image shape: (400, 536, 3)
#             if ret != True:
#                 break
#             # frame_sep: 跳帧
#             if frame_index % frame_sep == 0 and frame_index >= 1:
#                 hd_coord = det.human_detection_coord(img_arr)
