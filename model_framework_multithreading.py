import os
from timeit import time
import cv2
import numpy as np
import sys
sys.path.insert(0, './yolov5')
import torch
import collections
import traceback
# Human Detection & Tracking
from deep_sort_pytorch.tracking_pipeline import TrackingModel
from yolov5.models.experimental import attempt_load
from yolov5.utils.datasets import letterbox
from yolov5.utils.general import non_max_suppression, xyxy2xywh, scale_coords
# Human Pose
from pytorch_Realtime_Multi_Person_Pose_Estimation_master import huaat_pose_estimation
# Face Detection
from retinaface_face_detection.face_pipeline import load_fd_model_params
from retinaface_face_detection.main_face_detect import inference
from emotion_score import pose_to_emotion
# Plot Video
from draw_box_and_emotion import draw_box_emotion
from moviepy.editor import VideoFileClip
# Threading Related Module
import inspect
import ctypes
import threading

# Emotion Score
os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"

########################  关闭线程
def _async_raise(tid, exctype):
	"""raises the exception, performs cleanup if needed"""
	tid = ctypes.c_long(tid)
	if not inspect.isclass(exctype):
		exctype = type(exctype)
	res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(exctype))
	if res == 0:
		raise ValueError("invalid thread id")
	elif res != 1:
		# """if it returns a number greater than one, you're in trouble,
		# and you should call it again with exc=NULL to revert the effect"""
		ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
		raise SystemError("PyThreadState_SetAsyncExc failed")

def stop_thread(thread):
	_async_raise(thread.ident, SystemExit)

########################


def ifPointInBbox(point_coord, bbox):
	if_point_in_bbox = False
	if point_coord[0] >= bbox[0] and point_coord[0] <= bbox[2] and point_coord[1] >= bbox[1] and point_coord[1] <= bbox[
		3]:
		if_point_in_bbox = True
	return if_point_in_bbox


def getHumanID(body_parts_new, bbox_updated):
	count_index = np.zeros((1, len(bbox_updated)))
	count_index = list(count_index.flatten())
	for key, value in body_parts_new.items():
		if value != 'null':
			for index, bbx in enumerate(bbox_updated):
				bbox = bbx[1]
				if ifPointInBbox(value, bbox):
					count_index[index] += 1
	if max(count_index) >= 3:
		hit_index = count_index.index(max(count_index))
		bbox_updated[hit_index] = (bbox_updated[hit_index][0], bbox_updated[hit_index][1], body_parts_new)
	return bbox_updated


def match_poseAndTrack_2(humans, bbox_updated, frame):
	image_h, image_w = frame.shape[:2]  # original frame image size
	for human in humans:
		# body_parts_new = {}
		body_parts_new = dict(zip(list(range(18)), list(range(18))))
		for i in range(18):
			if i not in human.body_parts.keys():
				body_parts_new[i] = 'null'
				continue
			body_part = human.body_parts[i]
			key_point_coord = [int(body_part.x * image_w + 0.5), int(body_part.y * image_h + 0.5)]
			body_parts_new[i] = key_point_coord
		bbox_updated = getHumanID(body_parts_new, bbox_updated)
	return bbox_updated


class ModelPipeline(object):
	def __init__(self):
		self.COLORS = np.random.randint(0, 255, size=(200, 3), dtype="uint8")
		################ Human Detection Model Initiation
		# yolov5 model
		self.yolo_model = attempt_load('yolov5/yolov5x.pt', map_location='cuda')
		self.yolo_model(torch.zeros(1, 3, 640, 640).cuda().type_as(next(self.yolo_model.parameters())))
		## tracking model
		self.tracking_model = TrackingModel()
		############### Face Detection Model Initiation
		self.net, self.device, self.cfg, self.prior_data = load_fd_model_params(1080, 1920)

	def pipeline(self, video_path, camera_id, date):
		# num of frames to go through  跳帧数量
		frame_hd_sep = 6
		frame_fd_sep = 2
		sv_data = {}
		hd_data = {}
		fd_data = {}
		poseTrack_info_dict = {}
		########## Human Detection, Pose, & Tracking  + Face Detection ################
		try:
			### 1. Human Detection
			t1 = threading.Thread(target=self.detection_and_pose,
								  args=(sv_data, hd_data, poseTrack_info_dict, video_path, frame_hd_sep))
			t1.start()

			### 2. Face Detection
			t2 = threading.Thread(target=self.face_detection, args=(fd_data, video_path, frame_fd_sep))
			t2.start()
			# 等待t1 （人检测+跟踪模型）程序跑完
			t1.join()
		except Exception as e:
			print(f'Error in Human or Face Detection: {e}')
			print(traceback.format_exc())
		#############################################################

		# If model detects no people inside the video
		if len(sv_data) == 0:
			# delete video
			os.remove(video_path)
			print('### No People Found in video {}, so this video is removed!!!'.format(video_path.split('/')[-1]))
			stop_thread(t2)
			return 0, None, None
		else:  # if there is people inside the video
			### 等待t2 （人脸模型） 程序跑完
			t2.join()
			print('### Number of people in video: {} !!!'.format(len(sv_data)))
			# Emotion Score Calculation 情绪值计算
			poseTrack_info_dict = collections.OrderedDict(sorted(poseTrack_info_dict.items()))
			emotion_value, image_list, voice_emotion = pose_to_emotion(poseTrack_info_dict, video_path, str(camera_id),
																	   date, 4)

			###################### SAVE VIDEO TO OUTPUT FOR DEMO
			# 1. draw box & emotion --> save video to ./output/output.avi
			drawing_video_path = draw_box_emotion(video_path, frame_hd_sep, frame_fd_sep, sv_data, fd_data, hd_data)
			new_draw_video_path = drawing_video_path.replace('.mp4', '_new.mp4')
			# convert video encoding to X264
			os.system(f"ffmpeg -i {drawing_video_path} -vcodec libx264 {new_draw_video_path}")
			# remove original video file
			os.remove(drawing_video_path)

			# 2. Add Audio into the generated video file 将新生成的视频填上原有的音频
			if voice_emotion != 'Null':
				################# Process audio
				video_for_audio = VideoFileClip(video_path)
				audio = video_for_audio.audio
				# Add audio into video
				video_out = VideoFileClip(new_draw_video_path)
				os.remove(new_draw_video_path)
				video_aud = video_out.set_audio(audio)
				video_aud.write_videofile(new_draw_video_path, audio_codec='aac')
				print('Added audio into the new video file')

			# delete the original saved video
			os.remove(video_path)

			return emotion_value, image_list, new_draw_video_path

	def detection_and_pose(self, sv_data, hd_data, poseTrack_info_dict, video_path, frame_sep=1):
		########## Human Detection (人检测） + Human Tracking (人跟踪） + Human Pose （姿态估计）
		### Read in Video
		video_capture = cv2.VideoCapture(video_path)
		total_frame = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
		w = int(video_capture.get(3))
		h = int(video_capture.get(4))
		video_fps = video_capture.get(cv2.CAP_PROP_FPS)
		assert video_fps == 25, "FPS of video {} is not 25.".format(video_path)
		assert w == 1920, "Width of video does not match image size in prior (1920)."
		assert h == 1080, "Height of video does not match image size in prior (1080)."
		print('video total frame number is: {}'.format(total_frame))

		##### START HUMAN DETECTION
		# set empty dictionaries that need to be saved later
		print('###Start Human Detection, Tracking, and Emotion Estimation')
		### Reading Frames from Video
		frame_num = 0
		detect_track_time_start = time.time()
		while True:
			ret, frame = video_capture.read()  # frame shape 640*480*3
			if ret != True:
				break

			if frame_num % frame_sep == 0:
				##---------------1. human detection model
				### resize frame
				frame_resize = letterbox(frame)[0]
				# Convert frame
				frame_resize = frame_resize[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
				frame_resize = np.ascontiguousarray(frame_resize)
				# Rescale frame to frame for inputting in Yolo model
				frame_resize = torch.from_numpy(frame_resize).cuda().float()
				frame_resize /= 255.0
				if frame_resize.ndimension() == 3: frame_resize = frame_resize.unsqueeze(0)

				###################### Human Detection (人检测 ） YoloV5
				try:
					pred_org = self.yolo_model(frame_resize)[0]
				except Exception as e:
					print(f'Error in Yolo, does it is illegal memory ? add by wxm: {e}')
					print(traceback.format_exc())

				pred = non_max_suppression(pred_org, classes=0)[0]

				# if there is people inside the frame, then tracking the person
				if pred.shape[0] > 0:
					##---------------2. human tracking model
					## rescale bbox
					pred[:, :4] = scale_coords(frame_resize.shape[2:], pred[:, :4], frame.shape).round()

					##### Put Human Detection Box inside dictionary hd_data
					# hd_data: {'0': [[a,b,c,d],[aa,bb,cc,dd]], '6':...]
					person_coord_list = [pred[i][0:4].tolist() for i in range(len(pred))]
					hd_data[frame_num] = person_coord_list

					# (left,top, width,height) --> (left,top,bottom,right)
					xywhs = xyxy2xywh(pred[:, 0:4])
					bboxes_updated = []
					sv_data, bboxes_updated = self.tracking_model.tracking(frame, pred, xywhs, sv_data, frame_num,
																		   total_frame, bboxes_updated)
					# print(frame_num, bboxes_updated, sv_data)
					if frame_num % 12 == 0 and frame_num >= 1:
						############## 3. Human Pose Estimation Model ################
						frame, humans = huaat_pose_estimation.poseEstimation(frame)
						### Human Detection & Pose Matching
						if len(bboxes_updated) != 0 and len(humans) != 0:
							bboxes_updated = match_poseAndTrack_2(humans, bboxes_updated, frame)
						# elif len(bboxes_updated) == 0 and len(humans) != 0:
						# 	bboxes_updated = [['null', ['null'], {'null': 'null'}]]
						# elif len(bboxes_updated) != 0 and len(humans) == 0:
						else:
							bboxes_updated = [['null', ['null'], {'null': 'null'}]]
						trackPose_perFrame_dict = {}
						for bbox_uppdated in bboxes_updated:
							track_id = bbox_uppdated[0]
							if len(bbox_uppdated) == 2:
								object_pose = {'null': 'null'}
							else:
								object_pose = bbox_uppdated[2]
							trackPose_perFrame_dict[str(track_id)] = object_pose
							trackPose_perFrame_dict = collections.OrderedDict(sorted(trackPose_perFrame_dict.items()))

						poseTrack_info_dict[frame_num] = trackPose_perFrame_dict

			frame_num += 1
			# Press Q to stop!
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
		print('#### Human Detection, Tracking and Pose Time: {} seconds'.format(time.time() - detect_track_time_start))
		video_capture.release()
		cv2.destroyAllWindows()

		return sv_data, hd_data, poseTrack_info_dict

	####### Face Detection
	def face_detection(self, fd_data, video_path, frame_sep):
		### Read in Video
		print('###Start Face Detection')
		video_capture = cv2.VideoCapture(video_path)
		total_frame = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
		w = int(video_capture.get(3))
		h = int(video_capture.get(4))
		video_fps = video_capture.get(cv2.CAP_PROP_FPS)
		assert video_fps == 25, "FPS of video {} is not 25.".format(video_path)
		assert w == 1920, "Width of video does not match image size in prior (1920)."
		assert h == 1080, "Height of video does not match image size in prior (1080)."

		# Set Variable
		face_detection_time = time.time()
		frame_index = 0
		# Read Video Frame
		while True:
			ret, frame = video_capture.read()  # frame shape 640*480*3
			if ret != True:
				break
			# frame_sep: 跳帧
			if frame_index % frame_sep == 0 and frame_index >= 1:
				face_coord_dict = {}
				# convert frame to float32 format
				img = np.float32(frame)
				##!!!! MAIN: Face Detection Inference
				# output shape: [x,15] where x is the number of faces
				face_output = inference(img, self.net, self.device, self.cfg, self.prior_data)
				# face detection time inference time: average 50ms per second

				# if faces are detected
				if len(face_output) > 0:
					# filter out faces with area fewer than 2000 [smaller than 2000 indicates that the face is too small]
					if face_output.shape[0] > 0:
						area = abs((face_output[:, 1] - face_output[:, 3]) * (face_output[:, 0] - face_output[:, 2]))
					# final face coordinates of each person in format: left, top, right, bottom
					face_output = face_output[np.where(area > 2000)]
					for id, hf in enumerate(face_output):
						# append dictionary
						face_coord_dict[id] = [int(hf[0]), int(hf[1]), int(hf[2]), int(hf[3])]
				# face_coord_dict output format: {0: [861, 125, 961, 262], 1:[100,342, 150, 400]}
				fd_data[frame_index] = face_coord_dict
			frame_index += 1
		print('######### Face Detection Time: {} seconds \n'.format(time.time() - face_detection_time))
		# release video capture
		video_capture.release()

		return fd_data
