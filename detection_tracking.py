import os
from timeit import time
import cv2
import numpy as np
import sys
import torch
import collections
import traceback
import matplotlib
import math
import pickle



matplotlib.use('Agg')
import matplotlib.pyplot as plt
import logging

sys.path.insert(0, 'yolov5')
sys.path.insert(0, './yolov5/models')

# set cuda devices
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# sys.path.append('yolov5')
# Human Pose
from pytorch_Realtime_Multi_Person_Pose_Estimation_master import huaat_pose_estimation
# Human Detection & Tracking
from deep_sort_pytorch.tracking_pipeline import TrackingModel
from yolov5.models.experimental import attempt_load
from yolov5.utils.datasets import letterbox
from yolov5.utils.general import non_max_suppression, xyxy2xywh, scale_coords
# Face Detection
from retinaface_face_detection.face_pipeline import load_fd_model_params
from retinaface_face_detection.main_face_detect import inference
# Plot Video
from moviepy.editor import VideoFileClip


# set video path
vid_folder = '/workspace/JinganXinfang_Project/huaat-detection-tracking-pose-emotion/CAERS/CAER/'

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
        # self.yolo_model = attempt_load('yolov5/yolov5x.pt', map_location='cuda')
        self.yolo_model = attempt_load('/workspace/JinganXinfang_Project/' \
                                       'huaat-detection-tracking-pose-emotion/face_huaat_deep_sort_yolov5_new_format_wxmlog/yolov5/yolov5x.pt',
                                       map_location='cuda')
        self.yolo_model(torch.zeros(1, 3, 640, 640).cuda().type_as(next(self.yolo_model.parameters())))
        ## tracking model
        self.tracking_model = TrackingModel()
        # ############### Face Detection Model Initiation
        # self.net, self.device, self.cfg, self.prior_data = load_fd_model_params(400, 536)

        # num of frames to go through  跳帧数量
        self.frame_hd_sep = 1
        self.frame_fd_sep = 1

    def pipeline(self, video_path): # video_path: /test/Disgust/0001.avi

        print('########## Processing Video {} ######### '.format(video_path))

        self.global_vid_path = vid_folder + video_path ## global video path: /workspace/.../test/Disgust/0001.avi
        self.vid_path = video_path # local video path: /test/Digust/0001.avi

        ########## Human Detection, Pose, & Tracking ################
        try:
            self.sv_data, self.hd_data, poseTrack_info_dict = self.detection_and_pose({},
                                                              self.global_vid_path, self.frame_hd_sep)
        except Exception as e:
            print(f'Error in Human Detection: {e}')
            print(traceback.format_exc())
        #############################################################

        # If model detects no people inside the video
        if len(self.sv_data) == 0:
            print('### No People Found in video {}, so this video is removed!!!'.format(video_path.split('/')[-1]))
            return 0, None, None
        else:  # if there is people inside the video
            print('### Number of people in video: {} !!!'.format(len(self.sv_data)))
            ########## Face Detection ##################################
            self.fd_data = self.face_detection(self.global_vid_path, self.frame_fd_sep)
            #################################################################

            ###################### SAVE VIDEO TO OUTPUT FOR DEMO
            # 1. draw box & emotion --> save video to ./output/output.avi
            # drawing_video_path = draw_box_emotion(video_path, frame_hd_sep, frame_fd_sep, sv_data, fd_data, hd_data)
            # drawing_video_path = 'temp.mp4'
            # new_draw_video_path = drawing_video_path.replace('.mp4', '_new.mp4')
            #new_draw_video_path = '/workspace/JinganXinfang_Project/huaat-detection-tracking-pose-emotion/CAERS/code/output/final.mp4'
            # convert video encoding to X264
            #os.system(f"ffmpeg -i {drawing_video_path} -vcodec libx264 {new_draw_video_path}")
            # remove original video file
            #os.remove(drawing_video_path)

            ### Save File into pickle


    def detection_and_pose(self, sv_data, video_path, frame_sep=1):
        ########## Human Detection (人检测） + Human Tracking (人跟踪） + Human Pose （姿态估计）
        ### Read in Video
        video_capture = cv2.VideoCapture(video_path)
        total_frame = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(video_capture.get(3))
        h = int(video_capture.get(4))
        # assert w == 536, "Width of video does not match image size in prior (536)."
        # assert h == 400, "Height of video does not match image size in prior (400)."
        print('video total frame number is: {}'.format(total_frame))

        if total_frame == 0:
            print('No Frame in Video')
            sys.exit()

        ##### START HUMAN DETECTION
        # set empty dictionaries that need to be saved later
        print('# Start Human Detection, Tracking, and Emotion Estimation')
        ### Reading Frames from Video
        frame_num = 0
        hd_data = {}
        detect_track_time_start = time.time()
        poseTrack_info_dict = {}

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
                    # logger.info('Human detection model', model_name='YOLOV5', prod_name='静安信访')
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
                    if frame_num >= 1:
                        ############## 3. Human Pose Estimation Model ################
                        frame, humans = huaat_pose_estimation.poseEstimation(frame)
                        # logger.info('Human pose estimation model', model_name='OpenPose', prod_name='静安信访')
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

        # # ### Save SV_data
        # with open('/workspace/JinganXinfang_Project/huaat-detection-tracking-pose-emotion/CAERS/code/output/hd','wb') as f:
        #     pickle.dump(sv_data,f)

        return sv_data, hd_data, poseTrack_info_dict

    ####### Face Detection
    def face_detection(self, video_path, frame_sep):

        ### Read in Video
        print('# Start Face Detection')
        video_capture = cv2.VideoCapture(video_path)
        total_frame = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(video_capture.get(3))
        h = int(video_capture.get(4))
        self.net, self.device, self.cfg, self.prior_data = load_fd_model_params(h, w)
        # assert w == 536, "Width of video does not match image size in prior (536)."
        # assert h == 400, "Height of video does not match image size in prior (400)."

        # Set Variable
        face_detection_time = time.time()
        frame_index = 0
        fd_dict = {}
        # Read Video Frame
        while True:
            ret, frame = video_capture.read()  # frame shape 640*480*3
            if ret != True:
                break
            # frame_sep: 跳帧
            if frame_index % frame_sep == 0:
                face_coord_dict = {}
                # convert frame to float32 format
                img = np.float32(frame)
                ##!!!! MAIN: Face Detection Inference
                # output shape: [x,15] where x is the number of faces
                face_output = inference(img, self.net, self.device, self.cfg, self.prior_data)
                # logger.info('Human face detection model', model_name='RetinaFace', prod_name='静安信访')
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
                fd_dict[frame_index] = face_coord_dict
            frame_index += 1
        print('#### Face Detection Time: {} seconds \n'.format(time.time() - face_detection_time))
        # release video capture
        video_capture.release()

        ### Save Face Detection file into pickle
        # with open('/workspace/JinganXinfang_Project/huaat-detection-tracking-pose-emotion/CAERS/code/output/fd',
        #           'wb') as f:
        #     pickle.dump(fd_dict, f)
        return fd_dict

    def hd_fd_result(self):
        ### ----------------- Save Video Human Detection & Face Detection

        # Total Frame: 50
        ## self.sv_data (Hd+Tracking) --> {1: {'box': array([[0,0,0,0],[0,0,0,0],...]], 2: {[0,0,0,0]})
        ## self.fd_data(Face Detection) -->  {1: {0: [210, 65, 369, 289],1:[100,100,200,200}, 2: {0: [210, 63, 369, 288]},





        ### 1. Human Detection + Tracking Data
        new_sv_data = {}
        for key in self.sv_data:  # key: [1,2,3]
            new_sv_data.update({key: self.sv_data[key]['box']})


        ## Final Dictionary Format:
        ## {'Disgust/0001.avi' : {'human': {...}, 'face'': {...}}, 'yyy.avi': {...}}
        result = {self.vid_path: {'human':new_sv_data,
                                  'face':self.fd_data}}


        return result


    def draw_box_emotion(self,\
        output_video_name='/workspace/JinganXinfang_Project/huaat-detection-tracking-pose-emotion/CAERS/code/output/1008.mp4'):
        # 对人和人脸进行画框，最后保存成视频
        # read video configuration
        cap = cv2.VideoCapture(self.global_vid_path)
        fps_ori = cap.get(cv2.CAP_PROP_FPS)
        total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # save video wirh only half of width and half of length pixels
        w = int(cap.get(3) * 0.5)
        h = int(cap.get(4) * 0.5)

        # Define the codec and create VideoWriter object

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
            frame_next = math.ceil(frame_num / self.frame_hd_sep) * self.frame_hd_sep
            current_frame = math.floor(frame_num / self.frame_fd_sep) * self.frame_fd_sep
            # if frame greater than total frame, end while loop
            if frame_next > total_frame:
                break

            # #!# 1. Plot Human Tracking Detection box in frame
            # if len(self.sv_data) > 0:
            #     for key in self.sv_data:
            #         # 1. Draw Human Detection Box in Frame
            #         bbox = self.sv_data[key]['box']



            # !# 1. plot human detection frame
            if frame_next < total_frame and frame_next in self.hd_data:
                hdbox = self.hd_data[frame_next]
                if len(hdbox) > 0:
                    for p_cord in hdbox:
                        cv2.rectangle(frame, (int(p_cord[0]), int(p_cord[1])), \
                                      (int(p_cord[2]), int(p_cord[3])), [0, 255, 0], 2)

            # !# 2. Draw Face Detection Box in Frame
            if current_frame > 0 and current_frame < total_frame:  # ignore frame 0 and last few frames
                fbox = self.fd_data[current_frame]
                if len(fbox) > 0:
                    for pid in fbox:
                        cv2.rectangle(frame, (int(fbox[pid][0]), int(fbox[pid][1])), \
                                      (int(fbox[pid][2]), int(fbox[pid][3])), [0, 255, 255], 2)

            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
            out.write(frame)
            frame_num += 1

        # release video
        out.release()

        print('### Video has been Saved!!! #### ')
        print('Time to save video:', time.time() - start_time)





## if __name__ == __main__:
# ### 1. 读取视频+视频基本信息Read Video & Video Description
# video_name = '/workspace/JinganXinfang_Project/huaat-detection-tracking-pose-emotion/CAERS/Disgust/0035.avi'
# vc = cv2.VideoCapture(video_name)
# # Total Frame
# length = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
#
# # FPS
# fps = vc.get(cv2.CAP_PROP_FPS)
#
# ret, frame = vc.read()  # (height 400, width 536, 3)
# print('Frame shape:', frame.shape)


if __name__ == "__main__":

    # Pick Option HERE
    option = 1

    #### OPTION 1: A list of some files
    if option == 1:
        final = {}
        video_name = ['/test/Neutral/0176.avi']

        # Create Model Pipeline
        mp = ModelPipeline()
        for vid in video_name:

            # Input video
            mp.pipeline(vid)
            # Save pickle file
            result = mp.hd_fd_result()
            # add into final
            final.update(result)
        print(final)


    # #### OPTION 2: Go through all videos
    elif option == 2:
        which_set = 'validation'
        # emo = 'Disgust'
        # Create Object mp pipeline
        mp = ModelPipeline()
        final = {}

        iii = 0
        for tvt in os.listdir(vid_folder):
            if tvt == which_set:
                cat_path = os.path.join(vid_folder,tvt) # cat_path: /workspace/.../train

                for emo_cat in os.listdir(cat_path): # emo_cat:  Disgust,Anger,..
                    emo_path = os.path.join(tvt,emo_cat) # emo_path: train/Disgust

                    for vid in os.listdir(os.path.join(vid_folder,emo_path)): # vid: 0001.avi
                        print('---------------------------------------\n')
                        print('-- Sample {} -- '.format(iii))
                        video_name = os.path.join(emo_path,vid) # videoname: train/Disgust/0001.avi

                        # Obtain Human Detection + Face Detection Result
                        mp.pipeline(video_name)
                        result = mp.hd_fd_result()
                        final.update(result)


                        iii+=1


        ## Save pickle file
        with open(f'/workspace/JinganXinfang_Project/huaat-detection-tracking-pose-emotion/CAERS/code/'
                  f'output/{which_set}','wb') as f:
            pickle.dump(final,f)

    else: # other options
        print('Please specify an option. Either 1 or 2.')
    '''
    ### pickle 字典输出格式
    {'/test/Disgust/0001.avi' : {'human': {person1:[[0,0,1,1],...],
                                           person2:[[...]]
                                          }, 
                                 'face': {frame1:{person1: [a,b,c,d],person2:[e,f,g,h],
                                          frame2:{...}
                                          }
                                },
     '/test/Disgust/0002.avi': {...},
     ...
        
    }
    
    '''

    # # save video file
    # mp.draw_box_emotion(output_video_name ='/workspace/JinganXinfang_Project/huaat-detection-tracking-pose-emotion/' \
    #                                         'CAERS/code/output/1008.mp4')
