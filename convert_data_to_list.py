import argparse
import cv2
import numpy as np
import os
import pickle
import pandas as pd
import torch
# Add path of yolov5
import sys
# sys.path.append('/data/wangxiaomei/codes/research/paper1/paper1_swinTr/emotic-master/kevin/yolov5')
sys.path.append('/data/kevin/paper1_swinTr_tt/CAER_S/code/yolov5')

# Human Detection Model Packages
from yolov5.models.experimental import attempt_load
from yolov5.utils.datasets import letterbox
from yolov5.utils.general import non_max_suppression, xyxy2xywh, scale_coords

# Face Detection
sys.path.append('/data/kevin/paper1_swinTr_tt/CAER_S/code/retinaface_face_detection')
from retinaface_face_detection.face_pipeline import load_fd_model_params
from retinaface_face_detection.main_face_detect import inference
# ------------------------------------------------------------------------------------------

## image -> yolo-> body bbox
## iamge -> face detection -> face bbox
## List=[[category='angry', filenaem='ninjijnijidnd.png', folder='test/anger', body_bbox=[leftup_rightbottom], face_bbox=[]]]


################# 1. 人识别
class detection(object):
    def __init__(self):
        # Load yolov5 model
        self.yolo_model = attempt_load('/data/wangxiaomei/codes/research/paper1/paper1_swinTr/emotic-master/kevin/yolov5/yolov5x.pt', map_location='cuda')
        self.yolo_model(torch.zeros(1, 3, 640, 640).cuda().type_as(next(self.yolo_model.parameters())))

        # face detection model
        # self.net, self.device, self.cfg, self.prior_data = load_fd_model_params(400,712)

    # Read Image
    def human_detection_coord(self,image):

        ##---------------1. human detection model
        ### resize frame
        frame_resize = letterbox(image)[0]
        # Convert frame
        frame_resize = frame_resize[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        frame_resize = np.ascontiguousarray(frame_resize)
        # Rescale frame to frame for inputting in Yolo model
        frame_resize = torch.from_numpy(frame_resize).cuda().float()
        frame_resize /= 255.0
        if frame_resize.ndimension() == 3: frame_resize = frame_resize.unsqueeze(0)

        ### Human Detection Predict Coordinate
        pred_org = self.yolo_model(frame_resize)[0]
        pred = non_max_suppression(pred_org, classes=0)[0]  #([[411.34119,  13.16081, 590.19312, 149.35725,   0.89687,   0.00000],

        ### Final Answer Coordinate
        # image[top:bottom,left:right]
        hdc = [i[0:4].cpu().numpy().tolist() for i in pred]
        hdc_round = [[round(j) for j in i] for i in hdc]  # left top right bottom

        return hdc_round

    ################# 2. Based on Image, Get Face Bbox
    def face_detection_coord(self,image):

        self.net, self.device, self.cfg, self.prior_data = load_fd_model_params(image.shape[0], image.shape[1])

        image = np.float32(image)

        face_output = inference(image, self.net, self.device, self.cfg, self.prior_data)

        face_dict = []
        if len(face_output) > 0:
            # filter out faces with area fewer than 2000 [smaller than 2000 indicates that the face is too small]
            if face_output.shape[0] > 0:
                    area = abs((face_output[:, 1] - face_output[:, 3]) * (face_output[:, 0] - face_output[:, 2]))
            # final face coordinates of each person in format: left, top, right, bottom
            face_output = face_output[np.where(area > 2000)]
            for id, hf in enumerate(face_output):
                    # append dictionary
                    face_dict.append([int(hf[0]), int(hf[1]), int(hf[2]), int(hf[3])])
        return face_dict



### KEVIN: 9803 out of 20992 images have dimension shape (400,712)
import os
os.chdir('/data/kevin/paper1_swinTr_tt/CAER_S/CAER')

def create_dict():

    ##### FINAL: Go through each image
    # Set Directory


    path = 'train/'
    # emo_path = 'test/Angry'
    final = []

    # uplaod object
    det = detection()

    i = 0
    for emo in os.listdir(path): # emo: Disgust
        emo_path = os.path.join(path,emo)
        for img in os.listdir(emo_path): # img: 0834.png
            img_path = os.path.join(emo_path,img) # img_path: test/Disgust/0834.png

            # human detection
            img_arr = cv2.imread(img_path)
            if img_arr.shape[0] == 400 and img_arr.shape[1] == 712:
                continue
            else:
                hd_coord = det.human_detection_coord(img_arr)
                # face detection
                fd_coord = det.face_detection_coord(img_arr)

                final.append({'file_name': img,
                              'category':emo,
                              'folder':emo_path,
                              'body_bbox': hd_coord,
                              'face_bbox':fd_coord
                              }
                             )

            if i % 500 == 0:
                print('Round',i)

            i += 1



    # # save dictionary
    import pickle
    with open("/data/wangxiaomei/database/CAER-S_dataset/CAER-S/train_list", "wb") as fp:
        pickle.dump(final, fp)


## Generate Dictionary
# create_dict()

#
def combine_dict():
    ## Read dictionary
    import pickle
    with open("/data/wangxiaomei/database/CAER-S_dataset/CAER-S/test_400_712", "rb") as fp1:
        data1 = pickle.load(fp1)
    with open("/data/wangxiaomei/database/CAER-S_dataset/CAER-S/test_others", "rb") as fp2:
        data2 = pickle.load(fp2)
    # Combine Dictionary
    final = data1 + data2

    with open("/data/wangxiaomei/database/CAER-S_dataset/CAER-S/test_list", "wb") as fp:
        pickle.dump(final, fp)


#
#
# ######################### Processing Data
#
# with open("/data/wangxiaomei/database/CAER-S_dataset/CAER-S/test_list", "rb") as fi:
#     # data: [{'file_name': '1827.png', 'category': 'Neutral', 'folder': 'test/Neutral', 'body_bbox': [], 'face_bbox': [[379, 54, 496, 213]]},...]
#     data = pickle.load(fi)


### Find Value Counts
# bb,fb = [],[]
#
# for i in data:
#     bb.append(len(i['body_bbox']))
#     fb.append(len(i['face_bbox']))
#
# bb = pd.DataFrame(bb)
# fb = pd.DataFrame(fb)
#
# print(bb.value_counts())
# print('\n')
# print(fb.value_counts())


# # Process data
# new_data = []
# for i in data:
#     if len(i['body_bbox']) == 0 or len(i['face_bbox']) == 0:
#         continue
#     else:
#         new_data.append(i)
#
#
# print(len(new_data)) # 17796 个数据