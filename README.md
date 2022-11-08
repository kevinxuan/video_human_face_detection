## Human & Face Detection Output on Videos
#### Author: Kevin Xuan
#### Date: 11/07/2022

### Function
Given a video, this code utilizes **yolov5** human detection to obtain the coordinates of all the humans and **retina face** detection to obtain the coordinates of all the human faces appearing in the input video. The output will be a dictionary that includes all coordinates of human and human face for each frame inside the vide. To use this code, please check out **main.py**


#### Input
The input of this code is a list of video files that we want to run our human & face detection on.

Example: ['test/Disgust/0001.avi','test/Disgust/0002.avi']

#### Output
The output of this code is a dictionary containing the human detection & human face rectangular box coordinates of every video file input.

Example: {'/test/Disgust/0001.avi' : {'human': {person1:[[0,0,1,1],...],
                                           person2:[[...]]
                                          }, 
                                 'face': {frame1:{person1: [a,b,c,d],person2:[e,f,g,h],
                                          frame2:{...}
                                          }
                                },
     '/test/Disgust/0002.avi': {...},...}
