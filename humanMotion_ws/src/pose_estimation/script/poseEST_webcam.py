#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 CESAR SINCHIGUANO <cesarsinchiguano@hotmail.es>
#
# Distributed under terms of the BSD license.

import numpy as np
import cv2
import sys
import time
from tqdm import *
from matplotlib import pyplot as plt
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh


#DEFAULT PARAMETERS
resize='432x368'#Recommends : 432x368
resize_to_default=True
resize_out_ratio=4.0#default


# Create a VideoCapture object and read from input camera
cap = cv2.VideoCapture(0)

def main():
    # Capture frame-by-frame
    ret, frame = cap.read()# A frame of a video is simply an image
    #print('cam image=%dx%d' % (frame.shape[1], frame.shape[0]))

    #Resize image before they are processed.
    w, h = map(int, resize.split('x'))

    #Load my pretrained model according to my target size
    model_PEST = TfPoseEstimator(model_path, target_size=(w, h))
    #================================================================
    pbar = tqdm(ascii=True)
    counter=0
    while(True):
        counter=counter+1

        # Capture frame-by-frame
        _, frame = cap.read()# A frame of a video is simply an image
        #Estimate human poses from a single image
        poseESTIMATION=model_PEST.inference(frame,resize_to_default,resize_out_ratio)
        # draw points and draw lines
        imgPOSEST= TfPoseEstimator.draw_humans(frame, poseESTIMATION, imgcopy=False)
        # show image with matplotlib
        cv2.imshow('Pose Estimation', imgPOSEST)

        if  cv2.waitKey(1) & 0xFF== ord('s'):
            print('Goodbye!!!')
            break
        pbar.update(counter)
        time.sleep(0.01)
    pbar.close()
    #==================================================================
    # When everything done, release the video capture object
    cap.release()
    # Closes all the frames
    cv2.destroyAllWindows()
if __name__=='__main__':
    #>>Address about the pretrained model
    model_path='/home/casch/myproject/models/graph/mobilenet_thin/graph_opt.pb'
    main()
