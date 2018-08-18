#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 CESAR SINCHIGUANO <cesarsinchiguano@hotmail.es>
#
# Distributed under terms of the BSD license.
#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 CESAR SINCHIGUANO <cesarsinchiguano@hotmail.es>
#
# Distributed under terms of the BSD license.


import cv2
import matplotlib.pyplot as plt
from tf_pose.estimator import TfPoseEstimator
from tf_pose import *
import numpy as np


#===================
#DEFAULT PARAMETERS
resize='432x368'#Recommends : 432x368
resize_to_default=True
resize_out_ratio=4.0#default


def plotting(img):
	plt.figure('Pose Estimation!')
	plt.title('Result!')
	plt.imshow(img)
	plt.show()

def main():

    #Load the image
    img= cv2.imread(image_path, cv2.IMREAD_COLOR)


    #Resize image before they are processed.
    w, h = map(int, resize.split('x'))

    #Load my pretrained model according to my target size
    model_POSEest = TfPoseEstimator(model_add, target_size=(w, h))

    #==========================================================================
    #==========================================================================

    #Estimate human poses from a single image
    poseESTIMATION=model_POSEest.inference(img,True,4.0)
    print('.................................')
    print('Prediction of my post estimation')
    print('.................................')
    print(poseESTIMATION)
    #print('type(poseESTIMATION):\n',type(poseESTIMATION))

    points_2d=list();


    for human in poseESTIMATION:
        points_2d,visibility=common.MPIIPart.from_coco(human);
    points_2d=np.array(points_2d)

    print('======================================')
    print('Informations about the kinematics!!!!!')
    print('======================================')
    print(type(points_2d),points_2d.shape,len(points_2d))
    tmp=list()
    aux=list()
    part=['RAnkle = 0','RKnee = 1','RHip = 2','LHip = 3','LKnee = 4','LAnkle = 5','RWrist = 6',
    'RElbow = 7','RShoulder = 8','LShoulder = 9','LElbow = 10','LWrist = 11','Neck = 12','Head = 13']
    for i in range(len(points_2d)):
        print(part[i],':',points_2d[i])
        if (i==6)or(i==7)or(i==8):
            tmp.append(list(points_2d[i]))
            aux.append(part[i])

    print('--------------------------------------------')
    print('keypoints of interest for the right hand(x,y coordinates)!!!')
    print('--------------------------------------------')
    for i in range(len(tmp)):
        print(aux[i],tmp[i])







    #==========================================================================
    #==========================================================================


    # draw points and draw lines
    img= TfPoseEstimator.draw_humans(img, poseESTIMATION, imgcopy=False)

    #Colors and color conversions>>Convert image to RGB color for matplotlib
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # show image with matplotlib
    plotting(img)

if __name__=='__main__':
    #>>Address about the pretrained model
    model_add='./mobilenet_thin/graph_opt.pb'
    #image_path='./images/e1.jpg'
    #image_path='./images/arm1.jpg'
    #image_path='./images/ski.jpg'
    image_path='./images/q1.jpg'
    main()
