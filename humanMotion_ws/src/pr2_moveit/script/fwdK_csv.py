#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 CESAR SINCHIGUANO <cesarsinchiguano@hotmail.es>
#
# Distributed under terms of the BSD license.


import sys
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from math import pi
from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list
import csv
import time
import copy



class move_group(object):

    def __init__(self):
        ''' First initialize moveit_commander and rospy.'''
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('move_group_python',anonymous=True)

        '''Instantiate a RobotCommander object. This object is an interface to
        the robot as a whole.'''
        self.robot = moveit_commander.RobotCommander()

        '''Instantiate a MoveGroupCommander object.  This object is an interface
        to one group of joints.'''
        self.group = moveit_commander.MoveGroupCommander("right_arm")

        """ Get the current configuration of the group as a list (these are values published on /joint_states) """
        self.home_joints= self.group.get_current_joint_values()
        ''''Get the current pose of the end-effector of the group.'''
        self.home_state=self.robot.get_current_state()

        '''Instantiate a `PlanningSceneInterface`_ object.  This object is an interface
        to the world surrounding the robot:'''
        self.scene = moveit_commander.PlanningSceneInterface()

        self.tmp_joints=['r_shoulder_pan_joint',
                      'r_shoulder_lift_joint',
                      'r_upper_arm_roll_joint',
                      'r_elbow_flex_joint',
                      'r_forearm_roll_joint',
                      'r_wrist_flex_joint',
                      'r_wrist_roll_joint']
        self.part=['RAnkle = 0','RKnee = 1','RHip = 2','LHip = 3',
                    'LKnee = 4','LAnkle = 5','RWrist = 6',
                    'RElbow = 7','RShoulder = 8','LShoulder = 9',
                    'LElbow = 10','LWrist = 11','Neck = 12','Head = 13']

        self.header= copy.copy(self.tmp_joints)

        self.name_file='joint_data.csv'

        with open(self.name_file, 'wb') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for i in range(len(self.part)):
                if (i==6)or(i==7)or(i==8):
                    self.header.append(self.part[i])
            filewriter.writerow(self.header)

    def go_to_joint_state(self,counter):
        group = self.group

        ## Planning to a joint-space goal
        #================================
        joint_goal=group.get_random_joint_values()

        #===============================
        '''Creating my CSV file'''
        self.csv_file(joint_goal)#go to my method csvfile with the joint_goal
        #It is better to create a copy instead of passing the value directly, it will overwrite.
        print "============ Joint values: "
        #print(self.tmp_joints)
        #print(joint_goal)#for debugging

        for i in range(len(joint_goal)):
            print(i,self.tmp_joints[i],joint_goal[i])
        print('################################')
        group.set_joint_value_target(joint_goal)

        plan = group.plan()
        #When working with the real robot uncomment the following line...
        #group.execute(plan)

        print "============ Waiting while RVIZ displays plan..."
        self.box_alert()
        print('Counte:',counter)
        print('done!!!')

    def box_alert(self):
        box_pose = geometry_msgs.msg.PoseStamped()
        box_pose.header.frame_id = self.group.get_planning_frame()
        box_pose.pose.position.x =0;
        box_pose.pose.position.y = 0.5;
        box_pose.pose.position.z = 0.5;
        box_pose.pose.orientation.w = 0.5
        box_name = "box"

        for i in range(12):
            self.scene.add_box(box_name, box_pose, size=(0.1, 0.1, 0.1))
            time.sleep(0.3)
            self.scene.remove_world_object(box_name)
            time.sleep(0.3)

    def csv_file(self,joint_list):
        #Opening a file with the 'a' parameter allows you to append to the end of the file instead of simply overwriting the existing content.
        tmp_frame=list()
        tmp_list=copy.copy(joint_list)#it is a good approach to create copy in order not to overwrite the arguments.

        with open(self.name_file, 'a') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)

            tmp_frame=self.cv2_frame()
            #tmp_pose_list=pose_estimation(tmp_frame)
            #joint_list.append(tmp_pose_list)
            for i in tmp_frame:
                tmp_list.append(i)
            filewriter.writerow(tmp_list)
    #Take a picture
    def cv2_frame(self):
        img=list()
        print('Hello cv2 frame')
        img=[1,2,3]
        return img




'''
https://github.com/ros-planning/moveit/blob/kinetic-devel/moveit_commander/src/moveit_commander/move_group.py
'''
