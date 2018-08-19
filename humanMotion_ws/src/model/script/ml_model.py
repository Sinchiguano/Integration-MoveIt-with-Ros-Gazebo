#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 CESAR SINCHIGUANO <cesarsinchiguano@hotmail.es>
#
# Distributed under terms of the BSD license.

"""

"""
import csv
import pandas as pd


file_name='joint_data.csv'


def csv_as_dataframe():
    """Load 'csv file and return dataframe"""
    tmp_data= pd.read_csv(file_name)
    X_=tmp_data[['RWrist = 6x','RWrist = 6y','RElbow = 7x','RElbow = 7y','RShoulder = 8x','RShoulder = 8y']]
    y_=tmp_data[['r_shoulder_pan_joint','r_shoulder_lift_joint','r_upper_arm_roll_joint',
            'r_elbow_flex_joint','r_forearm_roll_joint','r_wrist_flex_joint',
            'r_wrist_roll_joint']]

    #print(tmp_data.head(3))
    return X_,y_



def main():
    print('=================')
    X_data,y_data=csv_as_dataframe()

    print(X_data.head(3))
    print(y_data.head(3))

if __name__=='__main__':
    main()
