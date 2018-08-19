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
from sklearn import linear_model, metrics, svm, preprocessing
import numpy as np


file_name='joint_data.csv'
joints=['r_shoulder_pan_joint','r_shoulder_lift_joint','r_upper_arm_roll_joint',
        'r_elbow_flex_joint','r_forearm_roll_joint','r_wrist_flex_joint',
        'r_wrist_roll_joint']

def csv_as_dataframe():
    """Load 'csv file and return dataframe"""
    tmp_data= pd.read_csv(file_name)
    X_=tmp_data[['RWrist = 6x','RWrist = 6y','RElbow = 7x','RElbow = 7y','RShoulder = 8x','RShoulder = 8y']]
    y_=tmp_data['label']
    tmp_out=tmp_data[joints]
    #print('joints')
    #print(tmp_out)
    #print(tmp_data.head(3))
    #return X_,y_
    return X_,y_

def main():
    print('=================')
    X_data,y_data=csv_as_dataframe()
    print('data for training')
    #print(X_data.head(3))
    #print(y_data.head(3 ))

    # Scale the data
    sc = preprocessing.StandardScaler()
    X_data= sc.fit_transform(X_data)

    ####################################################

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data,test_size=0.25)
    print(y_train[:5])
    print(y_test[:5])
    #exit(0)
    print('\nShapes of training and testing X:')
    print('X_train.shape',X_train.shape)
    print('X_test.shape',X_test.shape)
    print('Shapes of training and testing y:')
    print('y_train.shape',y_train.shape)
    print('y_test.shape',y_test.shape)


    #===================================================
    # Support vector classification with RBF kernel
    #====================================================
    from sklearn import svm
    from sklearn.svm import SVC
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import accuracy_score

    #Create a model!!!
    #C_range = np.arange(0.1,2.5 , 0.2)
    #gamma_range = np.arange(0, 5, 0.1)
    #param_grid = dict(gamma=gamma_range, C=C_range)

    model = svm.SVC(kernel='rbf',gamma=0.7, probability=True,C=1)
    #model = GridSearchCV(svc, param_grid=param_grid, cv=3)
    model.fit(X_train,y_train)

    #Make predictions!!!
    y_=model.predict(X_test)


    #Accuracy!!!
    print('\nThe training and testing accuracies when simple split is used:')
    acc_tst=accuracy_score(y_test,y_)
    print('test  acc:', acc_tst)


    #======================================
    #Save my training model
    from sklearn.externals import joblib

    filename = 'pre-trained_model.sav'
    joblib.dump(model, filename)
    #loaded_model = joblib.load(filename)

if __name__=='__main__':
    main()
