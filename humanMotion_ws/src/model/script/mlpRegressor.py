#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 CESAR SINCHIGUANO <cesarsinchiguano@hotmail.es>
#
# Distributed under terms of the BSD license.

"""

"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor


file_name='joint_data.csv'
joints=['r_shoulder_pan_joint','r_shoulder_lift_joint','r_upper_arm_roll_joint',
        'r_elbow_flex_joint','r_forearm_roll_joint','r_wrist_flex_joint',
        'r_wrist_roll_joint']
posexy=['RWrist = 6x','RWrist = 6y','RElbow = 7x','RElbow = 7y','RShoulder = 8x','RShoulder = 8y']

def csv_as_dataframe():
    """Load csv file and return dataframe"""
    tmp_data= pd.read_csv(file_name)
    X_=tmp_data[posexy].values[:,:]
    y_=tmp_data[joints].values[:,:]
    return X_,y_

def compute_err_MSE(y, yhat):
    """Return the mean squared error given the true values of y and predictions yhat."""
    err=(yhat-y)**2
    err=np.sum(err)/len(y)
    return err

def main():

    X_data,y_data=csv_as_dataframe()

    # Scale the data
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()

    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data,test_size=0.10)

    # Fit only to the training data
    scaler.fit(X_train)

    # Now apply the transformations to the data:
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # print('\nShapes of training and testing X:')
    # print('X_train.shape',X_train.shape)
    # print('X_test.shape',X_test.shape)
    # print('Shapes of training and testing y:')
    # print('y_train.shape',y_train.shape)
    # print('y_test.shape',y_test.shape)
    # print('X_test')



    #MODEL

    MLPRegressorModel = MLPRegressor(hidden_layer_sizes=(10,10,10),activation='identity',solver='lbfgs',verbose=False)

    # #Training step

    MLPRegressorModel.fit(X_train,y_train)

    # #Make predictions
    y_hat=MLPRegressorModel.predict(X_test)

    # #Accuracy!!!

    print('\n========MLPRegressor=========\n')
    print('Training:',mean_squared_error(y_train, MLPRegressorModel.predict(X_train)))
    print('Testing:',mean_squared_error(y_test, y_hat))

    print('my mean square error!!!')
    print('Training:',compute_err_MSE(y_train, MLPRegressorModel.predict(X_train)))
    print('Testing:',compute_err_MSE(y_test, y_hat))

    #======================================
    #Save my training model for machine learning algorithm done in python
    from sklearn.externals import joblib
    #
    # filename = 'mlpRegressor_model.sav'
    # joblib.dump(MLPRegressorModel, filename)

    #loaded_model = joblib.load(filename)



if __name__=='__main__':
    main()
