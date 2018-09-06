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
import sys


from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD



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

    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data,test_size=0.15)

    # Fit only to the training data
    scaler.fit(X_train)

    # Now apply the transformations to the data:
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # scaler.fit(y_train)
    #
    # # Now apply the transformations to the data:
    # y_train = scaler.transform(y_train)
    # y_test = scaler.transform(y_test)

    # print('\nShapes of training and testing X:')
    # print('X_train.shape',X_train.shape)
    # print('X_test.shape',X_test.shape)
    # print('Shapes of training and testing y:')
    # print('y_train.shape',y_train.shape)
    # print('y_test.shape',y_test.shape)
    # print('X_test')


    # create model
    model = Sequential()
    model.add(Dense(units=40, input_dim=6, kernel_initializer='normal', activation='relu'))
    model.add(Dense(40, kernel_initializer='normal', activation='relu'))
    model.add(Dense(40, kernel_initializer='normal', activation='relu'))
    model.add(Dense(7, kernel_initializer='normal'))
    # Compile model
    # For a mean squared error regression problem
    model.compile(loss='mean_squared_error', optimizer='rmsprop')

    # #Training step

    model.fit(X_train, y_train)


    # #Make predictions
    y_hat=model.predict(X_test)

    #print('y_hat:',y_hat[:2])



    # #Accuracy!!!

    print('\n=====keras.models=========\n')
    print('Training:',mean_squared_error(y_train, model.predict(X_train)))
    print('Testing:',mean_squared_error(y_test, y_hat))
    #Mean mean_squared_error from Keras library
    print('Mean mean_squared_error from Keras library from keras')
    print(model.evaluate(X_test, y_test))



    #======================================
    from keras.models import model_from_json
    import h5py


    ##serialize model to JSON
    # model_json = model.to_json()
    # with open("model.json", "w") as json_file:
    #     json_file.write(model_json)
    #
    # # serialize weights to HDF5
    # model.save_weights("model.h5")
    # print("Saved model to disk")
    # print('======================================')

    # print("\nLoaded model from disk")
    # print('======================================\n')
    #
    # json_file = open('model.json', 'r')
    # loaded_model_json = json_file.read()
    # json_file.close()
    # loaded_model = model_from_json(loaded_model_json)
    # # load weights into new model
    # loaded_model.load_weights("model.h5")
    #
    #
    # # evaluate loaded model on test data
    #
    # print('Training:',mean_squared_error(y_train, loaded_model.predict(X_train)))
    # print('Testing:',mean_squared_error(y_test, loaded_model.predict(X_test)))
    # loaded_model.compile(loss='mean_squared_error', optimizer='rmsprop')
    # print('mse',loaded_model.evaluate(X_test, y_test))



if __name__=='__main__':
    main()
