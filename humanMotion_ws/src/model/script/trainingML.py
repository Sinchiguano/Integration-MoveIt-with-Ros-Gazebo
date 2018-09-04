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

    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data,test_size=0.25)

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


    #===================================================
    # MODELS
    # #Create a model
    #====================================================
    MLPRegressorModel = MLPRegressor(hidden_layer_sizes=(10,10,10,10),verbose=False)

    # model = Sequential()
    # model.add(Dense(20, activation='relu', input_dim=6))
    # model.add(Dropout(0.5))
    # model.add(Dense(64, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(7, activation='softmax'))
    #
    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    # model.compile(loss='categorical_crossentropy',
    #               optimizer=sgd,
    #               metrics=['accuracy'])

    # #Training step

    MLPRegressorModel.fit(X_train,y_train)
    # model.fit(X_train, y_train)


    # #Make predictions
    y_hat3=MLPRegressorModel.predict(X_test)
    # y_hat4=model.predict(X_test)



    # #Accuracy!!!

    print('\n========MLPRegressor=========\n')
    print('Training:',mean_squared_error(y_train, MLPRegressorModel.predict(X_train)))
    print('Testing:',mean_squared_error(y_test, y_hat3))


    # print('\n=====keras.models=========\n')
    # print('Training:',mean_squared_error(y_train, model.predict(X_train)))
    # print('Testing:',mean_squared_error(y_test, y_hat4))
    # scores = model.evaluate(X_test, y_test, verbose=0)
    # print("Test: %s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    # scores = model.evaluate(X_train, y_train, verbose=0)
    # print("Training:%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))



    # #======================================
    # #Save my training model for machine learning algorithm done in python
    # from sklearn.externals import joblib
    #
    # filename = 'pre_trained_model.sav'
    # joblib.dump(MLPRegressorModel, filename)
    # #loaded_model = joblib.load(filename)
    #
    #
    # from keras.models import model_from_json
    # import numpy,h5py
    # import os
    #
    # # serialize model to JSON
    # model_json = model.to_json()
    # with open("model.json", "w") as json_file:
    #     json_file.write(model_json)
    #
    # # serialize weights to HDF5
    # model.save_weights("model.h5")
    # print("Saved model to disk")
    # print('======================================')

    # # later...
    # # load json and create model
    # json_file = open('model.json', 'r')
    # loaded_model_json = json_file.read()
    # json_file.close()
    # loaded_model = model_from_json(loaded_model_json)
    # # load weights into new model
    # loaded_model.load_weights("model.h5")
    # print("Loaded model from disk")
    # print('======================================')
    #
    # # evaluate loaded model on test data
    # loaded_model.compile(loss='categorical_crossentropy',
    #               optimizer=sgd,
    #               metrics=['accuracy'])
    # score = loaded_model.evaluate(X_test, y_test, verbose=0)
    # print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
    # scores = loaded_model.evaluate(X_train, y_train, verbose=0)
    # print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))



if __name__=='__main__':
    main()
