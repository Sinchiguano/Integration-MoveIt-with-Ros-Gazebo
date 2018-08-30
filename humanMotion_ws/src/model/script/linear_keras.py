#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 CESAR SINCHIGUANO <cesarsinchiguano@hotmail.es>
#
# Distributed under terms of the BSD license.

import numpy as np
import scipy.sparse as sp
from sklearn.utils import shuffle
from sklearn.utils.testing import assert_almost_equal
from sklearn.utils.testing import assert_raises
from sklearn.utils.testing import assert_raises_regex
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_equal
from sklearn.exceptions import NotFittedError
from sklearn import datasets
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import Lasso
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier

def compute_err_MSE(y, yhat):
    """Return the mean squared error given the true values of y and predictions yhat."""
    # MODIFY THE FOLLOWING WITH YOUR CODE
    err=(yhat-y)**2
    err=np.sum(err)/len(y)
    return err


X, y = datasets.make_regression(n_targets=3)
X_train, y_train = X[:50], y[:50]
X_test, y_test = X[50:], y[50:]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.75)

print(len(X_train[0]))
print(len(X))
# print(X[:1])
# print(y[:1])
# exit(0)
# references = np.zeros_like(y_test)
# for n in range(3):
#     rgr = GradientBoostingRegressor(random_state=0)
#     rgr.fit(X_train, y_train[:, n])
#     references[:,n] = rgr.predict(X_test)

rgr = MultiOutputRegressor(GradientBoostingRegressor(random_state=0))
rgr.fit(X_train, y_train)
y_pred = rgr.predict(X_test)

from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
print('Training:',mean_squared_error(y_train, rgr.predict(X_train)))
print('Testing:',mean_squared_error(y_test, rgr.predict(X_test)))
print('mine:',compute_err_MSE(y_test, rgr.predict(X_test)))
print('mine:',compute_err_MSE(y_train, rgr.predict(X_train)))



from sklearn.metrics import mean_squared_error
y_true = [3, -0.5, 2, 7]
y_pred = [3, -0.5, 2, 7]
print(mean_squared_error(y_true, y_pred))












"""


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

# Generate dummy data
import numpy as np
x_train = np.random.random((1000, 6))
#tmp=np.random.randint(7, size=(1000, 1))
y_train = keras.utils.to_categorical(np.random.randint(7, size=(1000, 1)), num_classes=7)
#print(y_train[:10])
#exit(0)
x_test = np.random.random((100, 6))
y_test = keras.utils.to_categorical(np.random.randint(7, size=(100, 1)), num_classes=7)

model = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model.add(Dense(64, activation='relu', input_dim=6))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=20,
          batch_size=128)
classes = model.predict(x_test, batch_size=128)
print(classes[:1])
print('score = model.evaluate')
score = model.evaluate(x_test, y_test, batch_size=128)
print(score)
"""
