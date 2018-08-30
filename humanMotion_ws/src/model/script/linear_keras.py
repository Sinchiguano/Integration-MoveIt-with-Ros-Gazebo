#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 CESAR SINCHIGUANO <cesarsinchiguano@hotmail.es>
#
# Distributed under terms of the BSD license.

"""

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
