# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 12:59:42 2018

@author: brandt84828
"""

from keras.datasets import boston_housing
from keras import models,layers
import numpy as np
(train_data,train_labels),(test_data,test_labels) = boston_housing.load_data()

#求出mean 和 std >>>normalization
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std
test_data -= mean
test_data /= std

def bulid_model():
    model = models.Sequential()
    model.add(layers.Dense(64,activation='relu',input_shape=(train_data.shape[1],))) #特徵數量
    model.add(layers.Dense(64,activation='relu'))
    model.add(layers.Dense(1))#輸出一個結果
    model.compile(optimizer='rmsprop',
                  loss='mse',
                  metrics=['mae'])
    return model



