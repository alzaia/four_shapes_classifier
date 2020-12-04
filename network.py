#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Definition of the CNN architectures.
Baseline: very simple net with few params and no regularization.
Regularized: simple net combined with dropout and data augmentation.
"""

import numpy as np
import random

import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.preprocessing import *


def define_baseline(num_classes,input_shape=(200, 200, 1)):
    
    inputs = Input(input_shape)
    inputs_preproc = experimental.preprocessing.Resizing(100, 100, interpolation='bilinear')(inputs)
    inputs_preproc = experimental.preprocessing.Rescaling(1./255, input_shape=input_shape)(inputs_preproc)
    
    conv1 = Conv2D(8, 3, padding='same', activation='relu')(inputs_preproc)
    conv1 = MaxPool2D(pool_size=(2,2))(conv1)
    
    conv2 = Conv2D(16, 3, padding='same', activation='relu')(conv1)
    conv2 = MaxPool2D(pool_size=(2,2))(conv2)
    
    flatten1 = Flatten()(conv2)
    outputs = Dense(16, activation='relu')(flatten1)
    outputs = Dense(num_classes, activation='softmax')(outputs)
    
    model = Model(inputs = inputs, outputs = outputs)
    
    return model


data_augmentation = Sequential(
  [
    experimental.preprocessing.RandomFlip("horizontal", input_shape=(200,200,1)),
    experimental.preprocessing.RandomZoom(0.4),
    experimental.preprocessing.RandomContrast(0.9)
  ]
)

def define_regularized(num_classes,input_shape=(200, 200, 1)):
    
    inputs = Input(input_shape)
    inputs_preproc = data_augmentation(inputs)
    inputs_preproc = experimental.preprocessing.Resizing(100, 100, interpolation='bilinear')(inputs_preproc)
    inputs_preproc = experimental.preprocessing.Rescaling(1./255, input_shape=input_shape)(inputs_preproc)
    
    conv1 = Conv2D(16, 3, padding='same', activation='relu')(inputs_preproc)
    conv1 = MaxPool2D(pool_size=(2,2))(conv1)
    conv1 = Dropout(0.25)(conv1)
    
    conv2 = Conv2D(32, 3, padding='same', activation='relu')(conv1)
    conv2 = MaxPool2D(pool_size=(2,2))(conv2)
    conv2 = Dropout(0.25)(conv2)
    
    flatten1 = Flatten()(conv2)
    outputs = Dense(16, activation='relu')(flatten1)
    outputs = Dense(num_classes, activation='softmax')(outputs)
    
    model = Model(inputs = inputs, outputs = outputs)
    
    return model

