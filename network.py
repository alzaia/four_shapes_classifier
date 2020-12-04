#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Definition of the CNN architectures.
Baseline: very simple net with few params and no regularization.
Regularized: simple net combined with dropout and data augmentation.
"""



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

