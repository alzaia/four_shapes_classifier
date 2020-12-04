#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Definition of the dataset handlers.
Datasets for train/val come from the Kaggle link (see readme).
Samples are loaded from the dataset directory.
"""

import numpy as np
import random

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing import *



def train_data_handler(batch_size,img_size,val_split,train_path):
    train_data = image_dataset_from_directory(directory=train_path,batch_size=batch_size,labels='inferred',image_size=img_size,
    seed=5555,validation_split=val_split,subset="training",color_mode="grayscale")
    val_data = image_dataset_from_directory(directory=train_path,batch_size=batch_size,labels='inferred',image_size=img_size,
    seed=5555,validation_split=val_split,subset="validation",color_mode="grayscale")
    return train_data, val_data

def test_data_handler(batch_size,img_size,val_split,test_path):
    test_data = image_dataset_from_directory(directory=test_path,batch_size=batch_size,labels='inferred',image_size=img_size,
    shuffle=False,seed=5555,color_mode="grayscale")
    return test_data



