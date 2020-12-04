#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Definition of the CNN architectures.
Baseline: very simple net with few params and no regularization.
Regularized: simple net combined with dropout and data augmentation.
"""

import numpy as np
import random
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


def plot_training_curves(history):
    
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(len(history.epoch))

    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()




