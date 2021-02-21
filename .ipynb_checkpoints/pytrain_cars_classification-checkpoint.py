import tensorflow as tf
import numpy as np
import scipy as sp
import scipy.io
import matplotlib.pyplot as plt
import cv2
import os
from PIL import Image
import csv
import random
import copy

from pyimagesearch.siamese_network import *
from pyimagesearch import config
from pyimagesearch import utils
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow_addons as tfa


ROOT_DIR = os.getcwd()
print('Root directory: %s \n' % ROOT_DIR)

# Dataset parameters
total_classes = 150
elements_per_class = 75
training_split = 0.75
img_resolution = (224, 224)

print('Total number of classes: %d' % total_classes)
print('Elements per class (if exist): %d' % elements_per_class)
print('Training - Validation split: %d - %d' %(int(training_split*100), int((1-training_split)*100)) )
print('Total number of classes: %d' % total_classes)

# Select car models
os.chdir(ROOT_DIR + '/dataset')
car_names = os.listdir('./')
car_names.sort()
labels = random.sample(car_names, k = total_classes)
labels.sort()
print(labels)


(trainX, trainY), (testX, testY), (trainX_bbox, testX_bbox) = utils.load_dataset(ROOT_DIR, labels, elements_per_class, img_resolution=img_resolution, crop=True)
trainX = np.asarray(trainX)
trainY = np.asarray(trainY)
testX = np.asarray(testX)
testY = np.asarray(testY)
print(trainX.shape)


