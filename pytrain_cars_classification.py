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
import datetime


ROOT_DIR = os.getcwd()
print('Root directory: %s \n' % ROOT_DIR)
LOGS_DIR = os.path.join(ROOT_DIR, 'logs/fit', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

# Dataset parameters
total_classes = 150
elements_per_class = 100
training_split = 0.75
img_resolution = (224, 224)

print('Total number of classes: %d' % total_classes)
print('Elements per class (if exist): %d' % elements_per_class)
print('Training - Validation split: %d%% - %d%%' %(int(training_split*100), int((1-training_split)*100)) )
print('Total number of classes: %d\n' % total_classes)

# Select car models
os.chdir(ROOT_DIR + '/dataset')
car_names = os.listdir('./')
car_names.sort()
labels = random.sample(car_names, k = total_classes)
labels.sort()

print('Loading dataset ...')
(trainX, trainY), (testX, testY), (trainX_bbox, testX_bbox) = utils.load_dataset(ROOT_DIR, labels, elements_per_class, img_resolution=img_resolution, crop=True)
trainX = np.asarray(trainX)
trainY = np.asarray(trainY)
testX = np.asarray(testX)
testY = np.asarray(testY)
print('Dataset loaded')
print('%d training images, %d validation images \n' % (len(trainX), len(testX)))

trainX = np.expand_dims(trainX, axis=-1)
testX = np.expand_dims(testX, axis=-1)
print(trainX.shape)
print(testX.shape)

norm_trainY = utils.normalize_labels(trainY)
norm_testY = utils.normalize_labels(testY)

oh_train = tf.one_hot(norm_trainY, len(np.unique(norm_trainY)))
oh_test = tf.one_hot(norm_testY, len(np.unique(norm_testY)))

vgg16 = tf.keras.applications.VGG16()
inputs = Input(shape=(224,224,3))
vgg16_outputs = vgg16(inputs)
outputs = Dense(len(np.unique(norm_trainY)), activation='relu')(vgg16_outputs)
model = Model(inputs, outputs)

model.compile(loss='categorical_crossentropy', optimizer="adam",
	metrics=["accuracy"])

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=LOGS_DIR, histogram_freq=1)

history = model.fit(trainX, oh_train,validation_data=(trainX, oh_train),
                    batch_size=4, epochs=50, callbacks=[tensorboard_callback])
