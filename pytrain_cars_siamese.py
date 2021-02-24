#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

# ### Paths

# In[2]:


ROOT_DIR = os.getcwd()
print(ROOT_DIR)


# It should be /home/user/Vehicle-Model-Recognition

# ### Dataset parameters

# In[3]:

def create_datetime_dirs(root_dir):
    date = datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    tb_logs_dir = os.path.join(root_dir, 'trainings', 'tensorboard_logs', date)
    save_models_dir = os.path.join(root_dir, 'trainings', 'models', date)
    os.makedirs(tb_logs_dir)
    os.makedirs(save_models_dir)
    return tb_logs_dir, save_models_dir


total_classes = 50
elements_per_class = 100
training_split = 0.75
img_resolution = (224, 224)


# ### Training parameters

# In[ ]:


lr = 1e-4
epochs = 5000
batch_size = 64


# ### Load images

# In[4]:


os.chdir(ROOT_DIR + '/dataset')


# In[5]:


car_names = os.listdir('./')
car_names.sort()


# In[6]:


labels = random.sample(car_names, k = total_classes)
labels.sort()
print(labels)


# In[7]:


(trainX, trainY), (testX, testY), (trainX_bbox, testX_bbox) = utils.load_dataset(ROOT_DIR, labels, elements_per_class, img_resolution=img_resolution, crop=True)
trainX = np.asarray(trainX)
trainY = np.asarray(trainY)
testX = np.asarray(testX)
testY = np.asarray(testY)


# In[8]:


plt.imshow(trainX[0])


# In[9]:


# add a channel dimension to the images
trainX = np.expand_dims(trainX, axis=-1)
testX = np.expand_dims(testX, axis=-1)
print(trainX.shape)
print(testX.shape)


# In[10]:


norm_trainY = utils.normalize_labels(trainY)
norm_testY = utils.normalize_labels(testY)


# In[11]:


# prepare the positive and negative pairs
print("[INFO] preparing positive and negative pairs...")
(pairTrain, labelTrain) = utils.make_pairs(trainX, norm_trainY)
(pairTest, labelTest) = utils.make_pairs(testX, norm_testY)


# In[12]:




# In[13]:


# configure the siamese network
print("[INFO] building siamese network...")
imgA = Input(shape=trainX.shape[1:4])
imgB = Input(shape=trainX.shape[1:4])
# featureExtractor = build_siamese_model((img_resolution[0], img_resolution[1], 1))
featureExtractor = build_vgg16(trainX.shape[1:4])
featsA = featureExtractor(imgA)
featsB = featureExtractor(imgB)

# finally, construct the siamese network
# distance = Lambda(utils.euclidean_distance)([featsA, featsB])
feats = Concatenate()([featsA, featsB])
feats = Dense(128, activation='relu')(feats)
distance = Dense(64, activation='relu')(feats)
outputs = Dense(1, activation="sigmoid")(distance)
model = Model(inputs=[imgA, imgB], outputs=outputs)

# compile the model
print("[INFO] compiling model...")
# model.compile(loss="binary_crossentropy", optimizer="adam",
# 	metrics=["accuracy"])
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
model.compile(loss=tfa.losses.ContrastiveLoss(), optimizer=optimizer,
	metrics=["binary_accuracy"])



class TensorboardCallback(tf.keras.callbacks.Callback):
    def __init__(self, log_dir):
        self.writer = tf.summary.create_file_writer(log_dir)
        self.writer.set_as_default()

    def on_epoch_end(self, epoch, logs=None):
        tf.summary.scalar('Train/Loss', logs['loss'], epoch)
        tf.summary.scalar('Train/Accuracy', logs['binary_accuracy'], epoch)
        tf.summary.scalar('Val/Loss', logs['val_loss'], epoch)
        tf.summary.scalar('Val/Accuracy', logs['val_binary_accuracy'], epoch)

# In[14]:

TB_LOG_DIR, SAVE_MODELS_DIR = create_datetime_dirs(ROOT_DIR)
tb_callback = TensorboardCallback(TB_LOG_DIR)
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(SAVE_MODELS_DIR, monitor='val_binary_accuracy', verbose=0, save_best_only=True,
                                                           save_weights_only=False, mode='auto', save_freq='epoch')





# train the model
print(' ')
print("[INFO] training model...")
history = model.fit(
	[pairTrain[:, 0], pairTrain[:, 1]], labelTrain[:],
	validation_data=([pairTest[:, 0], pairTest[:, 1]], labelTest[:]),
	batch_size=batch_size,
	epochs=epochs,
    callbacks=[tb_callback, checkpoint_callback],
	verbose=1)


# In[ ]:
