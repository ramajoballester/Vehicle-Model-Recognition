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
# from pyimagesearch import config
from pyimagesearch import utils
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
import datetime
import os
import pickle


# ### Paths

# In[2]:


ROOT_DIR = os.getcwd()
print(ROOT_DIR)

prev_trainings = os.listdir(os.path.join(ROOT_DIR, 'trainings', 'models'))
prev_trainings.sort()

# It should be /home/user/Vehicle-Model-Recognition

# In[3]:


# ### Dataset parameters

# In[4]:


total_classes = 196
elements_per_class = 200
training_split = 0.75
img_resolution = (224, 224)


# ### Training parameters

# In[21]:

resume = False
lr = 1e-4
epochs = 5000
batch_size = 128


# ### Load images

# In[12]:


def create_datetime_dirs(root_dir):
    date = datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    tb_logs_dir = os.path.join(root_dir, 'trainings', 'tensorboard_logs', date)
    save_models_dir = os.path.join(root_dir, 'trainings', 'models', date)
    os.makedirs(tb_logs_dir)
    os.makedirs(save_models_dir)
    return tb_logs_dir, save_models_dir

os.chdir(ROOT_DIR + '/dataset')


# In[7]:


car_names = os.listdir('./')
car_names.sort()


# In[8]:


labels = random.sample(car_names, k = total_classes)
labels.sort()
print(labels)


# In[9]:


(trainX, trainY), (testX, testY), (trainX_bbox, testX_bbox) = utils.load_dataset(ROOT_DIR, labels, elements_per_class, img_resolution=img_resolution,
                                                                                 crop=True, greyscale=False)
trainX = np.asarray(trainX)
trainY = np.asarray(trainY)
testX = np.asarray(testX)
testY = np.asarray(testY)
trainX.shape


# In[13]:


plt.imshow(trainX[0])


# In[14]:


# add a channel dimension to the images
trainX = np.expand_dims(trainX, axis=-1)
testX = np.expand_dims(testX, axis=-1)
print(trainX.shape)
print(testX.shape)


# In[15]:


norm_trainY = utils.normalize_labels(trainY)
norm_testY = utils.normalize_labels(testY)


# In[16]:


oh_train = tf.one_hot(norm_trainY, len(np.unique(norm_trainY)))
oh_test = tf.one_hot(norm_testY, len(np.unique(norm_testY)))


# In[17]:


vgg16 = tf.keras.applications.VGG16(include_top=True, weights=None,input_shape=trainX.shape[1:4], classes=total_classes)


# In[18]:


vgg16 = build_vgg16(trainX.shape[1:4])


# In[19]:


vgg16.summary()


# In[20]:


inputs = Input(shape=trainX.shape[1:4])
vgg16_outputs = vgg16(inputs)
outputs = Dense(len(np.unique(norm_trainY)), activation='relu')(vgg16_outputs)
model = Model(inputs, outputs)


# In[22]:


optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
model.compile(loss='categorical_crossentropy', optimizer=optimizer,
	metrics=["categorical_accuracy"])


if resume:
    model = tf.keras.models.load_model('/home/alvaro/Vehicle-Model-Recognition/trainings/models/' + prev_trainings[-1])

# In[23]:


model.summary()


# In[24]:


class TensorboardCallback(tf.keras.callbacks.Callback):
    def __init__(self, log_dir):
        self.writer = tf.summary.create_file_writer(log_dir)
        self.writer.set_as_default()

    def on_epoch_end(self, epoch, logs=None):
        tf.summary.scalar('Train/Loss', logs['loss'], epoch)
        tf.summary.scalar('Train/Accuracy', logs['categorical_accuracy'], epoch)
        tf.summary.scalar('Val/Loss', logs['val_loss'], epoch)
        tf.summary.scalar('Val/Accuracy', logs['val_categorical_accuracy'], epoch)


# In[ ]:

if resume:
    with open(os.path.join(ROOT_DIR, 'pickle', 'train.pickle'), 'rb') as f:
        trainX, oh_train, testX, oh_test = pickle.load(f)
        f.close()
else:
    with open(os.path.join(ROOT_DIR, 'pickle', 'train.pickle'), 'wb') as f:
        pickle.dump([trainX, oh_train, testX, oh_test], f)
        f.close()

TB_LOG_DIR, SAVE_MODELS_DIR = create_datetime_dirs(ROOT_DIR)
tb_callback = TensorboardCallback(TB_LOG_DIR)
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(SAVE_MODELS_DIR, monitor='val_categorical_accuracy', verbose=0, save_best_only=True,
                                                           save_weights_only=False, mode='auto', save_freq='epoch')

history = model.fit(
	trainX, oh_train,
	validation_data=(testX, oh_test),
	batch_size=batch_size,
	epochs=epochs,
    callbacks=[tb_callback, checkpoint_callback],
    verbose=1)


# In[ ]:
