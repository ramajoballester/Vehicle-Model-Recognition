import tensorflow as tf
import numpy as np
import scipy as sp
import scipy.io
import matplotlib.pyplot as plt
import cv2
import os
from PIL import Image
import csv

# Read images function
def load_images(folder):
    images = []
    for i in range(1, len(os.listdir(folder)) + 1):
        filename = format(i, '06d') + '.jpg'
        img = cv2.imread(os.path.join(folder,filename))
        # OpenCV loads images to BGR by default (modify this to RGB)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img is not None:
            images.append(img)
    return images


car_images = np.array(load_images('car_ims'), dtype=object)
annotation_file = sp.io.loadmat('cars_annos.mat')
img_labels = annotation_file.get('annotations')[0]
labels = annotation_file.get('class_names')[0]

print('There are %d images from %d car models\n' % (len(img_labels), len(labels)))

# Names, bounding box and class
names = np.array([img_labels[i][0][0] for i in range(len(img_labels))])
top_left_x = np.array([img_labels[i][1][0][0] for i in range(len(img_labels))])
top_left_y = np.array([img_labels[i][2][0][0] for i in range(len(img_labels))])
bot_right_x = np.array([img_labels[i][3][0][0] for i in range(len(img_labels))])
bot_right_y = np.array([img_labels[i][4][0][0] for i in range(len(img_labels))])
car_class = np.array([img_labels[i][5][0][0] for i in range(len(img_labels))])
car_test = np.array([img_labels[i][6][0][0] for i in range(len(img_labels))])

# Create subdirectories
os.makedirs('dataset')
os.chdir('dataset')
for each in labels:
    os.makedirs(each[0])

# Create filenames
img_filenames = [names[i].split('/')[-1] for i in range(len(names))]
csv_filenames = [names[i].split('/')[-1].split('.')[0] + '.csv' for i in range(len(names))]

# Dump images and save information to folders
for i in range(len(car_images)):
    row = [top_left_x[i], top_left_y[i], bot_right_x[i], bot_right_y[i], car_class[i], car_test[i]]
    os.chdir(labels[car_class[i]-1][0] if labels[car_class[i]-1][0] != 'Ram C/V Cargo Van Minivan 2012' else 'Ram CV Cargo Van Minivan 2012')
    cv2.imwrite(img_filenames[i], cv2.cvtColor(car_images[i], cv2.COLOR_RGB2BGR))
    with open(csv_filenames[i], 'w') as file:
        writer = csv.writer(file)
        writer.writerow(row)
        file.close()
    os.chdir('..')
