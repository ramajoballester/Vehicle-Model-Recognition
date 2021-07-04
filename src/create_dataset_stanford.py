import numpy as np
import scipy as sp
import os
from PIL import Image
from utils import *
import argparse
import scipy.io
from shutil import copyfile

parser = argparse.ArgumentParser(description='Create Stanford dataset')
parser.add_argument('-crop', action='store_true', help='Create cropped images dataset')
parser.add_argument('-replace', action='store_true', help='Delete raw images')

parser.add_argument('-original_path', default='car_ims', help='Original images path')
parser.add_argument('-train_path', default=os.path.join('datasets', 'stanford196', 'car_ims', 'train'), help='Train images path')
parser.add_argument('-val_path', default=os.path.join('datasets', 'stanford196', 'car_ims', 'val'), help='Val images path')
parser.add_argument('-train_crop_path', default=os.path.join('datasets', 'stanford196', 'car_ims_crop', 'train'), help='Train cropped images path')
parser.add_argument('-val_crop_path', default=os.path.join('datasets', 'stanford196', 'car_ims_crop', 'val'), help='Val cropped images path')

parser.add_argument('-n_train_mini', default=7, help='Number of images in mini classes', type=int)
parser.add_argument('-n_val_mini', default=3, help='Number of images in mini classes', type=int)
parser.add_argument('-train_mini_path', default=os.path.join('datasets', 'stanford196', 'car_ims_mini', 'train'), help='Train mini images path')
parser.add_argument('-val_mini_path', default=os.path.join('datasets', 'stanford196', 'car_ims_mini', 'val'), help='Val mini images path')
parser.add_argument('-train_crop_mini_path', default=os.path.join('datasets', 'stanford196', 'car_ims_crop_mini', 'train'), help='Train mini cropped images path')
parser.add_argument('-val_crop_mini_path', default=os.path.join('datasets', 'stanford196', 'car_ims_crop_mini', 'val'), help='Val mini cropped images path')

args = parser.parse_args()


ROOT_DIR = get_git_root(os.getcwd())

ann_file = os.path.join(ROOT_DIR, 'cars_annos.mat')
ann_file = sp.io.loadmat(ann_file)

annotations = ann_file.get('annotations')[0]
labels = ann_file.get('class_names')[0]

for i in range(len(labels)):
    labels[i] = str(labels[i][0]).replace('/', '')

i_train = 0
i_test = 0
i_train_crop = 0
i_test_crop = 0

for each_row in annotations:
    img_file = each_row[0][0].split('/')[1]
    x_min = each_row[1][0][0]
    y_min = each_row[2][0][0]
    x_max = each_row[3][0][0]
    y_max = each_row[4][0][0]
    label_class = int(each_row[5][0][0])
    is_test = each_row[6][0][0]

    origin = os.path.join(ROOT_DIR, args.original_path, img_file)

    if args.crop:
        if is_test:
            os.makedirs(os.path.join(ROOT_DIR, args.val_crop_path, str(labels[label_class-1])), exist_ok=True)
            destination = os.path.join(ROOT_DIR, args.val_crop_path, str(labels[label_class-1]), img_file)
            i_test_crop += 1
        else:
            os.makedirs(os.path.join(ROOT_DIR, args.train_crop_path, str(labels[label_class-1])), exist_ok=True)
            destination = os.path.join(ROOT_DIR, args.train_crop_path, str(labels[label_class-1]), img_file)
            i_train_crop += 1

        img = Image.open(origin)
        img = img.resize(size=(x_max-x_min, y_max-y_min), box=(x_min, y_min, x_max, y_max))
        img.save(destination)


    if is_test:
        os.makedirs(os.path.join(ROOT_DIR, args.val_path, str(labels[label_class-1])), exist_ok=True)
        destination = os.path.join(ROOT_DIR, args.val_path, str(labels[label_class-1]), img_file)
        i_test += 1
    else:
        os.makedirs(os.path.join(ROOT_DIR, args.train_path, str(labels[label_class-1])), exist_ok=True)
        destination = os.path.join(ROOT_DIR, args.train_path, str(labels[label_class-1]), img_file)
        i_train += 1

    if args.replace:
        os.replace(origin, destination)
    else:
        copyfile(origin, destination)


for each_class in labels:
    if args.crop:
        os.makedirs(os.path.join(ROOT_DIR, args.train_crop_mini_path, each_class), exist_ok=True)
        os.makedirs(os.path.join(ROOT_DIR, args.val_crop_mini_path, each_class), exist_ok=True)
        img_files = os.listdir(os.path.join(ROOT_DIR, args.train_crop_path, each_class))
        img_files.sort()
        img_files = img_files[:args.n_train_mini]
        for each_file in img_files:
            origin = os.path.join(ROOT_DIR, args.train_crop_path, each_class, each_file)
            destination = os.path.join(ROOT_DIR, args.train_crop_mini_path, each_class, each_file)
            copyfile(origin, destination)

        img_files = os.listdir(os.path.join(ROOT_DIR, args.val_crop_path, each_class))
        img_files.sort()
        img_files = img_files[:args.n_val_mini]
        for each_file in img_files:
            origin = os.path.join(ROOT_DIR, args.val_crop_path, each_class, each_file)
            destination = os.path.join(ROOT_DIR, args.val_crop_mini_path, each_class, each_file)
            copyfile(origin, destination)


    img_files = os.listdir(os.path.join(ROOT_DIR, args.train_path, each_class))
    img_files.sort()
    img_files = img_files[:args.n_train_mini]
    os.makedirs(os.path.join(ROOT_DIR, args.train_mini_path, each_class), exist_ok=True)
    os.makedirs(os.path.join(ROOT_DIR, args.val_mini_path, each_class), exist_ok=True)
    for each_file in img_files:
        origin = os.path.join(ROOT_DIR, args.train_path, each_class, each_file)
        destination = os.path.join(ROOT_DIR, args.train_mini_path, each_class, each_file)
        copyfile(origin, destination)

    img_files = os.listdir(os.path.join(ROOT_DIR, args.val_path, each_class))
    img_files.sort()
    img_files = img_files[:args.n_val_mini]
    for each_file in img_files:
        origin = os.path.join(ROOT_DIR, args.val_path, each_class, each_file)
        destination = os.path.join(ROOT_DIR, args.val_mini_path, each_class, each_file)
        copyfile(origin, destination)


print('Copied %d train images' % i_train)
print('Copied %d val images' % i_test)
print('Copied %d cropped train images' % i_train_crop)
print('Copied %d cropped val images' % i_test_crop)
