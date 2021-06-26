import os
import argparse
import random
from utils import *
from shutil import copyfile

parser = argparse.ArgumentParser(description='Create VeRi-wild dataset')
parser.add_argument('-all_list', default=os.path.join('veri-wild', 'train_test_split', 'vehicle_info.txt'), help='Original all list')
parser.add_argument('-train_list', default=os.path.join('veri-wild', 'train_test_split', 'train_list_start0.txt'), help='Original train list')
parser.add_argument('-original_path', default=os.path.join('veri-wild', 'images'), help='Original images path')
parser.add_argument('-train_path', default=os.path.join('datasets', 'veri-wild', 'original', 'train'), help='Train images path')
parser.add_argument('-train_path_mini', default=os.path.join('datasets', 'veri-wild', 'mini', 'train'), help='Mini train images path')
parser.add_argument('-val_path', default=os.path.join('datasets', 'veri-wild', 'original', 'val'), help='Val images path')
parser.add_argument('-val_path_mini', default=os.path.join('datasets', 'veri-wild', 'mini', 'val'), help='Mini val images path')
parser.add_argument('-n_classes_mini', default=1000, help='Number of mini dataset classes')
parser.add_argument('-n_images_mini', default=10, help='Number of mini dataset images per class')
parser.add_argument('-mini_val_split', default=0.5, help='Validation split mini dataset')


args = parser.parse_args()


ROOT_DIR = get_git_root(os.getcwd())

train_img_names = []
train_class_ids = []
os.makedirs(os.path.join(ROOT_DIR, args.train_path), exist_ok=True)
os.makedirs(os.path.join(ROOT_DIR, args.val_path), exist_ok=True)

i = 0
train_list = os.path.join(ROOT_DIR, args.train_list)
with open(train_list, 'r') as file:
	for line in file.readlines():
		class_id = line.split('/')[0]
		img_name = line.split('/')[1].split(' ')[0]
		train_img_names.append(img_name)
		train_class_ids.append(class_id)
		os.makedirs(os.path.join(ROOT_DIR, args.train_path, str(class_id)), exist_ok=True)
		origin = os.path.join(ROOT_DIR, args.original_path, img_name)
		destination = os.path.join(ROOT_DIR, args.train_path, str(class_id), img_name)
		if os.path.exists(origin):
			os.replace(origin, destination)
			i += 1

	file.close()
	print('%d training images allocated' % (i))


i = 0
all_list = os.path.join(ROOT_DIR, args.all_list)
all_img_names = []
all_class_ids = []
with open(all_list, 'r') as file:
	for line in file.readlines():
		class_id = line.split('/')[0]
		img_name = line.split('/')[1].split(';')[0] + '.jpg'
		all_img_names.append(img_name)
		all_class_ids.append(class_id)

	all_zipped = zip(all_img_names, all_class_ids)
	train_zipped = zip(train_img_names, train_class_ids)

	val_zipped = list(set(all_zipped) - set(train_zipped))

	for each_img, each_class in val_zipped:
		os.makedirs(os.path.join(ROOT_DIR, args.val_path, str(each_class)), exist_ok=True)
		origin = os.path.join(ROOT_DIR, args.original_path, each_img)
		destination = os.path.join(ROOT_DIR, args.val_path, str(each_class), each_img)
		if os.path.exists(origin):
			os.replace(origin, destination)
			i += 1

	print('%d test images allocated' % (i))
	file.close()


# Mini datasets
mini_classes = os.listdir(os.path.join(ROOT_DIR, args.train_path))
mini_classes = random.sample(mini_classes, args.n_classes_mini)
i_train = 0
i_val = 0
for each_class in mini_classes:
	img_names = os.listdir(os.path.join(ROOT_DIR, args.train_path, each_class))
	random.shuffle(img_names)
	if len(img_names) > args.n_images_mini:
		img_names = random.sample(img_names, args.n_images_mini)

	train_img_names = img_names[:round(len(img_names)*(1-args.mini_val_split))]
	val_img_names = img_names[round(len(img_names)*(1-args.mini_val_split)):]
	os.makedirs(os.path.join(ROOT_DIR, args.train_path_mini, each_class), exist_ok=True)
	os.makedirs(os.path.join(ROOT_DIR, args.val_path_mini, each_class), exist_ok=True)
	for each_img_name in train_img_names:
		origin = os.path.join(ROOT_DIR, args.train_path, each_class, each_img_name)
		destination = os.path.join(ROOT_DIR, args.train_path_mini, each_class, each_img_name)
		copyfile(origin, destination)

	for each_img_name in val_img_names:
		origin = os.path.join(ROOT_DIR, args.train_path, each_class, each_img_name)
		destination = os.path.join(ROOT_DIR, args.val_path_mini, each_class, each_img_name)
		copyfile(origin, destination)

print('%d mini dataset train images allocated' % (i_train))
print('%d mini dataset val images allocated' % (i_val))
