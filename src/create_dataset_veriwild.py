import os
from utils import *

ROOT_DIR = get_git_root(os.getcwd())
train_list = os.path.join(ROOT_DIR, 'veri-wild', 'train_test_split', 'train_list_start0.txt')
all_list = os.path.join(ROOT_DIR, 'veri-wild', 'train_test_split', 'vehicle_info.txt')

train_img_names = []
os.makedirs(os.path.join(ROOT_DIR, 'veri-wild', 'images', 'train'), exist_ok=True)

i = 0
with open(train_list, 'r') as file:
	for line in file.readlines():
		i += 1
		class_id = line.split('/')[0]
		img_name = line.split('/')[1].split(' ')[0]
		train_img_names.append(img_name)
		os.makedirs(os.path.join(ROOT_DIR, 'veri-wild', 'images', 'train', str(class_id)), exist_ok=True)
		origin = os.path.join(ROOT_DIR, 'veri-wild', 'images', img_name)
		destination = os.path.join(ROOT_DIR, 'veri-wild', 'images', 'train', str(class_id), img_name)
		os.replace(origin, destination)

	file.close()
	print('%d training images allocated' % (i))

os.makedirs(os.path.join(ROOT_DIR, 'veri-wild', 'images', 'val'), exist_ok=True)

i = 0
with open(all_list, 'r') as file:
	for line in file.readlines():
		class_id = line.split('/')[0]
		img_name = line.split('/')[1].split(';')[0] + '.jpg'
		if img_name not in train_img_names:
			i += 1
			os.makedirs(os.path.join(ROOT_DIR, 'veri-wild', 'images', 'val', str(class_id)), exist_ok=True)
			origin = os.path.join(ROOT_DIR, 'veri-wild', 'images', img_name)
			destination = os.path.join(ROOT_DIR, 'veri-wild', 'images', 'val', str(class_id), img_name)
			if os.path.exists(origin):
				os.replace(origin, destination)

	print('%d test images allocated' % (i))
	file.close()
