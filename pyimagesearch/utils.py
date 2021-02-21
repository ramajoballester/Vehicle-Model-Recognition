# import the necessary packages
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import os
import csv
import random
import copy
import cv2

def make_pairs(images, labels):
	# initialize two empty lists to hold the (image, image) pairs and
	# labels to indicate if a pair is positive or negative
	pairImages = []
	pairLabels = []

	# calculate the total number of classes present in the dataset
	# and then build a list of indexes for each class label that
	# provides the indexes for all examples with a given label
	numClasses = len(np.unique(labels))
	idx = [np.where(labels == i)[0] for i in range(0, numClasses)]

	# loop over all images
	for idxA in range(len(images)):
		# grab the current image and label belonging to the current
		# iteration
		currentImage = images[idxA]
		label = labels[idxA]

		# randomly pick an image that belongs to the *same* class
		# label
		idxB = np.random.choice(idx[label])
		posImage = images[idxB]

		# prepare a positive pair and update the images and labels
		# lists, respectively
		pairImages.append([currentImage, posImage])
		pairLabels.append([1])

		# grab the indices for each of the class labels *not* equal to
		# the current label and randomly pick an image corresponding
		# to a label *not* equal to the current label
		negIdx = np.where(labels != label)[0]
		negImage = images[np.random.choice(negIdx)]

		# prepare a negative pair of images and update our lists
		pairImages.append([currentImage, negImage])
		pairLabels.append([0])

	# return a 2-tuple of our image pairs and labels
	return (np.array(pairImages), np.array(pairLabels))

def euclidean_distance(vectors):
	# unpack the vectors into separate lists
	(featsA, featsB) = vectors

	# compute the sum of squared distances between the vectors
	sumSquared = K.sum(K.square(featsA - featsB), axis=1,
		keepdims=True)

	# return the euclidean distance between the vectors
	return K.sqrt(K.maximum(sumSquared, K.epsilon()))

def plot_training(H, plotPath):
	# construct a plot that plots and saves the training history
	plt.style.use("ggplot")
	plt.figure()
	plt.plot(H.history["loss"], label="train_loss")
	plt.plot(H.history["val_loss"], label="val_loss")
	plt.plot(H.history["accuracy"], label="train_acc")
	plt.plot(H.history["val_accuracy"], label="val_acc")
	plt.title("Training Loss and Accuracy")
	plt.xlabel("Epoch #")
	plt.ylabel("Loss/Accuracy")
	plt.legend(loc="lower left")
	plt.savefig(plotPath)
    

##########################################################
##### Own utils functions #####
##########################################################
def load_dataset(root_dir, labels, elements_per_class=15, training_split=0.75, img_resolution=(960,640), crop=True):
    os.chdir(os.path.join(root_dir, 'dataset'))
    trainX = []
    trainY = []
    testX = []
    testY = []
    trainX_bbox = []
    testX_bbox = []
    
    for each_dir in labels:
        os.chdir(each_dir)
        img_filenames = [file for file in os.listdir('./') if file.endswith('.jpg')]
        max_elements_per_class = len(img_filenames)
        if max_elements_per_class < elements_per_class:
            img_filenames = random.sample(img_filenames, max_elements_per_class)
        else:
            img_filenames = random.sample(img_filenames, elements_per_class)
        random.shuffle(img_filenames)
        n = len(img_filenames)
        split = round(training_split * n)
        train_img_filename = img_filenames[:split]
        test_img_filename = img_filenames[split:]
        
        for each_img in train_img_filename:
            img = cv2.imread(each_img)
            # OpenCV loads images to BGR by default (modify this to RGB)
#             img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            csv_filename = each_img.split('.')[0] + '.csv'
            with open(csv_filename) as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    trainX_bbox.append([int(row[0]), int(row[1]), int(row[2]), int(row[3])])
                    trainY.append(int(row[4]))
                csvfile.close()
            
            if crop:
                img = img[trainX_bbox[-1][1]-1:trainX_bbox[-1][3]-1, trainX_bbox[-1][0]-1:trainX_bbox[-1][2]-1]
            img = cv2.resize(img, img_resolution)
            trainX.append(img)
            
        for each_img in test_img_filename:
            img = cv2.imread(each_img)
            # OpenCV loads images to BGR by default (modify this to RGB)
#             img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            csv_filename = each_img.split('.')[0] + '.csv'
            with open(csv_filename) as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    testX_bbox.append([int(row[0]), int(row[1]), int(row[2]), int(row[3])])
                    testY.append(int(row[4]))
                csvfile.close()
            
            if crop:
                img = img[testX_bbox[-1][1]-1:testX_bbox[-1][3]-1, testX_bbox[-1][0]-1:testX_bbox[-1][2]-1]
            img = cv2.resize(img, img_resolution)
            testX.append(img)
        
        os.chdir('..')
        
    return (trainX, trainY), (testX, testY), (trainX_bbox, testX_bbox)


def normalize_labels(labels):
    norm_labels = copy.deepcopy(labels)
    uniq = np.unique(labels)
    sorted_uniq = np.sort(uniq)
    
    for i in range(len(sorted_uniq)):
        for j in range(len(labels)):
            if sorted_uniq[i] == labels[j]:
                norm_labels[j] = i
    
    return norm_labels