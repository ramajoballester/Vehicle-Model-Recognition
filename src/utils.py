import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.layers import *
import matplotlib.pyplot as plt
import numpy as np
import os
import csv
import random
import copy
import cv2
import git
import datetime

def make_pairs(images, labels):
    pairImages = []
    pairLabels = []
    numClasses = len(np.unique(labels))
    idx = [np.where(labels == i)[0] for i in range(0, numClasses)]
    for idxA in range(len(images)):
        currentImage = images[idxA]
        label = labels[idxA]
        idxB = np.random.choice(idx[label])
        posImage = images[idxB]
        pairImages.append([currentImage, posImage])
        pairLabels.append([1])
        negIdx = np.where(labels != label)[0]
        negImage = images[np.random.choice(negIdx)]
        pairImages.append([currentImage, negImage])
        pairLabels.append([0])

    return (np.array(pairImages), np.array(pairLabels))


def euclidean_distance(vectors):
    featsA, featsB = vectors
    sumSquared = K.sum((K.square(featsA - featsB)), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sumSquared, K.epsilon()))


def plot_training(H, plotPath):
    plt.style.use('ggplot')
    plt.figure()
    plt.plot((H.history['loss']), label='train_loss')
    plt.plot((H.history['val_loss']), label='val_loss')
    plt.plot((H.history['accuracy']), label='train_acc')
    plt.plot((H.history['val_accuracy']), label='val_acc')
    plt.title('Training Loss and Accuracy')
    plt.xlabel('Epoch #')
    plt.ylabel('Loss/Accuracy')
    plt.legend(loc='lower left')
    plt.savefig(plotPath)


def load_dataset(root_dir, labels, elements_per_class=15, training_split=0.75, img_resolution=(960, 640), crop=True, greyscale=False, random_sample=False):
    os.chdir(os.path.join(root_dir, 'dataset'))
#     os.chdir(os.path.join(root_dir, 'veri-wild', 'images', 'train'))
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

        if random_sample:
            random.shuffle(img_filenames)
        else:
            img_filenames.sort()

        n = len(img_filenames)
        split = round(training_split * n)
        train_img_filename = img_filenames[:split]
        test_img_filename = img_filenames[split:]

        for each_img in train_img_filename:
            img = cv2.imread(each_img)
            if greyscale:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            csv_filename = each_img.split('.')[0] + '.csv'
            with open(csv_filename) as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    trainX_bbox.append([int(row[0]), int(row[1]), int(row[2]), int(row[3])])
                    trainY.append(int(row[4]))
                csvfile.close()

            if crop:
                img = img[trainX_bbox[(-1)][1] - 1:trainX_bbox[(-1)][3] - 1, trainX_bbox[(-1)][0] - 1:trainX_bbox[(-1)][2] - 1]
            img = cv2.resize(img, img_resolution)
            trainX.append(img)

        for each_img in test_img_filename:
            img = cv2.imread(each_img)
            if greyscale:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            csv_filename = each_img.split('.')[0] + '.csv'
            with open(csv_filename) as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    testX_bbox.append([int(row[0]), int(row[1]), int(row[2]), int(row[3])])
                    testY.append(int(row[4]))
                csvfile.close()

                if crop:
                    img = img[testX_bbox[(-1)][1] - 1:testX_bbox[(-1)][3] - 1, testX_bbox[(-1)][0] - 1:testX_bbox[(-1)][2] - 1]
                img = cv2.resize(img, img_resolution)
                testX.append(img)

        os.chdir('..')

    # Shuffles output
    tmp = list(zip(trainX, trainY, trainX_bbox))
    random.shuffle(tmp)
    trainX, trainY, trainX_bbox = zip(*tmp)

    tmp = list(zip(testX, testY, testX_bbox))
    random.shuffle(tmp)
    testX, testY, testX_bbox = zip(*tmp)

    return ((trainX, trainY), (testX, testY), (trainX_bbox, testX_bbox))


def normalize_labels(labels):
    norm_labels = copy.deepcopy(labels)
    uniq = np.unique(labels)
    sorted_uniq = np.sort(uniq)
    for i in range(len(sorted_uniq)):
        for j in range(len(labels)):
            if sorted_uniq[i] == labels[j]:
                norm_labels[j] = i

    return norm_labels


def build_siamese_model(inputShape, embeddingDim=48):
    inputs = Input(inputShape)
    x = Conv2D(64, (2, 2), padding='same', activation='relu')(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.3)(x)
    x = Conv2D(64, (2, 2), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=2)(x)
    x = Dropout(0.3)(x)
    pooledOutput = GlobalAveragePooling2D()(x)
    outputs = Dense(embeddingDim)(pooledOutput)
    model = tf.keras.models.Model(inputs, outputs)
    return model


def build_vgg16(inputShape, embeddingDim=128, config='D'):
    inputs = Input(inputShape)

    x = Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)
    if config >= 'D':
        x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2,2))(x)

    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    if config >= 'D':
        x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2,2))(x)

    x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    if config >= 'D':
        x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2,2))(x)

    x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    if config >= 'D':
        x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    if config >= 'E':
        x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2,2))(x)

    x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    if config >= 'D':
        x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    if config >= 'E':
        x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2,2))(x)

    x = Flatten()(x)
    x = Dense(4096, activation='relu')(x)
    x = Dense(4096, activation='relu')(x)

    output = Dense(embeddingDim)(x)
    model = tf.keras.models.Model(inputs, output, name='VGG16_conf' + config)

    return model


def get_git_root(path):
    git_repo = git.Repo(path, search_parent_directories=True)
    git_root = git_repo.git.rev_parse('--show-toplevel')
    return git_root


def load_data_cfg(path):
    labels = []
    with open(os.path.join(path, 'data_cfg.txt'), 'r') as label_file:
        first = True
        for row in label_file:
            if first:
                n_elements = int(row.split('\n')[0].split(' ')[1])
                first = False
            else:
                labels.append(row.split('\n')[0])

        label_file.close()

    return labels, n_elements


def load_train_cfg(path):
    with open(os.path.join(path, 'train_cfg.txt'), 'r') as train_cfg_file:
        for row in train_cfg_file:
            if row.split('\n')[0].split(' ')[0] == 'arch':
                arch = row.split('\n')[0].split(' ')[1]
            elif row.split('\n')[0].split(' ')[0] == 'batch_size':
                batch_size = int(row.split('\n')[0].split(' ')[1])
            elif row.split('\n')[0].split(' ')[0] == 'lr':
                lr = float(row.split('\n')[0].split(' ')[1])
            elif row.split('\n')[0].split(' ')[0] == 'loss':
                loss = float(row.split('\n')[0].split(' ')[1])
            elif row.split('\n')[0].split(' ')[0] == 'metrics':
                metrics = float(row.split('\n')[0].split(' ')[1])
            elif row.split('\n')[0].split(' ')[0] == 'optimizer':
                optimizer = float(row.split('\n')[0].split(' ')[1])
            elif row.split('\n')[0].split(' ')[0] == 'output':
                output = row.split('\n')[0].split(' ')[1]
        train_cfg_file.close()

    return [arch, batch_size, lr, loss, metrics, optimizer, output]


class Error(Exception):

    def __init__(self, message):
        if message == 0:
            self.message = 'Fatal error'
        if message == 1:
            self.message = '-data_cfg directory is not correct. Please introduce a -data_cfg directory of type ./cfg/training_data'
        if message == 2:
            self.message = '-train_cfg directory is not correct. Please introduce a -train_cfg directory of type ./cfg/training_data'
        if message == 3:
            self.message = '-optimizer is not correct. Please introduce a valid -optimizer option'
        if message == 4:
            self.message = '-loss function is not correct. Please introduce a valid -loss function'
        if message == 5:
            self.message = '-metrics is not correct. Please introduce a valid -metrics parameter'
        super().__init__(self.message)


def create_datetime_dirs(root_dir):
    date = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    tb_logs_dir = os.path.join(root_dir, 'trainings', 'tensorboard_logs', date)
    save_models_dir = os.path.join(root_dir, 'trainings', 'models', date)
    data_train_dir = os.path.join(root_dir, 'cfg', date)
    os.makedirs(tb_logs_dir)
    os.makedirs(save_models_dir)
    os.makedirs(data_train_dir)
    return (tb_logs_dir, save_models_dir, data_train_dir)


class TensorboardCallback(tf.keras.callbacks.Callback):

    def __init__(self, log_dir, metrics='categorical_accuracy'):
        self.writer = tf.summary.create_file_writer(log_dir)
        self.writer.set_as_default()
        self.metrics = metrics

    def on_epoch_end(self, epoch, logs=None):
        tf.summary.scalar('Train/Loss', logs['loss'], epoch)
        tf.summary.scalar('Train/Accuracy', logs[self.metrics], epoch)
        tf.summary.scalar('Val/Loss', logs['val_loss'], epoch)
        tf.summary.scalar('Val/Accuracy', logs['val_' + self.metrics], epoch)


def save_data_cfg(save_data_dir, labels, n_elements):
    with open(os.path.join(save_data_dir, 'data_cfg.txt'), 'w') as file:
        file.write('elements ' + str(n_elements) + '\n')
        for each_label in labels:
            file.write(each_label + '\n')
        file.close()


def save_train_cfg(save_train_dir, args):
    with open(os.path.join(save_train_dir, 'train_cfg.txt'), 'w') as file:
        file.write('arch ' + str(args.arch) + ' \n')
        file.write('batch_size ' + str(args.batch_size) + ' \n')
        file.write('lr ' + str(args.lr) + ' \n')
        file.write('loss ' + str(args.loss) + ' \n')
        file.write('metrics ' + str(args.metrics) + ' \n')
        file.write('optimizer ' + str(args.optimizer) + ' \n')
        file.close()

def print_args(args):
    print(' ')
    print('Starting training with parameters:')
    print('\t Architecture: ' + str(args.arch))
    print('\t Batch size: ' + str(args.batch_size))
    print('\t # Epochs: ' + str(args.epochs))
    print('\t Learning rate: ' + str(args.lr))
    print('\t Loss function: ' + str(args.loss))
    print('\t Metrics: ' + str(args.metrics))
    print('\t # Classes: ' + str(args.n_classes))
    print('\t # Elements: ' + str(args.n_elements))
    print('\t Optimizer: ' + str(args.optimizer))
