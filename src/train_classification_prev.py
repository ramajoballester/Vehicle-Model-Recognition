import argparse
import git
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from utils import *
from tensorflow.keras.layers import Input, Dense, Flatten, Concatenate, Dropout, AveragePooling2D
import datetime
from efficientnet.keras import EfficientNetB0, EfficientNetB3

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# TO-DO
# -data_cfg, -train_cfg as 'model'
# Implement last option data_cfg, train_cfg

parser = argparse.ArgumentParser(description='Main Vehicle Model Recognition training program')
parser.add_argument('-arch', default='VGG16_pretrained', help='Network architecture [VGG16_pretrained, VGG19]')
parser.add_argument('-batch_size', default='64', help='Batch size', type=int)
parser.add_argument('-data_augmentation', action='store_true', help='Data augmentation option')
parser.add_argument('-data_cfg', default=None, help='Data labels configuration file')
parser.add_argument('-dropout', default=0.7, help='Dropout rate in last layers', type=float)
parser.add_argument('-epochs', default='5000', help='Number of training epochs', type=int)
parser.add_argument('-img_resolution', default='224', help='Image resolution', type=int)
parser.add_argument('-lr', default='1e-4', help='Learning rate', type=float)
parser.add_argument('-ls', default='0.0', help='Label smoothing', type=float)
parser.add_argument('-loss', default='categorical_crossentropy', help='Loss function [binary_crossentropy, categorical_crossentropy, categorical_hinge, KLD, MSE]')
parser.add_argument('-metrics', default='categorical_accuracy', help='Metrics for visualization [binary_accuracy, categorical_accuracy]')
parser.add_argument('-model', default=None, help='Model path')
parser.add_argument('-multi_gpu', action='store_true', help='Use all available GPUs for training')
parser.add_argument('-n_classes', default='196', help='Number of different classes', type=int)
parser.add_argument('-n_elements', default='50', help='Number of different elements per class', type=int)
parser.add_argument('-optimizer', default='Adam', help='Optimizer for loss reduction')
parser.add_argument('-resume', action='store_true', help='Resume previous training')
parser.add_argument('-train_cfg', default=None, help='Load training configuration')


args = parser.parse_args()

gpus = tf.config.experimental.list_physical_devices('GPU')
try:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except RuntimeError as e:
    print(e)


ROOT_DIR = get_git_root(os.getcwd())


if args.resume:
    if not args.data_cfg:
        data_cfg_files = os.listdir(os.path.join(ROOT_DIR, 'cfg'))
        data_cfg_files.sort()
        args.data_cfg = os.path.join('cfg', data_cfg_files[-1])
    if not args.train_cfg:
        train_cfg_files = os.listdir(os.path.join(ROOT_DIR, 'cfg'))
        train_cfg_files.sort()
        args.train_cfg = os.path.join('cfg', train_cfg_files[-1])
    if not args.model:
        # Load last trained model
        pass

    # Load data_cfg
    files = os.listdir(os.path.join(ROOT_DIR, args.data_cfg))
    files = np.sort(files)
    if len(files) == 2:
        if files[0] == 'data_cfg.txt' and files[1] == 'train_cfg.txt':
            labels, args.n_elements = load_data_cfg(os.path.join(ROOT_DIR, args.data_cfg))
        else:
            raise Error(1)
    else:
        raise Error(1)

    # Load train_cfg
    files = os.listdir(os.path.join(ROOT_DIR, args.train_cfg))
    files = np.sort(files)
    if len(files) == 2:
        if files[0] == 'data_cfg.txt' and files[1] == 'train_cfg.txt':
            train_cfg = load_train_cfg(os.path.join(ROOT_DIR, args.train_cfg))
            args.arch = train_cfg[0]
            args.batch_size = train_cfg[1]
            args.lr = train_cfg[2]
            args.loss = train_cfg[3]
            args.metrics = train_cfg[4]
            args.optimizer = train_cfg[5]
        else:
            raise Error(2)
    else:
        raise Error(2)

else:
    if args.data_cfg:
        labels, args.n_elements = load_data_cfg(os.path.join(ROOT_DIR, args.data_cfg))
        args.n_classes = len(labels)
    else:
        car_names = os.listdir(os.path.join(ROOT_DIR, 'car_ims', 'train'))
        labels = random.sample(car_names, k = args.n_classes)

    if args.train_cfg:
        train_cfg = load_train_cfg(os.path.join(ROOT_DIR, args.train_cfg))
        args.arch = train_cfg[0]
        args.batch_size = train_cfg[1]
        args.lr = train_cfg[2]


print_args(args)

(trainX, trainY), (testX, testY), (trainX_bbox, testX_bbox) = load_dataset(ROOT_DIR, labels, args.n_elements, img_resolution=(args.img_resolution,args.img_resolution),
                                                                                 greyscale=False, random_sample = False)
trainX = np.asarray(trainX)
trainY = np.asarray(trainY)
testX = np.asarray(testX)
testY = np.asarray(testY)

trainX = np.expand_dims(trainX, axis=-1)
testX = np.expand_dims(testX, axis=-1)
norm_trainY = normalize_labels(trainY)
norm_testY = normalize_labels(testY)

input_shape = trainX.shape[1:4]

trainY = tf.one_hot(norm_trainY, len(np.unique(norm_trainY)))
testY = tf.one_hot(norm_testY, len(np.unique(norm_testY)))


if not args.model:
    if args.arch == 'VGG16A':
        model = build_vgg16(input_shape, embeddingDim=128, config='A')
    elif args.arch == 'VGG16D':
        model = build_vgg16(input_shape, embeddingDim=128, config='D')
    elif args.arch == 'VGG16E' or args.arch == 'VGG19':
        model = build_vgg16(input_shape, embeddingDim=128, config='E')
    elif args.arch == 'VGG16_pretrained':
        model = tf.keras.applications.VGG16(include_top=False, weights='imagenet',
                                    input_shape=input_shape, pooling=None)
        for each_layer in model.layers:
            each_layer.trainable = False

    elif args.arch == 'EfficientNetB0':
        model = EfficientNetB0(include_top=False, weights='imagenet',
                                input_shape=input_shape)


    else:
        # Other models, to be implemented
        pass

    # Own last layers
    input = Input(shape=input_shape)
    output = model(input)
    output = AveragePooling2D((5,5), name='avg_pool')(output)
    output = Dropout(args.dropout)(output)
    output = Flatten(name='Flatten_1')(output)
    output = Dense(4096, activation='relu', name='Dense_1')(output)
    output = Dropout(args.dropout)(output)
    output = Dense(1000, activation='relu', name='Dense_2')(output)

    # output = AveragePooling2D((5,5), name='avg_pool')(output)
    # output = AveragePooling2D((2,2), name='avg_pool')(output)
    # output = Flatten()(output)
    # output = Dropout(0.7)(output)

    output = Dense(args.n_classes, activation='softmax', name='Dense_output')(output)

    model = tf.keras.models.Model(input, output)


    # Optimizer
    if args.optimizer == 'SGD':
        optimizer = tf.keras.optimizers.SGD(learning_rate=args.lr)
    elif args.optimizer == 'Adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    elif args.optimizer == 'RMS':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=args.lr)
    else:
        raise Error(3)
        pass

    # Loss function
    if args.loss == 'binary_crossentropy':
        loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=args.ls)
    elif args.loss == 'categorical_crossentropy':
        loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=args.ls)
    elif args.loss == 'categorical_hinge':
        loss = tf.keras.losses.CategoricalHinge()
    elif args.loss == 'KLD':
        loss = tf.keras.losses.KLD
    elif args.loss == 'MSE':
        loss = tf.keras.losses.MSE
    else:
        raise Error(4)


    if args.metrics == 'binary_accuracy':
        metrics = tf.keras.metrics.BinaryAccuracy()
    elif args.metrics == 'categorical_accuracy':
        metrics = tf.keras.metrics.CategoricalAccuracy()
    else:
        raise Error(5)

    model.compile(loss=loss, optimizer=optimizer, metrics=[metrics])

else:
    tf.keras.models.load_model(args.model)



TB_LOG_DIR, SAVE_MODELS_DIR, DATA_TRAIN_DIR = create_datetime_dirs(ROOT_DIR)
if not args.data_cfg:
    save_data_cfg(DATA_TRAIN_DIR, labels, args.n_elements)
if not args.train_cfg:
    save_train_cfg(DATA_TRAIN_DIR, args)

tb_callback = TensorboardCallback(TB_LOG_DIR, args.metrics)
ckpt_callback = tf.keras.callbacks.ModelCheckpoint(SAVE_MODELS_DIR, monitor='val_categorical_accuracy',
                                                    verbose=0, save_best_only=True,
                                                    save_weights_only=False, mode='auto',
                                                    save_freq='epoch')

print(model.summary())


if args.data_augmentation:
    # dataAug = ImageDataGenerator(rotation_range=30, zoom_range=0.15,
    #                             width_shift_range=0.2, height_shift_range=0.2,
    #                             shear_range=0.15, horizontal_flip=True,
    #                             fill_mode="nearest")

    dataAug = ImageDataGenerator(rescale=1./255, rotation_range=20, zoom_range=0.15,
                                width_shift_range=0.1, height_shift_range=0.1,
                                shear_range=0.2, horizontal_flip=True,
                                brightness_range=[0.5, 1.5])

    trainAug = dataAug.flow(trainX[:,:,:,:,0], trainY, shuffle=False, batch_size=args.batch_size)

    model.fit(trainAug, validation_data=(testX, testY),
    	batch_size=args.batch_size, epochs=args.epochs,
        callbacks=[tb_callback], verbose=1)

else:
    model.fit(trainX, trainY, validation_data=(testX, testY),
    	batch_size=args.batch_size, epochs=args.epochs,
        callbacks=[tb_callback, ckpt_callback], verbose=1)
