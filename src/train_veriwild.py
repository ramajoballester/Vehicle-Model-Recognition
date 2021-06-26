import tensorflow as tf
import os
import numpy as np
from utils import *
import datetime
from PIL import Image

mirrored_strategy = tf.distribute.MirroredStrategy()
with mirrored_strategy.scope():
    batch_size = 16
    dropout = 0.7
    ls = 0.0
    lr = 1e-4
    epochs = 1000

    ROOT_DIR = get_git_root(os.getcwd())
    train_path = os.path.join(ROOT_DIR, 'datasets', 'veri-wild', 'mini', 'train')
    val_path = os.path.join(ROOT_DIR, 'datasets', 'veri-wild', 'mini', 'val')
    train_gen = tf.keras.preprocessing.image.ImageDataGenerator()
    val_gen = tf.keras.preprocessing.image.ImageDataGenerator()
    train_gen = val_gen.flow_from_directory(train_path, target_size=(299,299),
                                              batch_size=batch_size)
    val_gen = val_gen.flow_from_directory(val_path, target_size=(299,299),
                                            batch_size=batch_size)

    # input_layer = tf.keras.layers.Input(shape=train_gen.next()[0][0].shape)
    # model = tf.keras.applications.EfficientNetB7(include_top=False,
    #                                              input_tensor=input_layer)
    # for each_layer in model.layers:
    #     each_layer.trainable = False
    #
    # output = model(input_layer)
    # output = AveragePooling2D((5,5), name='avg_pool')(output)
    # output = Dropout(dropout)(output)
    # output = Flatten(name='Flatten_1')(output)
    # output = Dense(1024, activation='relu', name='Dense_preoutput')(output)
    # output = Dropout(dropout)(output)
    # output = Dense(10001, activation='softmax', name='Dense_output')(output)
    #
    # model = tf.keras.models.Model(input_layer, output)

    model = tf.keras.models.load_model(os.path.join(ROOT_DIR, 'models', 'EfB3-full-299'))
    encoder = tf.keras.models.Model(model.input, model.get_layer('Dense_3').output)
    input_layer = tf.keras.layers.Input(shape=train_gen.next()[0][0].shape)
    output = encoder(input_layer)
    output = Dense(1000, activation='softmax', name='Dense_output')(output)
    veri_model = tf.keras.models.Model(input_layer, output)

    for each_layer in veri_model.layers[:-1]:
        each_layer.trainable = False


    loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=ls)
    metrics = 'categorical_accuracy'
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    TB_LOG_DIR, SAVE_MODELS_DIR, DATA_TRAIN_DIR = create_datetime_dirs(ROOT_DIR)
    tb_callback = TensorboardCallback(TB_LOG_DIR, metrics)
    ckpt_callback = tf.keras.callbacks.ModelCheckpoint(SAVE_MODELS_DIR, monitor='categorical_accuracy',
                                                        verbose=0, save_best_only=True,
                                                        save_weights_only=True, mode='auto',
                                                        save_freq='epoch')

    veri_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    veri_model.fit(train_gen, validation_data=val_gen, epochs=epochs, callbacks=[tb_callback, ckpt_callback], verbose=1)
