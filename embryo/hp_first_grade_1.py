import os
import tensorflow as tf
import pandas as pd 
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras import backend as K
from tensorboard.plugins.hparams import api as hp
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['RMSprop', 'Adam', 'SGD']))
HP_LR = hp.HParam('learning_rate', hp.Discrete([0.008, 0.001, 0.0003]))
HP_BATCH_SIZE_1 = hp.HParam('batch_size_1', hp.Discrete([8,16, 32]))
HP_BATCH_SIZE_2 = hp.HParam('batch_size_2', hp.Discrete([8,16]))

METRIC_ACCURACY = 'accuracy'

with tf.summary.create_file_writer('/data/embryo/method_1/grade/hp_first_grade').as_default():
  hp.hparams_config(
    hparams=[HP_OPTIMIZER, HP_LR, HP_BATCH_SIZE_1, HP_BATCH_SIZE_2],
    metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],
  )

def train_first_grade(hparams):
    base_model = InceptionV3(weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    predictions = Dense(3, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs= predictions)

    for layer in base_model.layers:
        layer.trainable = False

    model.summary()

    optimizer_name = hparams[HP_OPTIMIZER]
    lr = hparams[HP_LR]
    if optimizer_name == "RMSprop":
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr)
    elif optimizer_name == "Adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    elif optimizer_name == "SGD":
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr)

    model.compile(loss= 'categorical_crossentropy',
    optimizer=optimizer,
    metrics = ['acc'])

    callbacks = [
        ReduceLROnPlateau(verbose=1,patience=5,factor=0.1),
        EarlyStopping(patience=10, verbose=1),
        ModelCheckpoint('/data/embryo/method_1/grade/checkpoints/hp_first_grade.tf',
                        verbose=1, save_weights_only=True, save_best_only=True, monitor='val_acc'),
    ]

    train_datagen = ImageDataGenerator(rescale=1/255)
    validation_datagen = ImageDataGenerator(rescale=1/255)

    train_df = pd.read_csv('/data/embryo/method_1/train_grade.csv', dtype= str)
    validation_df = pd.read_csv('/data/embryo/method_1/validation_grade.csv', dtype= str)


    train_generator = train_datagen.flow_from_dataframe(
        dataframe = train_df,
        directory= '/data/embryo/method_1/',
        x_col = 'id',
        y_col = 'First',
        batch_size=hparams[HP_BATCH_SIZE_1],
        class_mode = 'categorical')

    validation_generator = validation_datagen.flow_from_dataframe(
        dataframe = validation_df,
        directory = '/data/embryo/method_1/',
        x_col = 'id',
        y_col = 'First',
        batch_size=hparams[HP_BATCH_SIZE_2],
        class_mode = 'categorical')

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=301//int(hparams[HP_BATCH_SIZE_1]),
        epochs= 100,
        verbose=1,
        callbacks= callbacks,
        validation_data= validation_generator,
        validation_steps = 80//int(hparams[HP_BATCH_SIZE_2])
    )
    _, accuracy = model.evaluate_generator(
        validation_generator,
        steps = 80//int(hparams[HP_BATCH_SIZE_2]))
    
    return accuracy

def run(run_dir, hparams):
  with tf.summary.create_file_writer(run_dir).as_default():
    hp.hparams(hparams)
    accuracy = train_first_grade(hparams)
    tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)

session_num = 0

for optimizer in HP_OPTIMIZER.domain.values:
    for learning_rate in HP_LR.domain.values:
        for batch_size_1 in HP_BATCH_SIZE_1.domain.values:
            for batch_size_2 in HP_BATCH_SIZE_2.domain.values:
                hparams = {
                    HP_OPTIMIZER: optimizer,
                    HP_LR: learning_rate,
                    HP_BATCH_SIZE_1: batch_size_1,
                    HP_BATCH_SIZE_2: batch_size_2
                }
                run_name = "run-{}".format(session_num)
                print('--- Starting trial: {}'.format(run_name))
                print({h.name: hparams[h] for h in hparams})
                run('/data/embryo/method_1/grade/hp_first_grade/' + run_name, hparams)
                session_num += 1