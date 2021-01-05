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

base_model = InceptionV3(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
predictions = Dense(3, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs= predictions)

for layer in base_model.layers:
    layer.trainable = False

model.summary()

model.compile(loss= 'categorical_crossentropy',
optimizer='RMSprop',
metrics = ['acc'])

callbacks = [
    ReduceLROnPlateau(verbose=1,patience=5,factor=0.1),
    ModelCheckpoint('/data/embryo/method_3/grade/checkpoints/second_grade.tf',
                    verbose=1, save_weights_only=True, save_best_only=True, monitor='val_acc'),
    TensorBoard(log_dir= '/data/embryo/method_3/grade/second_grade')
]

train_datagen = ImageDataGenerator(rescale=1/255)
validation_datagen = ImageDataGenerator(rescale=1/255)

train_df = pd.read_csv('/data/embryo/method_3/train_grade.csv', dtype= str)
validation_df = pd.read_csv('/data/embryo/method_3/validation_grade.csv', dtype= str)


train_generator = train_datagen.flow_from_dataframe(
    dataframe = train_df,
    directory= '/data/embryo/method_3/',
    x_col = 'id',
    y_col = 'Second',
    class_mode = 'categorical')

validation_generator = validation_datagen.flow_from_dataframe(
    dataframe = validation_df,
    directory = '/data/embryo/method_3/',
    x_col = 'id',
    y_col = 'Second',
    batch_size=7,
    class_mode = 'categorical')

history = model.fit_generator(
    train_generator,
    steps_per_epoch=9,
    epochs= 100,
    verbose=1,
    callbacks= callbacks,
    validation_data= validation_generator,
    validation_steps = 10
)