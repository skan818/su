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
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

model=tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(256, 256, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')                               
])
model.summary()

model.compile(loss= 'binary_crossentropy',
optimizer=RMSprop(lr=0.001),
metrics = ['acc'])

callbacks = [
    ReduceLROnPlateau(verbose=1,patience=5,factor=0.1),
    EarlyStopping(patience=10, verbose=1),
    ModelCheckpoint('/data/embryo/method_3/grade/checkpoints/first.tf',
                    verbose=1, save_weights_only=True, save_best_only=True),
    TensorBoard(log_dir= '/data/embryo/method_3/grade/logs')
]

train_datagen = ImageDataGenerator(rescale=1/255)
validation_datagen = ImageDataGenerator(rescale=1/255)

train_df = pd.read_csv('/data/embryo/method_3/train_grade.csv', dtype= str)
validation_df = pd.read_csv('/data/embryo/method_3/validation_grade.csv', dtype= str)


train_generator = train_datagen.flow_from_dataframe(
    dataframe = train_df,
    directory= '/data/embryo/method_3/',
    x_col = 'id',
    y_col = 'label',
    class_mode = 'categorical')

validation_generator = validation_datagen.flow_from_dataframe(
    dataframe = validation_df,
    directory = '/data/embryo/method_3/',
    x_col = 'id',
    y_col = 'label',
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