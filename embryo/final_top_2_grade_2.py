import os
import tensorflow as tf
import numpy as np
import albumentations as A
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd 
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import RMSprop, SGD, Adam
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
predictions = Dense(7, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs= predictions)

for layer in base_model.layers:
    layer.trainable = False

def random_90(image):
    transform = A.Compose([A.RandomRotate90()])
    new = transform(image=image)['image']
    return new

top_2 = tf.keras.metrics.TopKCategoricalAccuracy(
    k=2, name='top_2_accuracy', dtype=None
)
model.compile(loss= CategoricalCrossentropy(label_smoothing=0.2),
optimizer=SGD(learning_rate=0.0003),
metrics = [top_2])

callbacks = [
    ReduceLROnPlateau(verbose=1,patience=5,factor=0.1),
    EarlyStopping(patience=10, verbose=1, monitor='val_loss'),
    TensorBoard(log_dir= '/data/embryo/method_2/grade/final_top_2_grade')
]

train_datagen = ImageDataGenerator(rescale=1/255,
    preprocessing_function=random_90,
    horizontal_flip=True,
    brightness_range=[0.5,1.5])
validation_datagen = ImageDataGenerator(rescale=1/255)

train_df = pd.read_csv('/data/embryo/method_2/train_grade.csv', dtype= str)
validation_df = pd.read_csv('/data/embryo/method_2/validation_grade.csv', dtype= str)


train_generator = train_datagen.flow_from_dataframe(
    dataframe = train_df,
    directory= '/data/embryo/method_2/',
    x_col = 'id',
    y_col = 'label',
    batch_size=16,
    class_mode = 'categorical')

validation_generator = validation_datagen.flow_from_dataframe(
    dataframe = validation_df,
    directory = '/data/embryo/method_2/',
    x_col = 'id',
    y_col = 'label',
    batch_size=8,
    class_mode = 'categorical',
    shuffle=False)

history = model.fit_generator(
    train_generator,
    steps_per_epoch=304//16,
    epochs= 500,
    verbose=1,
    callbacks= callbacks,
    validation_data= validation_generator,
    validation_steps = 76//8
)

model.save('/data/embryo/method_2/grade/saved_model/top_2_grade')
_, accuracy = model.evaluate_generator(
    validation_generator,
    steps = 9)

print("val acc: ",accuracy)