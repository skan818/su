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
predictions = Dense(3, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs= predictions)

for layer in base_model.layers:
    layer.trainable = False

def random_90(image):
    transform = A.Compose([A.RandomRotate90()])
    new = transform(image=image)['image']
    return new

model.compile(loss= CategoricalCrossentropy(label_smoothing=0.1),
optimizer=SGD(learning_rate=0.008),
metrics = ['acc'])

callbacks = [
    ReduceLROnPlateau(verbose=1,patience=5,factor=0.1),
    EarlyStopping(patience=10, verbose=1, monitor='val_loss'),
    TensorBoard(log_dir= '/data/embryo/method_1/grade/final_first_grade')
]

train_datagen = ImageDataGenerator(rescale=1/255,
    preprocessing_function=random_90,
    horizontal_flip=True,
    brightness_range=[0.5,1.5])
validation_datagen = ImageDataGenerator(rescale=1/255)

train_df = pd.read_csv('/data/embryo/method_1/train_grade.csv', dtype= str)
validation_df = pd.read_csv('/data/embryo/method_1/validation_grade.csv', dtype= str)


train_generator = train_datagen.flow_from_dataframe(
    dataframe = train_df,
    directory= '/data/embryo/method_1/',
    x_col = 'id',
    y_col = 'First',
    batch_size=32,
    class_mode = 'categorical')

validation_generator = validation_datagen.flow_from_dataframe(
    dataframe = validation_df,
    directory = '/data/embryo/method_1/',
    x_col = 'id',
    y_col = 'First',
    batch_size=16,
    class_mode = 'categorical',
    shuffle=False)

history = model.fit_generator(
    train_generator,
    steps_per_epoch=300//32,
    epochs= 500,
    verbose=1,
    callbacks= callbacks,
    validation_data= validation_generator,
    validation_steps = 80//16
)

predicted_classes = model.predict_generator(validation_generator, steps=80//16+1)
predicted_classes = np.argmax(predicted_classes, axis=1)
true_classes = validation_generator.classes
class_labels = list(validation_generator.class_indices.keys())
matrix = confusion_matrix(true_classes, predicted_classes)
print('Confusion matrix:')
print(matrix)

report = classification_report(true_classes, predicted_classes, target_names=class_labels)
print('Report:')
print(report)

model.save('/data/embryo/method_1/grade/saved_model/first_grade')