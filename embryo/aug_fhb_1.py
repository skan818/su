import os
import albumentations as A
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
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
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs= predictions)

for layer in base_model.layers:
    layer.trainable = False

def random_90(image):
    transform = A.Compose([A.RandomRotate90()])
    new = transform(image)
    return new

model.compile(loss= 'binary_crossentropy',
optimizer=SGD(learning_rate=0.0003),
metrics = ['acc'])

callbacks = [
    ReduceLROnPlateau(verbose=1,patience=5,factor=0.5),
    EarlyStopping(patience=10, verbose=1),
    ModelCheckpoint('/data/embryo/method_1/checkpoints/data_aug.tf',
                    verbose=1, save_weights_only=True, save_best_only=True, monitor='val_acc'),
    TensorBoard(log_dir= '/data/embryo/method_1/data_aug')
]

train_datagen = ImageDataGenerator(rescale=1/255,
    preprocessing_function=random_90,
    horizontal_flip=True,
    brightness_range=[0.5,1.5],
    fill_mode='nearest')
validation_datagen = ImageDataGenerator(rescale=1/255)

train_generator = train_datagen.flow_from_directory(
    '/data/embryo/method_1/train/',
    batch_size=16,
    class_mode='binary'
)

validation_generator = validation_datagen.flow_from_directory(
    '/data/embryo/method_1/validation/',
    batch_size=8,
    class_mode='binary',
    shuffle=False
)

history = model.fit_generator(
    train_generator,
    steps_per_epoch=307//16,
    epochs= 100,
    verbose=1,
    callbacks= callbacks,
    validation_data= validation_generator,
    validation_steps = 77//8
)
predicted_classes = model.predict_generator(validation_generator, steps=77//8+1)
true_classes = validation_generator.classes
class_labels = list(validation_generator.class_indices.keys())
matrix = confusion_matrix(true_classes, predicted_classes, labels = class_labels)
print('Matrix:')
print(matrix)

report = classification_report(true_classes, predicted_classes, target_names=class_labels)
print('Report:')
print(report)