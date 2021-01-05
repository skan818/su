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

class ResnetIdentityBlock(tf.keras.Model):
  def __init__(self, kernel_size, filters):
    super(ResnetIdentityBlock, self).__init__(name='')
    filters1, filters2, filters3 = filters

    self.conv2a = tf.keras.layers.Conv2D(filters1, (1, 1))
    self.bn2a = tf.keras.layers.BatchNormalization()

    self.conv2b = tf.keras.layers.Conv2D(filters2, kernel_size, padding='same')
    self.bn2b = tf.keras.layers.BatchNormalization()

    self.conv2c = tf.keras.layers.Conv2D(filters3, (1, 1))
    self.bn2c = tf.keras.layers.BatchNormalization()

  def call(self, input_tensor, training=False):
    x = self.conv2a(input_tensor)
    x = self.bn2a(x, training=training)
    x = tf.nn.relu(x)

    x = self.conv2b(x)
    x = self.bn2b(x, training=training)
    x = tf.nn.relu(x)

    x = self.conv2c(x)
    x = self.bn2c(x, training=training)

    x += input_tensor
    return tf.nn.relu(x)

base_model = InceptionV3(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
predictions = Dense(7, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs= predictions)

for layer in base_model.layers:
    layer.trainable = False

model.summary()

top_2 = tf.keras.metrics.TopKCategoricalAccuracy(
    k=2, name='top_2_accuracy', dtype=None
)
model.compile(loss= 'categorical_crossentropy',
optimizer='RMSprop',
metrics = ['acc', top_2])

callbacks = [
    ReduceLROnPlateau(verbose=1,patience=5,factor=0.1),
    ModelCheckpoint('/data/embryo/method_1/grade/checkpoints/top_2.tf',
                    verbose=1, save_weights_only=True, save_best_only=True, monitor='val_top_2_accuracy'),
    TensorBoard(log_dir= '/data/embryo/method_1/grade/top_2')
]

train_datagen = ImageDataGenerator(rescale=1/255)
validation_datagen = ImageDataGenerator(rescale=1/255)

train_df = pd.read_csv('/data/embryo/method_1/train_grade.csv', dtype= str)
validation_df = pd.read_csv('/data/embryo/method_1/validation_grade.csv', dtype= str)


train_generator = train_datagen.flow_from_dataframe(
    dataframe = train_df,
    directory= '/data/embryo/method_1/',
    x_col = 'id',
    y_col = 'label',
    class_mode = 'categorical')

validation_generator = validation_datagen.flow_from_dataframe(
    dataframe = validation_df,
    directory = '/data/embryo/method_1/',
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