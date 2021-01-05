from pathlib import Path
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

train_datagen = ImageDataGenerator(rescale=1/255,
    rotation_range=50,
    horizontal_flip=True,
    brightness_range=[0.5,1.5],
    fill_mode='nearest')
validation_datagen = ImageDataGenerator(rescale=1/255)

train_generator = train_datagen.flow_from_directory(
    '/data/embryo/method_1/train/',
    batch_size=16,
    class_mode='binary'
)

labels = train_generator.class_indices
labels = dict((v,k) for k,v in labels.items())

model = tf.keras.models.load_model('/data/embryo/method_1/saved_model/fhb/')

example_list = ['D2016_10_19_S0128_I776_pdb_E2/D2016_10_19_S0128_I776_pdb_E2_6813.jpg',
'D2017_05_31_S0257_I776_pdb_E6/D2017_05_31_S0257_I776_pdb_E6_6820.jpg',
'D2018_07_20_S00263_I0831_D_pdb_E6/D2018_07_20_S00263_I0831_D_pdb_E6_6963.jpg',
'D2018_09_19_S00529_I0776_D_pdb_E5/D2018_09_19_S00529_I0776_E5_9223.jpg',
'D2019_05_11_S00603_I0776_D_pdb_E11/D2019_05_11_S00603_I0776_D_pdb_E11_6971.jpg']

for example in example_list:
    base_dir = '/data/embryo/base/'
    img_path = base_dir + example
    img = image.load_img(img_path, target_size=(224,224))
    img_array = image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis = 0)
    img_preprocessed = preprocess_input(img_batch)

    prediction = model.predict(img_preprocessed)[0][0]
    if prediction > 0.5:
        class_name = labels[1]
    else:
        class_name = labels[0]
    print(example)
    
    print(class_name, prediction)
