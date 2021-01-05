from pathlib import Path
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.models import load_model
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
 
model = tf.keras.models.load_model('/data/embryo/method_1/grade/saved_model/pred_grade/')

def load_image(filename):
	img = load_img(filename, target_size=(224, 224))
	img = img_to_array(img)
	img = img.reshape(1, 224, 224, 3)
	img = img.astype('float32')
	img = img - [123.68, 116.779, 103.939]
	return img

def run_example():
    img = load_image('/data/embryo/method_1/D2016_06_11_S0035_I776_pdb_E5_6796.jpg')
    result = model.predict_classes(img)
    print(result)

run_example()