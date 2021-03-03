import matplotlib
import matplotlib.pyplot as plt

import os
import random
import skimage
from skimage import io
import imageio
from absl import app, flags, logging
from absl.flags import FLAGS
import glob
import scipy.misc
import numpy as np
import pandas as pd
import re
import shutil
from six import BytesIO
from PIL import Image, ImageDraw, ImageFont
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from IPython.display import display, Javascript
from IPython.display import Image as IPyImage

import tensorflow as tf

from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

flags.DEFINE_string('pipeline_config', './model.config', 'path to pipeline config file')
flags.DEFINE_string('model_dir', '', 'path to model directory')
flags.DEFINE_string('ckpt', '', 'latest checkpoint')
flags.DEFINE_string('output', '', 'path to csv output file')

def sorted_nicely( l ): 
    """ Sort the given iterable in the way that humans expect.""" 
    convert = lambda text: int(text) if text.isdigit() else text 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

def load_image_into_numpy_array(path):
    img = io.imread(path)
    (width, height) = img.shape
    if img.ndim !=3:
        img = skimage.color.gray2rgb(img)
    img_arr = np.array(img).reshape((width, height, 3)).astype(np.uint8)
    return img_arr

def get_model_detection_function(model):
  """Get a tf.function for detection."""

  @tf.function
  def detect_fn(image):
    """Detect objects in image."""

    image, shapes = model.preprocess(image)
    prediction_dict = model.predict(image, shapes)
    detections = model.postprocess(prediction_dict, shapes)

    return detections, prediction_dict, tf.reshape(shapes, [-1])

  return detect_fn

def bbsize(list):
    y_min = list[0]
    x_min = list[1]
    y_max = list[2]
    x_max = list[3]

    width = x_max - x_min
    height = y_max - y_min

    size = width * height
    return size

def main(_argv):
    pipeline_config = FLAGS.pipeline_config
    model_dir = FLAGS.model_dir
    configs = config_util.get_configs_from_pipeline_file(pipeline_config)
    model_config = configs['model']
    detection_model  = model_builder.build(model_config=model_config, is_training=False)

    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(os.path.join(model_dir, FLAGS.ckpt)).expect_partial()

    detect_fn = get_model_detection_function(detection_model)

    label_map_path = configs['eval_input_config'].label_map_path
    label_map = label_map_util.load_labelmap(label_map_path)
    categories = label_map_util.convert_label_map_to_categories(
        label_map,
        max_num_classes=label_map_util.get_max_label_map_index(label_map),
        use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    label_map_dict = label_map_util.get_label_map_dict(label_map, use_display_name=True)

    base_dir = "/data/embryo/day_6/"
    embryos = os.listdir(base_dir)

    for embryo in sorted_nicely(embryos):
        dir = os.path.join(base_dir, embryo)
        lst = os.listdir(dir)
        max_size = 0
        closest = 6800
        for image in sorted_nicely(lst):
            print(image)

            basename = os.path.splitext(image)[0]
            embryo_id = basename.rsplit('_', 1)[0]
            minutes = basename.rsplit('_', 1)[-1]
            id = embryo_id.rsplit('_', 1)[-1]
            batch = embryo_id.rsplit('_', 1)[0]

            image_path = os.path.join(dir, image)
            image_np = load_image_into_numpy_array(image_path)
            input_tensor = tf.convert_to_tensor(
                np.expand_dims(image_np, 0), dtype=tf.float32)

            detections, predictions_dict, shapes = detect_fn(input_tensor)

            label_id_offset = 1

            for i in range(2):
                list = detections['detection_boxes'][0].numpy()[i].tolist()
                size = bbsize(list)
                score = float(detections['detection_scores'][0].numpy().tolist()[i])
                class_int = (detections['detection_classes'][0].numpy() + label_id_offset).astype(int)[i]
                class_name = category_index[class_int]['name']
                if score > 0.5:
                    if class_name == 'zona':
                        if int(minutes) > 6800:
                            if size > max_size:
                                max_size = size
                                max_img = image
            time_diff = abs(6800 - int(minutes))
            if time_diff < closest:
                closest = time_diff
                closest_image = image

        new_dir = "/data/embryo/method_3_day_6"
        old_path = os.path.join(dir, max_img)
        new_path = os.path.join(new_dir, max_img)
        old_path_a = os.path.join(dir, closest_image)
        new_path_a = os.path.join(new_dir, closest_image)
        try:
            shutil.copy(old_path, new_path)
        except FileNotFoundError:
            shutil.copy(old_path_a, new_path_a)


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass