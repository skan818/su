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
from six import BytesIO
from PIL import Image, ImageDraw, ImageFont
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
flags.DEFINE_string('batch', '', 'batch name')
flags.DEFINE_string('output', '', 'name of output csv file')
flags.DEFINE_boolean('test_dir', False, 'use test images')


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

def bb_iou(boxA, boxB):    
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou

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

    batch = FLAGS.batch
    labels = "/data/embryo/tfrecords/labels/"
    batch_lst = []
    id_lst = []
    time_lst = []
    embryo_size_lst = []
    zona_size_lst = []
    ratio_lst = []
    iou_lst = []

    embryos = []
    img_dir = "/data/embryo/img/"
    img_dir_lst = os.listdir(img_dir)
    for dir in img_dir_lst:
        if batch in dir:
            embryos.append(dir)
        else:
            pass

    for embryo in embryos:
        embryo_dir = os.path.join(img_dir, embryo)
        lst = os.listdir(embryo_dir)
        for image in lst:
            basename = os.path.splitext(image)[0]
            embryo_id = basename.rsplit('_', 1)[0]
            id = embryo_id.rsplit('_', 1)[-1]
            batch_name = embryo_id.rsplit('_', 1)[0]
            minutes = basename.rsplit('_', 1)[-1]
            embryo_size = []
            zona_size = []

            image_path = os.path.join(embryo_dir, image)
            print(image_path)
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
                    if class_name == 'embryo':
                        embryo_size.append("{0:.4f}".format(size))
                    elif class_name == 'zona':
                        zona_size.append("{0:.4f}".format(size))

            batch_lst.append(batch_name)
            id_lst.append(id)
            time_lst.append(minutes)
            try:
                embryo_size_lst.append(embryo_size[0])
            except IndexError:
                embryo_size_lst.append(np.nan)
            try:
                zona_size_lst.append(zona_size[0])
            except IndexError:
                zona_size_lst.append(np.nan)
            try:
                ratio = float(embryo_size[0]) / float(zona_size[0])
                ratio_lst.append("{0:.4f}".format(ratio))
            except IndexError:
                ratio_lst.append(np.nan)

    data ={"Batch": batch_lst, "ID": id_lst, "Time": time_lst, "Embryo": embryo_size_lst, "Zona": zona_size_lst, "Ratio": ratio_lst}
    df = pd.DataFrame(data)
    df.to_csv('{}.csv'.format(FLAGS.output))

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass

