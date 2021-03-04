import os
import time
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import numpy as np
import pandas as pd
import skimage
from skimage import io
import shutil
import tensorflow as tf
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images, load_tfrecord_dataset
from yolov3_tf2.utils import draw_outputs, broadcast_iou
from collections import namedtuple
from pandas import DataFrame
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

flags.DEFINE_string('classes', './data/coco.names', 'path to classes file')
flags.DEFINE_string('weights', './checkpoints/yolov3.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')
flags.DEFINE_string('output', '', 'path to csv output file')

def load_image_into_numpy_array(path):
    img = io.imread(path)
    (width, height) = img.shape
    if img.ndim !=3:
        img = skimage.color.gray2rgb(img)
    img_arr = np.array(img).reshape((width, height, 3)).astype(np.uint8)
    return img_arr

def bb_size(pred):
    x_min = pred[0]
    y_min = pred[1]
    x_max = pred[2]
    y_max = pred[3]

    width = x_max - x_min
    height = y_max - y_min

    size = width * height
    return size

def main(_argv):
    physical_devices = tf.config.experimental.list_physical_devices('CPU')
    for physical_device in physical_devices:
        #tf.config.experimental.set_memory_growth(physical_device, True)
        x=2

    if FLAGS.tiny:
        yolo = YoloV3Tiny(classes=FLAGS.num_classes)
    else:
        yolo = YoloV3(classes=FLAGS.num_classes)

    yolo.load_weights(FLAGS.weights).expect_partial()
    logging.info('weights loaded')

    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('classes loaded')

    batch_lst = []
    id_lst = []
    time_lst = []
    e_size_lst = []
    z_size_lst = []

    base_dir = "/data/embryo/base/"
    embryos = os.listdir(base_dir)

    for embryo in embryos:
        dir = os.path.join(base_dir, embryo)
        lst = os.listdir(dir)
        max_size = 0
        for image in lst:
            embryo_size = []
            zona_size = []

            image_path = os.path.join(dir, image)
            image_np = load_image_into_numpy_array(image_path)
            img = tf.convert_to_tensor(
                np.expand_dims(image_np, 0), dtype=tf.float32)
            img = transform_images(img, FLAGS.size)
            t1 = time.time()
            boxes, scores, classes, nums = yolo(img)
            t2 = time.time()
            logging.info('time: {}'.format(t2 - t1))

            basename = os.path.splitext(image)[0]
            embryo_id = basename.rsplit('_', 1)[0]
            minutes = basename.rsplit('_', 1)[-1]
            id = embryo_id.rsplit('_', 1)[-1]
            batch = embryo_id.rsplit('_', 1)[0]
            
            logging.info('detections:')
            for i in range(nums[0]):
                logging.info('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                                np.array(scores[0][i]),
                                                np.array(boxes[0][i])))
                bbox = np.array(boxes[0][i]).tolist()
                size = bb_size(bbox)
                class_name = class_names[int(classes[0][i])]
                if class_name == "embryo":
                    embryo_size.append("{0:.4f}".format(size))
                else:
                    zona_size.append("{0:.4f}".format(size))
                if size > max_size:
                    max_size = size
                    max_img = image
                else:
                    pass

            batch_lst.append(batch)
            id_lst.append(id)
            time_lst.append(minutes)

            try:
                e_size_lst.append(embryo_size[0])
            except IndexError:
                e_size_lst.append(np.nan)
            try:
                z_size_lst.append(zona_size[0])
            except IndexError:
                z_size_lst.append(np.nan)

        new_dir = "/data/embryo/yolo_max"
        old_path = os.path.join(dir, max_img)
        new_path = os.path.join(new_dir, max_img)
        shutil.copy(old_path, new_path)

    data ={"Batch": batch_lst, "ID": id_lst, "Time": time_lst, "Embryo": e_size_lst, "Zona": z_size_lst}
    df = pd.DataFrame(data)
    df.to_csv("{}.csv".format(FLAGS.output))

        


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
