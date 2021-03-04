import os
import time
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images, load_tfrecord_dataset
from yolov3_tf2.utils import draw_outputs, broadcast_iou
from collections import namedtuple
from pandas import DataFrame

flags.DEFINE_string('classes', './data/coco.names', 'path to classes file')
flags.DEFINE_string('weights', './checkpoints/yolov3.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')
flags.DEFINE_string('batch', '', 'single batch name')

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

    #np array order: zona_size, embryo_size *2
    results = np.zeros((0,4))

    batch = FLAGS.batch
    img_dir = "/data/embryo/img/"
    img_dir_lst = os.listdir(img_dir)
    embryos = []
    for dir in img_dir_lst:
        if batch in dir:
            embryos.append(dir)
        else:
            pass

    for embryo in embryos:
        dir = img_dir + embryo
        lst = os.listdir(dir)
        for image in lst:
            zona_size = []
            embryo_size = []
            img_raw = tf.image.decode_image(
                open(dir + "/" + image, 'rb').read(), channels=3)            
            img = tf.expand_dims(img_raw, 0)
            img = transform_images(img, FLAGS.size)
            t1 = time.time()
            boxes, scores, classes, nums = yolo(img)
            t2 = time.time()
            logging.info('time: {}'.format(t2 - t1))

            basename = os.path.splitext(image)[0]
            embryo_id = basename.rsplit('_', 1)[0]

            logging.info('detections:')
            for i in range(nums[0]):
                logging.info('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                                np.array(scores[0][i]),
                                                np.array(boxes[0][i])))
                bbox = np.array(boxes[0][i]).tolist()
                class_name = class_names[int(classes[0][i])]
                if class_name == "embryo":
                    size = bb_size(bbox)
                    embryo_size.append(size)

                else:
                    size = bb_size(bbox)
                    zona_size.append(size)
            if len(zona_size) <2:
                a = 0
                while a<2:
                    zona_size.append(np.nan)
                    a +=1
            if len(embryo_size) <2:
                a = 0
                while a<2:
                    embryo_size.append(np.nan)
                    a +=1
            results = np.append(results, [[zona_size[0], embryo_size[0], zona_size[1], embryo_size[1]]], axis = 0)

    print(results.shape)

    primary = results[:, [0,1]]
    p_count = 0
    for row in primary:
        if np.count_nonzero(~np.isnan(row)) != 0:
            p_count +=1
    p_percentage = (p_count/results.shape[0])*100
    print(p_count)
    print("Percentage detected: ", p_percentage)

    secondary = results[:, [2,3]]
    s_count = 0
    for row in secondary:
        if np.count_nonzero(~np.isnan(row)) != 0:
            s_count +=1
    print(s_count)

    s_percentage = (s_count/results.shape[0])*100
    print("Percentage of secondary detections: ", s_percentage)


        


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
