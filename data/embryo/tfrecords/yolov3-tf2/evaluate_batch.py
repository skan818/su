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
flags.DEFINE_string('img_dir', '', 'path to img directory')
flags.DEFINE_string('batch', '', 'single batch name')
flags.DEFINE_string('output', '', 'path to csv output file')

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

    if FLAGS.img_dir:
        filename_lst = []
        time_lst = []
        class_lst = []
        confidence_lst = []
        size_lst = []

        images = os.listdir(FLAGS.img_dir)
        for image in images:
            img_raw = tf.image.decode_image(
                open(FLAGS.img_dir + image, 'rb').read(), channels=3)            
            img = tf.expand_dims(img_raw, 0)
            img = transform_images(img, FLAGS.size)
            t1 = time.time()
            boxes, scores, classes, nums = yolo(img)
            t2 = time.time()
            logging.info('time: {}'.format(t2 - t1))

            basename = os.path.splitext(image)[0]
            embryo_id = basename.rsplit('_', 1)[0]
            minutes = basename.rsplit('_', 1)[-1]

            logging.info('detections:')
            for i in range(nums[0]):
                logging.info('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                                np.array(scores[0][i]),
                                                np.array(boxes[0][i])))
                bbox = np.array(boxes[0][i]).tolist()
                size = bb_size(bbox)
                class_name = class_names[int(classes[0][i])]
                confidence = scores[0][i]

                filename_lst.append(embryo_id)
                time_lst.append(minutes)
                class_lst.append(class_name)
                confidence_lst.append("{0:.4f}".format(confidence))
                size_lst.append("{0:.4f}".format(size))
    else:
        batch_lst = []
        id_lst = []
        time_lst = []
        class_lst = []
        confidence_lst = []
        size_lst = []

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
                id = embryo_id.rsplit('_', 1)[-1]
                minutes = basename.rsplit('_', 1)[-1]

                logging.info('detections:')
                for i in range(nums[0]):
                    logging.info('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                                    np.array(scores[0][i]),
                                                    np.array(boxes[0][i])))
                    bbox = np.array(boxes[0][i]).tolist()
                    size = bb_size(bbox)
                    class_name = class_names[int(classes[0][i])]
                    confidence = scores[0][i]

                    batch_lst.append(batch)
                    id_lst.append(id)
                    time_lst.append(minutes)
                    class_lst.append(class_name)
                    confidence_lst.append("{0:.4f}".format(confidence))
                    size_lst.append("{0:.4f}".format(size))

    data ={"Batch": batch_lst, "ID": id_lst, "Time": time_lst, "Class": class_lst, "Confidence": confidence_lst, "Size": size_lst}
    df = pd.DataFrame(data)
    df.to_csv("{}.csv".format(FLAGS.output))

        


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
