import os
import time
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import numpy as np
import pandas as pd
import shutil
import random
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
flags.DEFINE_string('output', './output.jpg', 'path to output image')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')

random.seed(330)


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

    embryo_lst = []
    id_lst = []
    train_dir = os.listdir("/data/embryo/tfrecords/train/")
    for file in train_dir:
        basename = file.rsplit('_', 1)[0]
        if basename not in id_lst:
            id_lst.append(basename)
        else:
            pass
    counter = 0
    while counter <50:
        x = random.choice(id_lst)
        if x not in embryo_lst:
            embryo_lst.append(x)
            counter +=1
    print(len(embryo_lst))

    image_names = []
    zona_sizes = []
    embryo_sizes = []

    for embryo in embryo_lst:
        img_dir = "/data/embryo/img/"
        embryo_dir = img_dir + embryo
        images = os.listdir(embryo_dir)
        for image in images:
            embryo_size = 0
            zona_size = 0
            image_path = embryo_dir + "/" + image
            img_raw = tf.image.decode_image(
                open(image_path, 'rb').read(), channels=3)            
            img = tf.expand_dims(img_raw, 0)
            img = transform_images(img, FLAGS.size)
            t1 = time.time()
            boxes, scores, classes, nums = yolo(img)
            t2 = time.time()
            logging.info('time: {}'.format(t2 - t1))

            logging.info('detections:')
            for i in range(nums[0]):
                logging.info('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                                np.array(scores[0][i]),
                                                np.array(boxes[0][i])))
                bbox = np.array(boxes[0][i]).tolist()
                size = bb_size(bbox)
                class_name = class_names[int(classes[0][i])]
                confidence = scores[0][i]
                if class_name == "embryo":
                    embryo_size = embryo_size + size
                else:
                    zona_size = zona_size + size
            image_names.append(image_path)
            zona_sizes.append(zona_size)
            embryo_sizes.append(embryo_size)
    
    data ={"Image": image_names, "Zona size": zona_sizes, "Embryo size": embryo_sizes}
    df = pd.DataFrame(data)
    destination_zona = "/data/embryo/tfrecords/train_more/zona/"
    destination_embryo = "/data/embryo/tfrecords/train_more/embryo/"
    for index, row in df.iterrows():
        a = row["Image"].rsplit('/', 1)[-1]
        if row["Zona size"] == 0:
            shutil.copy(row["Image"], destination_zona + a)
        elif row["Embryo size"] == 0:
            shutil.copy(row["Image"], destination_embryo + a)


    print("Zona dir: ", len(os.listdir(destination_zona)))
    print("Embryo dir: ", len(os.listdir(destination_embryo)))




if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
