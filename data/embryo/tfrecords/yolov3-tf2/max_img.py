import os
import time
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import numpy as np
import tensorflow as tf
import shutil
import re
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images, load_tfrecord_dataset
from yolov3_tf2.utils import draw_outputs

flags.DEFINE_string('classes', './data/coco.names', 'path to classes file')
flags.DEFINE_string('weights', './checkpoints/yolov3.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('embryo', '', 'embryo id')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')
flags.DEFINE_boolean('all', False, 'predict all')

def sorted_nicely( l ): 
    """ Sort the given iterable in the way that humans expect.""" 
    convert = lambda text: int(text) if text.isdigit() else text 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

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
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, True)

    if FLAGS.tiny:
        yolo = YoloV3Tiny(classes=FLAGS.num_classes)
    else:
        yolo = YoloV3(classes=FLAGS.num_classes)

    yolo.load_weights(FLAGS.weights).expect_partial()
    logging.info('weights loaded')

    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('classes loaded')


    if FLAGS.embryo:
        img_dir = "/data/embryo/img"
        embryo = FLAGS.embryo
        dir_path = os.path.join(img_dir, embryo)
        img_lst = os.listdir(dir_path)
        images = sorted_nicely(img_lst)
        max_size = 0

        for image in images:
            basename = os.path.splitext(image)[0]
            embryo_id = basename.rsplit('_', 1)[0]
            minutes = basename.rsplit('_', 1)[-1]
            if int(minutes) > 5000:
                img_raw = tf.image.decode_image(
                    open(os.path.join(dir_path, image), 'rb').read(), channels=3)
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
                    if class_name == "zona":
                        if size > 0.4:
                            if size > max_size:
                                max_size = size
                                max_img = image
                            else:
                                pass
                    last_img = image
            
        new_dir = "/data/embryo/max" 
        if max_size != 0:
            old_path = os.path.join(dir_path, max_img)
            new_path = os.path.join(new_dir, max_img)
        else:
            old_path = os.path.join(dir_path, last_img)
            new_path = os.path.join(new_dir, last_img)
        
        shutil.copy(old_path, new_path)
        print("Max: {}".format(new_path))
        


    


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
