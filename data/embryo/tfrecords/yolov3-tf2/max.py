import os
import time
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import numpy as np
import tensorflow as tf
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images, load_tfrecord_dataset
from yolov3_tf2.utils import draw_outputs, broadcast_iou
from collections import namedtuple

flags.DEFINE_string('classes', './data/coco.names', 'path to classes file')
flags.DEFINE_string('weights', './checkpoints/yolov3.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('output', './output.jpg', 'path to output image')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')
flags.DEFINE_string('img_dir', '', 'path to directory with multiple embryo directories')
flags.DEFINE_string('single', '', 'path to one embryo directory')

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

    ##should be a variable with embryo name: output_file = open(FLAGS.txt_output, "a")

    if FLAGS.img_dir:
        embryos = os.listdir(FLAGS.img_dir)
        for embryo in embryos:
            if os.path.isdir(embryo):
                images = os.listdir(embryo)
                for image in images:
                    img_raw = tf.image.decode_image(
                        open(FLAGS.img_dir + embryo + "/" + image, 'rb').read(), channels=3)            
                    img = tf.expand_dims(img_raw, 0)
                    img = transform_images(img, FLAGS.size)
                    t1 = time.time()
                    boxes, scores, classes, nums = yolo(img)
                    t2 = time.time()
                    logging.info('time: {}'.format(t2 - t1))

                    basename = os.path.splitext(image)[0]
                    
                    logging.info('detections:')
                    for i in range(nums[0]):
                        logging.info('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                                        np.array(scores[0][i]),
                                                        np.array(boxes[0][i])))
                        bbox_pred = np.array(boxes[0][i]).tolist()
                        class_name = class_names[int(classes[0][i])]
                        volume = bb_volume(bbox_pred)



    else:
        images = os.listdir(FLAGS.single)
        results = np.zeros((0,3))
        for image in images:
            zona_size = []
            embryo_size = []
            img_raw = tf.image.decode_image(
                open(FLAGS.single + image, 'rb').read(), channels=3)
            img = tf.expand_dims(img_raw, 0)
            img = transform_images(img, FLAGS.size)

            t1 = time.time()
            boxes, scores, classes, nums = yolo(img)
            t2 = time.time()
            logging.info('time: {}'.format(t2 - t1))

            basename = os.path.splitext(image)[0]
            minutes = basename.rsplit('_', 1)[-1]

            logging.info('detections:')
            for i in range(nums[0]):
                logging.info('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                                np.array(scores[0][i]),
                                                np.array(boxes[0][i])))
                bbox_pred = np.array(boxes[0][i]).tolist()
                class_name = class_names[int(classes[0][i])]
                size = bb_size(bbox_pred)
                if class_name == "embryo":
                    embryo_size.append(size)
                else:
                    zona_size.append(size)
            if len(zona_size) <2:
                a = 0
                while a<2:
                    zona_size.append(0.0)
                    a +=1
            if len(embryo_size) <2:
                a = 0
                while a<2:
                    embryo_size.append(0.0)
                    a +=1
                
            results = np.append(results, [[float(minutes), zona_size[0], embryo_size[0]]], axis = 0)
            results = np.append(results, [[float(minutes), zona_size[1], embryo_size[1]]], axis = 0)
        max = (np.amax(results, axis = 0))[1]
        print("Max size: ", max)
        index = (np.where(results == max))[0][0]
        print("Index of max size: ", index)
        max_time = int(results[(44,0)])
        print("Time to max size: ", max_time)

    


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
