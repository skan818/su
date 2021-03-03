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
flags.DEFINE_string('image', './data/girl.png', 'path to input image')
flags.DEFINE_string('output', './output.jpg', 'path to output image')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')
flags.DEFINE_string('img_dir', '', 'path to validation image directory')
flags.DEFINE_string('txt_output', '', 'path to text output file')
flags.DEFINE_string('labels', '', 'path to label file')


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

def bb_volume(pred):
    x_min = pred[0]
    y_min = pred[1]
    x_max = pred[2]
    y_max = pred[3]

    width = x_max - x_min
    height = y_max - y_min

    volume = width * height
    return volume

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

    output_file = open(FLAGS.txt_output, "a")

    if FLAGS.img_dir:
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
            output_file.write("Image: {}\n".format(basename))
            label_file = os.path.join(FLAGS.labels, "{}.txt".format(basename))
            try:
                bboxes = open(label_file, "r").readlines()

                logging.info('detections:')
                for i in range(nums[0]):
                    logging.info('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                                    np.array(scores[0][i]),
                                                    np.array(boxes[0][i])))
                    boxB = np.array(boxes[0][i]).tolist()
                    class_name = class_names[int(classes[0][i])]
                    boxA = []
                    if class_name == "embryo":
                        box = bboxes[0]
                        box = box[:-2].split()
                        width = float(box[3])
                        height = float(box[4])
                        half_width = width * 0.5
                        half_height = height * 0.5
                        x_center = float(box[1])
                        y_center = float(box[2])
                        xmin = x_center - half_width
                        boxA.append(xmin)
                        ymin = y_center - half_height
                        boxA.append(ymin)
                        xmax = x_center + half_width
                        boxA.append(xmax)
                        ymax = y_center + half_height
                        boxA.append(ymax)

                    else:
                        box = bboxes[1]
                        box = box[:-2].split()
                        width = float(box[3])
                        height = float(box[4])
                        half_width = width * 0.5
                        half_height = height * 0.5
                        x_center = float(box[1])
                        y_center = float(box[2])
                        xmin = x_center - half_width
                        boxA.append(xmin)
                        ymin = y_center - half_height
                        boxA.append(ymin)
                        xmax = x_center + half_width
                        boxA.append(xmax)
                        ymax = y_center + half_height
                        boxA.append(ymax)

                    iou = bb_iou(boxA, boxB)
                    output_file.write("IoU of {}: {}\n".format(class_name, iou))
                    volume = bb_volume(boxB)
                    output_file.write("Volume of {}: {}\n".format(class_name, volume))
            except FileNotFoundError:
                output_file.write("No detection found\n")
        output_file.close()

    else:
        img_raw = tf.image.decode_image(
            open(FLAGS.image, 'rb').read(), channels=3)
        img = tf.expand_dims(img_raw, 0)
        img = transform_images(img, FLAGS.size)

        t1 = time.time()
        boxes, scores, classes, nums = yolo(img)
        t2 = time.time()
        logging.info('time: {}'.format(t2 - t1))

        basename = os.path.splitext(FLAGS.image)[0]
        try:
            basename = basename.rsplit('/', 1)[-1]
        except:
            pass
        output_file.write("Image: {}\n".format(basename))
        label_file = os.path.join(FLAGS.labels, "{}.txt".format(basename))
        bboxes = open(label_file, "r").readlines()

        logging.info('detections:')
        for i in range(nums[0]):
            logging.info('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                            np.array(scores[0][i]),
                                            np.array(boxes[0][i])))
            boxB = np.array(boxes[0][i]).tolist()
            class_name = class_names[int(classes[0][i])]
            boxA = []
            if class_name == "embryo":
                box = bboxes[0]
                box = box[:-2].split()
                width = float(box[3])
                height = float(box[4])
                half_width = width * 0.5
                half_height = height * 0.5
                x_center = float(box[1])
                y_center = float(box[2])
                xmin = x_center - half_width
                boxA.append(xmin)
                ymin = y_center - half_height
                boxA.append(ymin)
                xmax = x_center + half_width
                boxA.append(xmax)
                ymax = y_center + half_height
                boxA.append(ymax)
            else:
                box = bboxes[1]
                box = box[:-2].split()
                width = float(box[3])
                height = float(box[4])
                half_width = width * 0.5
                half_height = height * 0.5
                x_center = float(box[1])
                y_center = float(box[2])
                xmin = x_center - half_width
                boxA.append(xmin)
                ymin = y_center - half_height
                boxA.append(ymin)
                xmax = x_center + half_width
                boxA.append(xmax)
                ymax = y_center + half_height
                boxA.append(ymax)
            iou = bb_iou(boxA, boxB)
            output_file.write("IoU of {}: {}\n".format(class_name, iou))
            volume = bb_volume(boxB)
            output_file.write("Volume of {}: {}\n".format(class_name, volume))
        output_file.close()

    


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
