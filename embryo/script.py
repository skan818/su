import tensorflow as tf
import os

from object_detection.utils import dataset_util


flags = tf.app.flags
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS


def create_tf_example(example):
  # TODO(user): Populate the following variables from your example.
  height = 608 
  width = 608 
  filename = example # Filename of the image. Empty if image is not from file
  image_format = b'jpeg' # b'jpeg' or b'png'

  x_center = [] 
  y_center = []            
  bbox_width = [] 
  bbox_height = [] 

  label_dict = {0 : "embryo" , 1 : "zona"}           
  classes = []
  classes_text = []

  try:
    
    label_file = "./labels/{}.txt".format(example[:-4])
    boxes = open(label_file , "r").readlines()
    for box in boxes:
      box = box[:-2].split()
      x_center.append(float(box[1]))
      y_center.append(float(box[2]))
      bbox_width.append(float(box[3]))
      bbox_height.append(float(box[4]))
      classes.append(int(box[0]))
      classes_text.append(label_dict[int(box[0])])
  except:
    classes_text.append("empty")
    classes.append(2)

  tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(filename),
      'image/source_id': dataset_util.bytes_feature(filename),
      'image/format': dataset_util.bytes_feature(image_format),
      'image/object/bbox/x_center': dataset_util.float_list_feature(x_center),
      'image/object/bbox/y_center': dataset_util.float_list_feature(y_center),
      'image/object/bbox/bbox_width': dataset_util.float_list_feature(bbox_width),
      'image/object/bbox/bbox_height': dataset_util.float_list_feature(bbox_height),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
  }))
  return tf_example


def main(_):
  writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

  # TODO(user): Write code to read in your dataset to examples 
  examples = os.listdir("/data/embryo/tfrecord/images/")
  for example in examples:
    tf_example = create_tf_example(example)
    writer.write(tf_example.SerializeToString())

  writer.close()


if __name__ == '__main__':
  tf.app.run()