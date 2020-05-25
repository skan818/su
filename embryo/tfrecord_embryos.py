import tensorflow as tf
import os
import numpy as np
import IPython.display as display

def _bytes_feature(value):
  if isinstance(value, str):
    value = value.encode('utf-8')
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  if isinstance(value, list):
    encoded_list = []
    for item in value:
      item = item.encode('utf-8')
      encoded_list.append(item)
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=encoded_list))
  else:
      return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  if isinstance(value, list):
      return tf.train.Feature(float_list=tf.train.FloatList(value=value))
  else:
      return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  if isinstance(value, list):
      return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
  else:
      return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

main_dir = "/data/embryo/random"

def create_tf_example(example):
  height = 608 
  width = 608 
  filename = example
  image_format = b'jpeg'

  x_center = [] 
  y_center = []            
  bbox_width = [] 
  bbox_height = [] 

  label_dict = {0 : "embryo" , 1 : "zona"}           
  classes = []
  classes_text = []

  try:
    basename = os.path.splitext(example)[0]
    label_file = os.path.join(main_dir, "{}.txt".format(basename))
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
      'image/height': _int64_feature(height),
      'image/width': _int64_feature(width),
      'image/filename': _bytes_feature(filename),
      'image/source_id': _bytes_feature(filename),
      'image/format': _bytes_feature(image_format),
      'image/object/bbox/x_center': _float_feature(x_center),
      'image/object/bbox/y_center': _float_feature(y_center),
      'image/object/bbox/bbox_width': _float_feature(bbox_width),
      'image/object/bbox/bbox_height': _float_feature(bbox_height),
      'image/object/class/text': _bytes_feature(classes_text),
      'image/object/class/label': _int64_feature(classes),
  }))
  return tf_example

record_file = 'embryo.tfrecords'
with tf.io.TFRecordWriter(record_file) as writer:
  list_all = os.listdir(main_dir)
  examples = []
  for file in list_all:
    ext = os.path.splitext(file)[1]
    if ext == '.jpg':
      examples.append(file)

  for example in examples:
    tf_example = create_tf_example(example)
    writer.write(tf_example.SerializeToString())
  writer.close()
