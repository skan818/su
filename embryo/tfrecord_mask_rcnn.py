import tensorflow as tf
import os

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

def create_tf_example(example):
  filename = example
  image_format = b'jpeg'
  encoded_image_data = open(main_dir + "/" + example , "rb").read()

  xmin_lst = [] 
  ymin_lst = []            
  xmax_lst = [] 
  ymax_lst = [] 

  label_dict = {1 : "embryo" , 2 : "zona"}           
  classes = []
  classes_text = []

  try:
    basename = os.path.splitext(example)[0]
    label_file = os.path.join(label_dir, "{}.txt".format(basename))
    boxes = open(label_file , "r").readlines()
    for box in boxes:
      box = box[:-2].split()
      half_width = float(box[3]) * 0.5
      half_height = float(box[4]) * 0.5
      x_center = float(box[1])
      y_center = float(box[2])
      xmin = x_center - half_width
      xmin_lst.append(xmin)
      ymin = y_center - half_height
      ymin_lst.append(ymin)
      xmax = x_center + half_width
      xmax_lst.append(xmax)
      ymax = y_center + half_height
      ymax_lst.append(ymax)
      classes.append(int(box[0])+1)
      classes_text.append(label_dict[int(box[0])])
  except:
    pass

  tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/filename': _bytes_feature(filename),
      'image/source_id': _bytes_feature(filename),
      'image/encoded' : _bytes_feature(encoded_image_data),
      'image/format': _bytes_feature(image_format),
      'image/object/bbox/xmin': _float_feature(xmin_lst),
      'image/object/bbox/ymin': _float_feature(ymin_lst),
      'image/object/bbox/xmax': _float_feature(xmax_lst),
      'image/object/bbox/ymax': _float_feature(ymax_lst),
      'image/object/class/text': _bytes_feature(classes_text),
      'image/object/class/label': _int64_feature(classes),
  }))
  return tf_example

tfrecords_dir = "/data/embryo/mask_rcnn"
train_dir = "/data/embryo/tfrecords/train"
test_dir = "/data/embryo/tfrecords/test"
label_dir = "/data/embryo/tfrecords/labels"
train_lst = os.listdir(train_dir)
test_lst = os.listdir(test_dir)

train_file = os.path.join(tfrecords_dir, "train.tfrecords")
with tf.io.TFRecordWriter(train_file) as writer:
    for example in train_lst:
        main_dir = train_dir
        tf_example = create_tf_example(example)
        writer.write(tf_example.SerializeToString())
    writer.close()


test_file = os.path.join(tfrecords_dir, "test.tfrecords")
with tf.io.TFRecordWriter(test_file) as writer:
    for example in test_lst:
        main_dir = test_dir
        tf_example = create_tf_example(example)
        writer.write(tf_example.SerializeToString())
    writer.close()