import tensorflow as tf
import os

from object_detection.utils import dataset_util


flags = tf.app.flags
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS


def create_tf_example(example):
  # TODO(user): Populate the following variables from your example.
  height = 608 # Image height
  width = 608 # Image width
  filename = example # Filename of the image. Empty if image is not from file
  encoded_image_data = None # Encoded image bytes
  image_format = b'jpeg' # b'jpeg' or b'png'

  xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
  xmaxs = [] # List of normalized right x coordinates in bounding box
             # (1 per box)
  ymins = [] # List of normalized bottom y coordinates in bounding box (1 per box)
  ymaxs = [] # List of normalized top y coordinates in bounding box
             # (1 per box)
  classes_text = ["embryo" , "zona"] # List of string class name of bounding box (1 per box)
  classes = [0 , 1] # List of integer class id of bounding box (1 per box)

  label_file = "/data/embryo/tfrecord/labels/{}.txt".format(example[:-4])
  boxes = open(label_file , "r").readlines()
  for box in boxes:
    box = box[:-2].split()
        # numbers in label file already normalised
        # class , x-centre , y-centre , bbox-width , bbox-height
        xmins.append()
        xmaxs.append()
        ymins.append()
        ymaxs.append()
        classes_text.append(box['label'].encode())
        classes.append(int(LABEL_DICT[box['label']]))

  tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(filename),
      'image/source_id': dataset_util.bytes_feature(filename),
      'image/encoded': dataset_util.bytes_feature(encoded_image_data),
      'image/format': dataset_util.bytes_feature(image_format),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
  }))
  return tf_example


def main(_):
  writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

  # TODO(user): Write code to read in your dataset to examples 
  examples = os.listdir("data/embryo/tfrecord/images/")
  for example in examples:
    tf_example = create_tf_example(example)
    writer.write(tf_example.SerializeToString())

  writer.close()


if __name__ == '__main__':
  tf.app.run()