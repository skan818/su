import tensorflow as tf

train_records = sum(1 for _ in tf.data.TFRecordDataset('/data/embryo/mask_rcnn/train.tfrecords'))
test_records = sum(1 for _ in tf.data.TFRecordDataset('/data/embryo/mask_rcnn/test.tfrecords'))

print('Number of training images: {}'.format(train_records))
print('Number of test images: {}'.format(test_records))