import os

print("Aug dir: ", len(os.listdir("/data/embryo/tfrecords/augmentations")))

print("Train dir: ", len(os.listdir("/data/embryo/tfrecords/train")))


print("Train more dir: ", len(os.listdir("/data/embryo/tfrecords/train_more/train")))

print("Label dir: ", len(os.listdir("/data/embryo/tfrecords/labels")))
