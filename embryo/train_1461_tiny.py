import os
os.chdir("/data/embryo/tfrecords/yolov3-tf2")

os.system("python train_NK.py \
--dataset /data/embryo/tfrecords/train_1461.tfrecords \
--val_dataset /data/embryo/tfrecords/test_1461.tfrecords \
--classes /data/embryo/random/classes.txt \
--num_classes 2 \
--mode fit \
--transfer darknet \
--batch_size 16 \
--size 416 \
--epochs 1000 \
--learning_rate 3e-4 \
--weights ./checkpoints/yolov3-tiny.tf \
--weights_num_classes 80 \
--useMish True \
--tiny True \
--logs 1461_images_tiny \
--checkpoint 1461_tiny")

