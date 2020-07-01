import os
os.chdir("/data/embryo/tfrecords/yolov3-tf2")

os.system("python train_NK.py \
--dataset /data/embryo/tfrecords/train_991.tfrecords \
--val_dataset /data/embryo/tfrecords/test3.tfrecords \
--classes /data/embryo/random/classes.txt \
--num_classes 2 \
--mode fit \
--transfer darknet \
--batch_size 32 \
--size 608 \
--epochs 5000 \
--learning_rate 0.00003 \
--weights ./checkpoints/yolov3.tf \
--weights_num_classes 80 \
--logs 991_images")