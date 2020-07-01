import os
os.chdir("/data/embryo/tfrecords/yolov3-tf2")

os.system("python train.py \
--dataset /data/embryo/tfrecords/train3.tfrecords \
--val_dataset /data/embryo/tfrecords/test3.tfrecords \
--classes /data/embryo/random/classes.txt \
--num_classes 2 \
--mode fit \
--transfer darknet \
--batch_size 16 \
--epochs 10 \
--weights ./checkpoints/yolov3.tf \
--weights_num_classes 80 \
--logs 10_epochs")