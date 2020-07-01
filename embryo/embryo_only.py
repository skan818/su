import os
os.chdir("/data/embryo/embryo_only")

os.system("python train.py \
--dataset /data/embryo/tfrecords/train_embryo_only.tfrecords \
--val_dataset /data/embryo/tfrecords/test_embryo_only.tfrecords \
--classes /data/embryo/tfrecords/embryo_only.txt \
--num_classes 1 \
--mode fit \
--transfer darknet \
--batch_size 32 \
--epochs 300 \
--weights ./checkpoints/yolov3.tf \
--weights_num_classes 80 \
--logs 300_epochs")