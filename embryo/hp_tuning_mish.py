import os
os.chdir("/data/embryo/tfrecords/yolov3-tf2/")

os.system("python hparams.py \
--dataset /data/embryo/tfrecords/train_1241.tfrecords \
--val_dataset /data/embryo/tfrecords/test_1241.tfrecords \
--classes /data/embryo/random/classes.txt \
--logs logs/hparam_tuning_mish/ \
--num_classes 2 \
--mode fit \
--transfer darknet \
--epochs 1000 \
--weights /data/embryo/tfrecords/yolov3-tf2/checkpoints/yolov3.tf \
--weights_num_classes 80 \
--useMish True")