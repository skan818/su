import os
os.chdir("/data/embryo/tfrecords/yolov3-tf2/")

os.system("python hparams.py \
--dataset /data/embryo/tfrecords/train_2540.tfrecords \
--val_dataset /data/embryo/tfrecords/test_2540.tfrecords \
--classes /data/embryo/random/classes.txt \
--logs logs/hparam_tuning_data_aug/ \
--num_classes 2 \
--mode fit \
--transfer darknet \
--epochs 1000 \
--weights /data/embryo/tfrecords/yolov3-tf2/checkpoints/yolov3-tiny.tf \
--weights_num_classes 80 \
--useMish True \
--tiny True")