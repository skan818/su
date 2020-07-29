import os
os.chdir("/data/embryo/tfrecords/yolov3-tf2")

os.system("python max.py \
--classes /data/embryo/random/classes.txt \
--weights ./checkpoints/yolov3_train_best.tf \
--size 608 \
--num_classes 2 \
--single /data/embryo/img/D2020_05_05_S00751_I3169_E1/")