import os
os.chdir("/data/embryo/tfrecords/yolov3-tf2")

os.system("python detect_SK.py \
--classes /data/embryo/random/classes.txt \
--weights ./checkpoints/yolov3_train_best.tf \
--size 608 \
--output ./detections/ \
--num_classes 2 \
--img_dir /data/embryo/tfrecords/test/")
