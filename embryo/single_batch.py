import os
os.chdir("/data/embryo/tfrecords/yolov3-tf2")

os.system("python evaluate_batch.py \
--classes /data/embryo/random/classes.txt \
--weights /data/embryo/tfrecords/nick_model/yolov3-tf2/checkpoints/yolov3_best_model.tf \
--num_classes 2 \
--batch D2018_02_25_S00434_I0776")