import os
os.chdir("/data/embryo/tfrecords/yolov3-tf2")

os.system("python eval_train.py \
--classes /data/embryo/random/classes.txt \
--weights /data/embryo/tfrecords/nick_model/yolov3-tf2/checkpoints/yolov3_best_model.tf \
--num_classes 2")