import os
os.chdir("/data/embryo/tfrecords/yolov3-tf2")

os.system("python evaluate.py \
--classes /data/embryo/random/classes.txt \
--weights ./checkpoints/yolov3_train_best.tf \
--size 608 \
--num_classes 2 \
--image /data/embryo/random/D2019_10_25_S00492_I3169_E1_1097.jpg \
--txt_output bbox_info.txt \
--labels /data/embryo/tfrecords/labels")