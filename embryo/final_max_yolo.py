import os
os.chdir("/data/embryo/tfrecords/yolov3-tf2")

os.system("python final_max.py \
--classes /data/embryo/tfrecords/labels/classes.txt \
--weights ./checkpoints/4766.tf \
--tiny True \
--num_classes 2 \
--useMish True \
--output final_max_yolo")