import os
os.chdir("/data/embryo/tfrecords/yolov3-tf2")

os.system("python no_detection.py \
--classes /data/embryo/random/classes.txt \
--weights /data/embryo/tfrecords/yolov3-tf2/checkpoints/1461_images.tf \
--useMish True \
--num_classes 2 \
--batch D2018_02_25_S00434_I0776")