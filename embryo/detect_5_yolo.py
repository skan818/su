import os
os.chdir("/data/embryo/tfrecords/yolov3-tf2")

os.system("python detect_5.py \
--classes /data/embryo/classes.txt \
--weights ./checkpoints/4766.tf \
--size 416 \
--output ./detections_4766/ \
--num_classes 2 \
--useMish True \
--tiny True")