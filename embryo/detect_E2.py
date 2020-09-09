import os
os.chdir("/data/embryo/tfrecords/yolov3-tf2")

os.system("python detect_SK.py \
--classes /data/embryo/random/classes.txt \
--weights ./checkpoints/2540_hp_2.tf \
--size 416 \
--output ./detections_E2/ \
--num_classes 2 \
--useMish True \
--tiny True \
--img_dir /data/embryo/img/D2018_02_25_S00434_I0776_E2/")
