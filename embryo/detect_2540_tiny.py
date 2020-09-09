import os
os.chdir("/data/embryo/tfrecords/yolov3-tf2")

os.system("python detect_SK.py \
--classes /data/embryo/random/classes.txt \
--weights ./checkpoints/2540_tiny.tf \
--size 416 \
--output ./detections_2540_tiny/ \
--num_classes 2 \
--useMish True \
--tiny True \
--img_dir /data/embryo/tfrecords/test/")
