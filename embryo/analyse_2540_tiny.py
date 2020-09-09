import os
os.chdir("/data/embryo/tfrecords/yolov3-tf2")

os.system("python size_over_time.py \
--classes /data/embryo/random/classes.txt \
--weights /data/embryo/tfrecords/yolov3-tf2/checkpoints/2540_tiny.tf \
--useMish True \
--tiny True \
--num_classes 2 \
--output size_over_time")