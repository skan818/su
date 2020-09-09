import os
os.chdir("/data/embryo/tfrecords/yolov3-tf2")

os.system("python add_ratio.py \
--classes /data/embryo/random/classes.txt \
--weights /data/embryo/tfrecords/yolov3-tf2/checkpoints/2540_tiny.tf \
--useMish True \
--tiny True \
--num_classes 2 \
--output ratio_over_time")