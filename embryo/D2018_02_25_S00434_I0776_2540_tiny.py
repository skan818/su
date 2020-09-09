import os
os.chdir("/data/embryo/tfrecords/yolov3-tf2")

os.system("python evaluate_batch.py \
--classes /data/embryo/random/classes.txt \
--weights /data/embryo/tfrecords/yolov3-tf2/checkpoints/2540_tiny.tf \
--useMish True \
--tiny True \
--num_classes 2 \
--batch D2018_02_25_S00434_I0776 \
--output D2018_02_25_S00434_I0776_2540_tiny")