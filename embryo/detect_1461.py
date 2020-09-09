import os
os.chdir("/data/embryo/tfrecords/yolov3-tf2")

os.system("python detect_SK.py \
--classes /data/embryo/random/classes.txt \
--weights ./checkpoints/yolov3_best_model.tf \
--size 416 \
--output ./detections_1461/ \
--num_classes 2 \
--useMish True \
--img_dir /data/embryo/tfrecords/test/")
