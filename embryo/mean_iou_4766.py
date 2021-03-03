import os
os.chdir("/data/embryo/tfrecords/yolov3-tf2")

os.system("python mean_iou.py \
--classes /data/embryo/classes.txt \
--weights ./checkpoints/4766.tf \
--num_classes 2 \
--useMish True \
--tiny True \
--img_dir /data/embryo/tfrecords/test/ \
--labels /data/embryo/tfrecords/labels/")