import os
os.chdir("/data/embryo/tfrecords/yolov3-tf2")

os.system("python mean_iou.py \
--classes /data/embryo/random/classes.txt \
--weights ./checkpoints/1461_images.tf \
--num_classes 2 \
--useMish True \
--img_dir /data/embryo/tfrecords/test/ \
--labels /data/embryo/tfrecords/labels/")