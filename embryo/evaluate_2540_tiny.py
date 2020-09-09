import os
os.chdir("/data/embryo/tfrecords/yolov3-tf2")

os.system("python evaluate_pd.py \
--classes /data/embryo/random/classes.txt \
--weights /data/embryo/tfrecords/yolov3-tf2/checkpoints/2540_tiny.tf \
--size 416 \
--num_classes 2 \
--useMish True \
--tiny True \
--img_dir /data/embryo/tfrecords/test/ \
--labels /data/embryo/tfrecords/labels \
--output 2540_tiny.csv")