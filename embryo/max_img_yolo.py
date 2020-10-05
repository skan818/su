import os
os.chdir("/data/embryo/tfrecords/yolov3-tf2")

os.system("python max_img.py \
--classes /data/embryo/random/classes.txt \
--weights ./checkpoints/2540_hp.tf \
--tiny True \
--embryo D2018_02_25_S00434_I0776_E7 \
--num_classes 2 \
--useMish True")