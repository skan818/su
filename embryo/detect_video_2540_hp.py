import os
os.chdir("/data/embryo/tfrecords/yolov3-tf2")

os.system("python detect_video.py \
--classes /data/embryo/random/classes.txt \
--weights ./checkpoints/2540_hp_2.tf \
--tiny True \
--size 416 \
--video /home/embryosu/git/su/embryo/video_2.avi \
--output /home/embryosu/git/su/embryo/output.avi \
--num_classes 2 \
--useMish True")
