import os
os.chdir("/data/embryo/tfrecords/yolov3-tf2")

os.system("python evaluate_pd.py \
--classes /data/embryo/random/classes.txt \
--weights /data/embryo/tfrecords/nick_model/yolov3-tf2/checkpoints/yolov3_best_model.tf \
--size 416 \
--num_classes 2 \
--img_dir /data/embryo/tfrecords/test/ \
--labels /data/embryo/tfrecords/labels")