import os
os.chdir("/data/embryo/tfrecords/yolov3-tf2")

os.system("python evaluate_np.py \
--classes /data/embryo/random/classes.txt \
--weights /data/embryo/tfrecords/nick_model/1241/yolov3_best_model.tf \
--size 608 \
--num_classes 2 \
--img_dir /data/embryo/tfrecords/test/ \
--labels /data/embryo/tfrecords/labels")