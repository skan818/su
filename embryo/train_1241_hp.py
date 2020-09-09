import os
os.chdir("/data/embryo/tfrecords/yolov3-tf2")

os.system("python train_NK.py \
--dataset /data/embryo/tfrecords/train_1241.tfrecords \
--val_dataset /data/embryo/tfrecords/test_1241.tfrecords \
--classes /data/embryo/random/classes.txt \
--num_classes 2 \
--mode fit \
--transfer darknet \
--batch_size 16 \
--size 416 \
--epochs 1000 \
--learning_rate 3e-4 \
--weights ./checkpoints/yolov3.tf \
--weights_num_classes 80 \
--useMish True \
--logs 1241_images \
--checkpoint 1241_best_params")