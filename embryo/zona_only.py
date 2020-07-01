import os
os.chdir("/data/embryo/zona_only")

os.system("python train.py \
--dataset /data/embryo/tfrecords/train_zona_only.tfrecords \
--val_dataset /data/embryo/tfrecords/test_zona_only.tfrecords \
--classes /data/embryo/tfrecords/zona_only.txt \
--num_classes 1 \
--mode fit \
--transfer darknet \
--batch_size 16 \
--epochs 100 \
--weights ./checkpoints/yolov3.tf \
--weights_num_classes 80 \
--logs zona_only")