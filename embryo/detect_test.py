import os
os.chdir("/data/embryo/tfrecords/yolov3-tf2")
test_dir = "/data/embryo/tfrecords/test/"
test_list = os.listdir(test_dir)

for image in test_list:
    os.system("python detect.py \
    --classes /data/embryo/random/classes.txt \
    --num_classes 2 \
    --size 608 \
    --weights ./checkpoints/yolov3_train_best.tf \
    --image /data/embryo/tfrecords/test/{} \
    --output ./test_output_3/{}".format(image, image))
