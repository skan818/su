#batch size 4
#lr 0.001
import os
os.chdir("~/data/embryo/models/research")

os.system("python object_detection/model_main_tf2.py \
--pipeline_config_path ~/data/embryo/mask_rcnn/data/model_2.config \
--model_dir ~/data/embryo/mask_rcnn/data/model-4-001 \
--alsologtostderr")