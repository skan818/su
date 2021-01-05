import os
os.chdir("/data/embryo/models/research")

os.system("python method_1.py \
--pipeline_config /data/embryo/models/research/model.config \
--model_dir /data/embryo/mask_rcnn/data/best_model \
--ckpt best")