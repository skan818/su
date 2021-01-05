import os
os.chdir("/data/embryo/models/research")

os.system("python final_max.py \
--pipeline_config /data/embryo/models/research/model.config \
--model_dir /data/embryo/mask_rcnn/data/best_model \
--ckpt best \
--output final_max_mask_4")