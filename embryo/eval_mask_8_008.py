#eval on test dataset
#mean iou before hp tuning
import os
os.chdir("/data/embryo/models/research")

os.system("python mean_iou.py \
--pipeline_config /data/embryo/models/research/model.config \
--model_dir /data/embryo/mask_rcnn/data/model-8-008 \
--ckpt ckpt-318")