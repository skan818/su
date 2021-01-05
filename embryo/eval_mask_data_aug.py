#eval on test dataset
#mean iou after data augmentation
import os
os.chdir("/data/embryo/models/research")

os.system("python mean_iou.py \
--pipeline_config /data/embryo/models/research/model.config \
--model_dir /data/embryo/mask_rcnn/data/best_model \
--ckpt ckpt-631")