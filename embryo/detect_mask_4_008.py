import os
os.chdir("/data/embryo/models/research")

os.system("python detect.py \
--pipeline_config /data/embryo/models/research/model.config \
--model_dir /data/embryo/mask_rcnn/data/model-4-008 \
--ckpt ckpt-636 \
--batch D2018_02_25_S00434_I0776 \
--output model-4-008-636")