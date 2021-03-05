# Improving IVF success rates: a deep learning approach
> Embryo selection in IVF is a subjective process based on morphological assessment by embryologists. This project aims to use a deep learniing approach to automate embryo grading and predict foetal heartbeat from embryo time lapse images.

## Table of contents
* [Toy Classification Models](#toy-classification-models)
* [Object detection](#object-detection)
* [Image Selection](#image-selection)
* [Image Classification](#image-classification)

## Toy Classification Models
Made toy classification models classifying ships and bikes.

[Python notebooks](./ships_and_bikes/) contain:
* Initial training
* Visualising the convolutions
* Adding dropout
* Hyper-parameter tuning
* Data augmentation
* Plotting ROC curves

## Object detection
Two object detections models were trained to identify and localise embryos in time lapse images.

* YOLOv3
  - [Training script](./embryo/train_4766.py)
  - [Plots of size over time](./embryo/day_5_plots_yolo/)
* Mask-RCNN
  - [Training script](./embryo/)
  - [Plots of size over time](./embryo/day_5_plots_mask/)

## Image Selection
Three datasets were created using Mask-RCNN, which outperformed YOLOv3 in terms of accuracy, based on three selection criteria:
* [Method 1](./embryo/method_1.py): Selecting the image with greater size- Image at 6800 minutes vs Image of last detected embryo
* [Method 2](./embryo/method_2.py): Selecting an image at a fixed time point of 6800 minutes.
* [Method 3](./embryo/method_3.py): Selecting the image when the embryo reaches maximum size (after 6800 minutes)

## Image Classification
Transfer learning was used to train an Inceptionv3 model to predict foetal heartbeat (FHB), embryo quality, inner cell mass (ICM) quality and trophectoderm (TE) quality. Embryo quality is represented as two letters from A~C which assesses ICM quality and TE quality in that order. Models were trained separately for the three datasets and four outcomes.
* Predicting FHB
  - Method 1
    - [Training script](./embryo/final_fhb_1.py)
    - [Final model](./data/embryo/method_1/saved_model/fhb/)
  - Method 2
    - [Training script](./embryo/final_fhb_2.py)
    - [Final model](./data/embryo/method_2/saved_model/fhb/)
  - Method 3
    - [Training script](./embryo/final_fhb_3.py)
    - [Final model](./data/embryo/method_3/saved_model/fhb/)
* Predicting Embryo Quality
  - Method 1
    - [Training script](./embryo/final_top_1_grade_1.py)
    - [Final model](./data/embryo/method_1/grade/saved_model/top_1_grade/)
  - Method 2
    - [Training script](./embryo/final_top_1_grade_2.py)
    - [Final model](./data/embryo/method_2/grade/saved_model/top_1_grade/)
  - Method 3
    - [Training script](./embryo/final_top_1_grade_3.py)
    - [Final model](./data/embryo/method_3/grade/saved_model/top_1_grade/)
* Predicting ICM Quality
  - Method 1
    - [Training script](./embryo/final_first_grade_1.py)
    - [Final model](./data/embryo/method_1/grade/saved_model/first_grade/)
  - Method 2
    - [Training script](./embryo/final_first_grade_2.py)
    - [Final model](./data/embryo/method_2/grade/saved_model/first_grade/)
  - Method 3
    - [Training script](./embryo/final_first_grade_3.py)
    - [Final model](./data/embryo/method_3/grade/saved_model/first_grade/)
* Predicting TE Quality
  - Method 1
    - [Training script](./embryo/final_second_grade_1.py)
    - [Final model](./data/embryo/method_1/grade/saved_model/second_grade/)
  - Method 2
    - [Training script](./embryo/final_second_grade_2.py)
    - [Final model](./data/embryo/method_2/grade/saved_model/second_grade/)
  - Method 3
    - [Training script](./embryo/final_second_grade_3.py)
    - [Final model](./data/embryo/method_3/grade/saved_model/second_grade/)

## Note
\*All mentions of Method 1 in this repository are equivalent to Method 2 in the thesis text

\*All mentions of Method 2 in this repository are equivalent to Method 1 in the thesis text
