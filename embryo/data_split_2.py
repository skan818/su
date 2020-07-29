import os
import shutil
import random

random.seed(6114)
random_dir = "/data/embryo/random/"
more_dir = "/data/embryo/more/"
train_dir = "/data/embryo/tfrecords/train/"
test_dir = "/data/embryo/tfrecords/test/"

random_list = os.listdir(random_dir)
more_list = os.listdir(more_dir)
train_images = []
test_images = []

def extension(file):
    ext = os.path.splitext(file)[1]
    return ext
def embryo_id(file):
    id = file.rsplit("_" , 1)[0]
    return id
#all images in random dir to train/
for file in random_list:
    ext = extension(file)
    if ext == ".jpg":
        train_images.append(file)

#randomly split more_dir in half
more_images = []
id_list = []
train_id = []
test_id = []
for file in more_list:
    ext = extension(file)
    if ext == ".jpg":
        more_images.append(file)
for image in more_images:
    id = embryo_id(image)
    if id not in id_list:
        id_list.append(id)
counter = 0
while counter < 50:
    a = random.choice(id_list)
    if a not in train_id:
        train_id.append(a)
        counter += 1
for b in id_list:
    if b not in train_id:
        test_id.append(b)

#sort more_images into train and test
more_train_images = []
for file in train_id:
    for image in more_images:
        if embryo_id(image) == file:
            more_train_images.append(image)
for file in test_id:
    for image in more_images:
        if embryo_id(image) == file:
            test_images.append(image)

print("Train images: ", len(train_images))
print("More train images: ", len(more_train_images))
print("Test images: ", len(test_images))

# for image in train_images:
#     shutil.copy(random_dir + image, train_dir +image)
# for image in more_train_images:
#     shutil.copy(more_dir + image, train_dir + image)
for image in test_images:
    shutil.copy(more_dir + image, test_dir + image)

print("Train directory: ", len(os.listdir(train_dir)))
print("Test directory: ", len(os.listdir(test_dir)))
