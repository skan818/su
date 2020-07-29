import os
import shutil
import random

random.seed(818)
main_dir = "/data/embryo/random/"
train_dir = "/data/embryo/tfrecords/train/"
test_dir = "/data/embryo/tfrecords/test_991/"

file_list = os.listdir(main_dir)

def extension(file):
    ext = os.path.splitext(file)[1]
    return ext
def embryo_id(file):
    id = file.rsplit("_" , 1)[0]
    return id

images = []
for file in file_list:
    ext = extension(file)
    if ext == ".jpg":
        images.append(file)
print("Total images: " , len(images))

id_list = []
train_id = []
test_id = []

for image in images:
    id = embryo_id(image)
    if id not in id_list:
        id_list.append(id)

n = 0
while n < 140:
    a = random.choice(id_list)
    if a not in train_id:
        train_id.append(a)
        n +=1

for b in id_list:
    if b not in train_id:
        test_id.append(b)

print("Training embryo IDs: " , len(train_id))
print("Testing embryo IDs: " , len(test_id))

train_lst = []
test_lst = []
for file in train_id:
    for image in images:
        if embryo_id(image) == file:
            train_lst.append(image)

for image in images:
    if image not in train_lst:
        test_lst.append(image)

print("Train images: ", len(train_lst))
print("Test images: ", len(test_lst))

# for image in train_lst:
#     shutil.copy(main_dir + image , train_dir + image)
for image in test_lst:
    shutil.copy(main_dir + image , test_dir + image)

print("Train folder: " , len(os.listdir(train_dir)))
print("Test folder: " , len(os.listdir(test_dir)))

    
