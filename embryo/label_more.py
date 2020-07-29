import os
import random
import shutil

random.seed(508)
embryo_dir = "/data/embryo/tfrecords/train_more/embryo/"
embryo_list = os.listdir(embryo_dir)
id_list = []
def get_id(image):
    id = image.rsplit("_",1)[0]
    return id
for image in embryo_list:
    id = get_id(image)
    if id not in id_list:
        id_list.append(id)

new_list = []

for id in id_list:
    list = []
    for img in embryo_list:
        if id in img:
            list.append(img)
    for x in range(5):
        a = random.choice(list)
        if a not in new_list:
            new_list.append(a)
print("New list: ",len(new_list))

new_dir = "/data/embryo/tfrecords/train_more/train/"

for image in new_list:
    shutil.copy(embryo_dir + image, new_dir + image)

print("New dir: ", len(os.listdir(new_dir)))