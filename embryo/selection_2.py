import random
import shutil
import os
import re

#selecting 100 random embryos
random.seed(6234)
folder_list = os.listdir("/data/embryo/img/")
m = 100
counter = 0
folders = []
while counter < m:
    b = random.choice(folder_list)
    if b not in folders:
        folders.append(b)
        counter+=1

print("Unique embryo IDs: ", len(folders))

#selecting 5 images per embryo
for folder in folders:
    src_dir = "/data/embryo/img/{}/".format(folder)
    dst_dir = "/data/embryo/more/"
    file_list = os.listdir(src_dir)
    splt = []
    for file in file_list:
       splt.append(re.split("[_ .]" , file))

    n = 5
    x = 960
    y = 1920
   
    for i in range(n):
        list = []
        for file in splt:
            if int(file[6]) in range(x , y):
                list.append(file)
        try:
            a = random.choice(list)
            a = "_".join(a)
            a = a.rsplit("_" , 1)
            a = ".".join(a)
            shutil.copy(src_dir + a, dst_dir + a)
        except IndexError:
            print("Another error...")
            b = random.choice(file_list)
            shutil.copy(src_dir + b, dst_dir + b)
        x = x + 1440
        y = y + 1440


#checking how many images in new directory
new_dir=os.listdir("/data/embryo/more/")
print("New directory: ", len(new_dir))