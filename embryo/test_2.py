import random
import shutil
import os

#selecting 200 random embryos
folder_list = os.listdir("/data/embryo/img/")
m = 200
folders = []
for j in range(m):
    b = random.choice(folder_list)
    folders.append(b)
    print(b)
    
#selecting 5 images per embryo
for folder in folders:
    src_dir = "/data/embryo/img/{}/".format(folder)
    dst_dir = "/data/embryo/random/"
    file_list = os.listdir(src_dir)
    n = 5
    for i in range(n):
        a = random.choice(file_list)
        shutil.copy(src_dir + a, dst_dir + a)

#checking how many images in random directory
random_dir=os.listdir("/data/embryo/random/")
print(len(random_dir))