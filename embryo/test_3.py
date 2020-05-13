import random
import shutil
import os
import re

#selecting 200 random embryos
random.seed(23432)
folder_list = os.listdir("/data/embryo/img/")
m = 200
folders = []
for j in range(m):
    b = random.choice(folder_list)
    folders.append(b)
    print(b)

#selecting the first image for each embryo
def f(x):
    convert = lambda text : int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    return sorted(x, key = alphanum_key)

for folder in folders:
    src_dir = "/data/embryo/img/{}/".format(folder)
    dst_dir = "/data/embryo/first/"
    file_list = f(os.listdir(src_dir))
    first = file_list[0]
    shutil.copy(src_dir + first, dst_dir + first)
    print(first)

#selecting 5 images per embryo
#for folder in folders:
#    src_dir = "/data/embryo/img/{}/".format(folder)
#    dst_dir = "/data/embryo/random/"
#    file_list = os.listdir(src_dir)
#    splt = []
#    for file in file_list:
#        splt.append(re.split("[_ .]" , file))
#
#    n = 5
#    x = 960
#    y = 1920
#    
#    for i in range(n):
#        list = []
#        for file in splt:
#            if int(file[6]) in range(x , y):
#                list.append(file)
#        try:
#            a = random.choice(list)
#            a = "_".join(a)
#            a = a.rsplit("_" , 1)
#            a = ".".join(a)
#            print("a is ", a)
#            shutil.copy(src_dir + a, dst_dir + a)
#        except IndexError:
#            print("Another error...")
#        x = x + 1440
#        y = y + 1440


#checking how many images in random directory
#random_dir=os.listdir("/data/embryo/random/")
#print(len(random_dir))