import os
import shutil

dir_a = "/data/embryo/tfrecord/images/"
list_a = os.listdir(dir_a)
print(len(list_a))

train = open("/data/embryo/tfrecord/train.txt" , "a")
for a in range(800):
    text = "images/{}\n".format(list_a[a])
    train.write(text)
train.close()

test = open("/data/embryo/tfrecord/test.txt" , "a")
for b in range(800,991):
    text = "images/{}\n".format(list_a[b])
    test.write(text)
test.close()

