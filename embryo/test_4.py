#calculating bounding box volume
import re
src_file = open("/data/content/darknet/bikes/result/results.txt" , "r").readlines()
dst_file = open("/home/embryosu/volume.txt" , "a")
list_a = []
for line in src_file:
    if "%" in line:
        splt_a = re.split("[()]" , line)
        splt_b = splt_a[1].split()
        width = int(splt_b[5])
        height = int(splt_b[7])
        area = width * height
        new_text = line + "Volume of bounding box: {}\n".format(area)
        dst_file.write(new_text)
    else:
        dst_file.write(line)
dst_file.close()


test = open("/home/embryosu/volume.txt" , "r").readlines()
print(test[0:5])
