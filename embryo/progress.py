import os
random_dir = os.listdir("/data/embryo/random/")
print("Length of random directory: " , len(random_dir))
progress = len(random_dir) - 992
print("Images annotated: " , progress)