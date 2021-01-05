import os
import shutil

base_dir = "/data/embryo/day_5/"
embryos = os.listdir(base_dir)

for embryo in embryos:
    dir = os.path.join(base_dir, embryo)
    lst = os.listdir(dir)
    closest = 6800
    for image in lst:
        print(image)
        basename = os.path.splitext(image)[0]
        embryo_id = basename.rsplit('_', 1)[0]
        minutes = basename.rsplit('_', 1)[-1]
        id = embryo_id.rsplit('_', 1)[-1]
        batch = embryo_id.rsplit('_', 1)[0]

        time_diff = abs(6800 - int(minutes))
        if time_diff < closest:
            closest = time_diff
            closest_image = image
    
    new_dir = "/data/embryo/method_2"
    old_path = os.path.join(dir, closest_image)
    new_path = os.path.join(new_dir, closest_image)
    shutil.copy(old_path, new_path)