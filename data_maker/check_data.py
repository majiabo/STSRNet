import matplotlib.pyplot as plt
import json
import os
import cv2

json_root = '/mnt/disk_8t/kara/3DSR/data/label/10140018'
json_names = os.listdir(json_root)
names_dict = {}
for jname in json_names:
    path = os.path.join(json_root, jname)
    with open(path) as f:
        names = json.load(f)
    names_dict[jname] = names

counter = 0
imgs = []
for key, value in names_dict.items():
    name = value[2500]
    img = cv2.imread(name)[..., ::-1]
    if img is None:
        raise ValueError()
    imgs.append(img)
    counter += 1
    if counter>=4:
        break


plt.subplot(223)
plt.imshow(imgs[0])
plt.subplot(221)
plt.imshow(imgs[1])

plt.subplot(222)
plt.imshow(imgs[2])
plt.subplot(224)
plt.imshow(imgs[3])
plt.show()
