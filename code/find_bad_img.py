"""
Find bad images and GTs in the dataset.
"""
from common import RESULTS_DIR, IMAGE_DIR
import os
import shutil


# find image's overlays
overlay_dir = IMAGE_DIR + 'stage1_train/overlays/'
# bbox score threshold for bad images or GTs
threshold = 0.6

with open(RESULTS_DIR + 'mask-rcnn-se-resnext50-train603-01/eval_train','r') as file:
    raw_file = file.readlines()

img_list = []
for img_str in raw_file:
    img_str = img_str.split('\t')
    img_idx = img_str[1]
    bbox_score = float(img_str[2].split('(')[1].rstrip(')\n'))
    img_list.append((img_idx, bbox_score))

img_sorted = sorted(img_list, key=lambda x: x[1])
os.makedirs(IMAGE_DIR + 'bad_images/', exist_ok=True)
with open(RESULTS_DIR + 'mask-rcnn-se-resnext50-train603-01/bad_img','w') as file:
    for img, score in img_sorted:
        if score <= threshold:
            file.write("%s\n" % img)
            shutil.copyfile(overlay_dir + '%s.png' % img, IMAGE_DIR + 'bad_images/%s.png' % img)
