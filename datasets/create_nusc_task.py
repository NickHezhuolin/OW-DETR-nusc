from pycocotools.coco import COCO
import numpy as np


# 8+2 
T1_CLASS_NAMES = [
    'pedestrain', 'barrier', 'trafficcone', 'bicycle', 'bus', 'car', 'truck', 'trailer', 'motorcycle', ''
]



# Train
nusc_coco_annotation_file = 'C:\Users\Hezhuolin\Desktop\hzl_ws\0_data\nuscenes-mini\nuscenes_infos_train_mono3d.coco.json'
dest_file = '../../data/VOC2007/ImageSets/Main/t1_train_new_split.txt'

coco_instance = COCO(nusc_coco_annotation_file)

image_ids = []
cls = []
for index, image_id in enumerate(coco_instance.imgToAnns):
    image_details = coco_instance.imgs[image_id]
    classes = [coco_instance.cats[annotation['category_id']]['name'] for annotation in coco_instance.imgToAnns[image_id]]
    if not set(classes).isdisjoint(T1_CLASS_NAMES):
        image_ids.append(image_details['file_name'].split('.')[0])
        cls.extend(classes)

(unique, counts) = np.unique(cls, return_counts=True)
print({x:y for x,y in zip(unique, counts)})

c = 0 
for x,y in zip(unique, counts):
    if x in T1_CLASS_NAMES:
        c += y
print(c)
print(len(image_ids))

with open(dest_file, 'w') as file:
    for image_id in image_ids:
        file.write(str(image_id)+'\n')

print('Created train file')

# Test
nusc_coco_annotation_file = 'C:\Users\Hezhuolin\Desktop\hzl_ws\0_data\nuscenes-mini\nuscenes_infos_val_mono3d.coco.json'
dest_file = '../../data/VOC2007/ImageSets/Main/t1_test_new_split.txt'

coco_instance = COCO(nusc_coco_annotation_file)

image_ids = []
cls = []
for index, image_id in enumerate(coco_instance.imgToAnns):
    image_details = coco_instance.imgs[image_id]
    classes = [coco_instance.cats[annotation['category_id']]['name'] for annotation in coco_instance.imgToAnns[image_id]]
    if not set(classes).isdisjoint(T1_CLASS_NAMES):
        image_ids.append(image_details['file_name'].split('.')[0])
        cls.extend(classes)

(unique, counts) = np.unique(cls, return_counts=True)
print({x:y for x,y in zip(unique, counts)})

c = 0
for x,y in zip(unique, counts):
    if x in T1_CLASS_NAMES:
        c += y
print(c)
print(len(image_ids))

with open(dest_file, 'w') as file:
    for image_id in image_ids:
        file.write(str(image_id)+'\n')
print('Created test file')
