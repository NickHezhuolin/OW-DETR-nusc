from pycocotools.coco import COCO
import numpy as np

# ALL_CLASS_NAMES = [
#     'pedestrain', 'barrier', 'trafficcone', 'bicycle', 'bus', 'car', 'truck', 'trailer', 'motorcycle', 'construction_vehicle'
# ]

# 5 + 5 : withour 'trafficcone', bus, motorcycle, 'truck', 'trailer'
T1_CLASS_NAMES = [
    'pedestrain', 'barrier', 'bicycle', 'car', 'construction_vehicle'
]

# 7 + 3 : withour motorcycle, 'truck', 'trailer'
# T2_CLASS_NAMES = [
#     'pedestrain', 'barrier', 'trafficcone', 'bicycle', 'bus', 'car', 'construction_vehicle'
# ]

# 9 + 1 : withour motorcycle
# T3_CLASS_NAMES = [
#     'pedestrain', 'barrier', 'trafficcone', 'bicycle', 'bus', 'car', 'truck', 'trailer', 'construction_vehicle'
# ]


# Train
nusc_coco_annotation_file = './data/OWDETR/Nuscenes/nuscenes_infos_train_mono3d.coco.json'
dest_file = './data/OWDETR/Nuscenes/ImageSets/t1_train_new_split.txt'

coco_instance = COCO(nusc_coco_annotation_file)

image_ids = []
cls = []
for index, image_id in enumerate(coco_instance.imgToAnns):
    image_details = coco_instance.imgs[image_id]
    classes = [coco_instance.cats[annotation['category_id']]['name'] for annotation in coco_instance.imgToAnns[image_id]]
    if not set(classes).isdisjoint(T1_CLASS_NAMES):
        image_ids.append(image_details['file_name'].split('/')[-1].split('.')[0])
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
nusc_coco_annotation_file = './data/OWDETR/Nuscenes/nuscenes_infos_val_mono3d.coco.json'
dest_file = './data/OWDETR/Nuscenes/ImageSets/t1_test_new_split.txt'

coco_instance = COCO(nusc_coco_annotation_file)

image_ids = []
cls = []
for index, image_id in enumerate(coco_instance.imgToAnns):
    image_details = coco_instance.imgs[image_id]
    classes = [coco_instance.cats[annotation['category_id']]['name'] for annotation in coco_instance.imgToAnns[image_id]]
    if not set(classes).isdisjoint(T1_CLASS_NAMES):
        image_ids.append(image_details['file_name'].split('/')[-1].split('.')[0])
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
