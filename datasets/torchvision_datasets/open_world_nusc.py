import functools
import torch

import os
import tarfile
import collections
import logging
import copy
from torchvision.datasets import VisionDataset
import itertools

import torchvision.transforms as T
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image
from torchvision.datasets.utils import download_url, check_integrity, verify_str_arg

# OWOD nusc split

ALL_CLASS_NAMES = [
    'pedestrian', 'barrier', 'traffic_cone', 'bicycle', 'bus', 'car', 'truck', 'trailer', 'motorcycle', 'construction_vehicle'
]

T1_CLASS_NAMES = [
    'pedestrian', 'barrier', 'traffic_cone', 'bicycle', 'bus', 'car', 'truck', 'construction_vehicle'
]

T2_CLASS_NAMES = [
    'trailer', 'motorcycle'
]

UNK_CLASS = ["unknown"]

CAMERA_SENSOR = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']

NUSC_CLASS_NAMES = tuple(itertools.chain(T1_CLASS_NAMES, T2_CLASS_NAMES, UNK_CLASS))
# print(NUSC_CLASS_NAMES)

class OWNuscDetection(VisionDataset):
    def __init__(self,
                 args,
                 root,
                 version='v1.0-trainval',
                 image_sets='t1_train_new_split',
                 transform=None,
                 target_transform=None,
                 transforms=None,
                 no_cats=False,
                 filter_pct=-1):
        super(OWNuscDetection, self).__init__(root, transforms, transform, target_transform)
        self.images = []
        self.annotations = []
        self.imgids = []
        self.imgid2annotations = {}
        self.image_set = []

        self.CLASS_NAMES = NUSC_CLASS_NAMES
        self.MAX_NUM_OBJECTS = 64
        self.no_cats = no_cats
        self.args = args
        # import pdb;pdb.set_trace()

        for version, image_set in zip(version, image_sets):

            nusc_root = self.root
            annotation_dir = os.path.join(nusc_root, 'Annotations')
            image_dir = os.path.join(nusc_root, 'JEPGImages')

            if not os.path.isdir(nusc_root):
                raise RuntimeError('Dataset not found or corrupted.' +
                                   ' You can use download=True to download it')
            file_names = self.extract_fns(image_set, nusc_root)
            self.image_set.extend(file_names)

            self.images.extend([os.path.join(image_dir, x + ".jpg") for x in file_names])
            self.annotations.extend([os.path.join(annotation_dir, x + ".xml") for x in file_names])

            self.imgids.extend(self.convert_image_id(x, to_string=True) for x in file_names)
            self.imgid2annotations.update(dict(zip(self.imgids, self.annotations)))

        if filter_pct > 0:
            num_keep = float(len(self.imgids)) * filter_pct
            keep = np.random.choice(np.arange(len(self.imgids)), size=round(num_keep), replace=False).tolist()
            flt = lambda l: [l[i] for i in keep]
            self.image_set, self.images, self.annotations, self.imgids = map(flt, [self.image_set, self.images,
                                                                                   self.annotations, self.imgids])
        assert (len(self.images) == len(self.annotations) == len(self.imgids))

    @staticmethod
    def convert_image_id(img_id, to_integer=False, to_string=False, prefix=''):
        if to_integer:
            return int(prefix + img_id.replace('_', ''))
        if to_string:
            x = str(img_id)
            assert x.startswith(prefix)
            x = x[len(prefix):]
            if len(x) == 12 or len(x) == 6:
                return x
            return x[:4] + '_' + x[4:]

    @functools.lru_cache(maxsize=None)
    def load_instances(self, img_id):
        tree = ET.parse(self.imgid2annotations[img_id])
        target = self.parse_voc_xml(tree.getroot())
        image_id = target['annotation']['filename']
        instances = []
        for obj in target['annotation']['object']:
            cls = obj["name"]
            bbox = obj["bndbox"]
            bbox = [float(bbox[x]) for x in ["xmin", "ymin", "xmax", "ymax"]]
            bbox[0] -= 1.0
            bbox[1] -= 1.0
            # print(cls)
            instance = dict(
                category_id=NUSC_CLASS_NAMES.index(cls),
                bbox=bbox,
                area=(bbox[2] - bbox[0]) * (bbox[3] - bbox[1]),
                image_id=img_id
            )
            instances.append(instance)
        return target, instances

    def extract_fns(self, image_set, nusc_root):
        splits_dir = os.path.join(nusc_root, 'ImageSets')
        split_f = os.path.join(splits_dir, image_set.rstrip('\n') + '.txt')
        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]
        return file_names

    ### OWOD
    def remove_prev_class_and_unk_instances(self, target):
        # For training data. Removing earlier seen class objects and the unknown objects..
        prev_intro_cls = 0
        curr_intro_cls = 8
        valid_classes = range(prev_intro_cls, prev_intro_cls + curr_intro_cls)
        entry = copy.copy(target)
        for annotation in copy.copy(entry):
            if annotation["category_id"] not in valid_classes:
                entry.remove(annotation)
        return entry

    def remove_unknown_instances(self, target):
        # For finetune data. Removing the unknown objects...
        prev_intro_cls = 0
        curr_intro_cls = 8
        valid_classes = range(0, prev_intro_cls+curr_intro_cls)
        entry = copy.copy(target)
        for annotation in copy.copy(entry):
            if annotation["category_id"] not in valid_classes:
                entry.remove(annotation)
        return entry

    def label_known_class_and_unknown(self, target):
        # For test and validation data.
        # Label known instances the corresponding label and unknown instances as unknown.
        prev_intro_cls = 0
        curr_intro_cls = 8 
        total_num_class = 11 # 10 + 1
        known_classes = range(0, prev_intro_cls+curr_intro_cls)
        entry = copy.copy(target)
        for annotation in  copy.copy(entry):
        # for annotation in entry:
            if annotation["category_id"] not in known_classes:
                annotation["category_id"] = total_num_class - 1
        return entry

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is a dictionary of the XML tree.
        """
        image_set = self.transforms[0]
        img = Image.open(self.images[index]).convert('RGB')
        # print(self.images[index])
        target, instances = self.load_instances(self.imgids[index])
        if 'train' in image_set:
            instances = self.remove_prev_class_and_unk_instances(instances)
        elif 'test' in image_set:
            instances = self.label_known_class_and_unknown(instances)
        elif 'ft' in image_set:
            instances = self.remove_unknown_instances(instances)

        w, h = map(target['annotation']['size'].get, ['width', 'height'])
        target = dict(
            image_id=self.imgids[index],
            labels=torch.tensor([i['category_id'] for i in instances], dtype=torch.int64),
            area=torch.tensor([i['area'] for i in instances], dtype=torch.float32),
            boxes=torch.as_tensor([i['bbox'] for i in instances], dtype=torch.float32),
            orig_size=torch.as_tensor([int(h), int(w)]),
            size=torch.as_tensor([int(h), int(w)]),
            iscrowd=torch.zeros(len(instances), dtype=torch.uint8)
        )

        if self.transforms[-1] is not None:
            img, target = self.transforms[-1](img, target)
            
        return img, target

    def __len__(self):
        return len(self.images)

    def parse_voc_xml(self, node):
        voc_dict = {}
        children = list(node)
        if children:
            def_dic = collections.defaultdict(list)
            for dc in map(self.parse_voc_xml, children):
                for ind, v in dc.items():
                    def_dic[ind].append(v)
            if node.tag == 'annotation':
                def_dic['object'] = [def_dic['object']]
            voc_dict = {
                node.tag:
                    {ind: v[0] if len(v) == 1 else v
                     for ind, v in def_dic.items()}
            }
        if node.text:
            text = node.text.strip()
            if not children:
                voc_dict[node.tag] = text
        return voc_dict
    
    
if __name__ == '__main__':
    owod_path = '/home/hez4sgh/1_workspace/OW-DETR-nusc/data/OWDETR/Nuscenes'
    train_set = 't1_train_new_split'
    args = []
    dataset_train = OWNuscDetection(args, owod_path, version=['v1.0-trainval'], image_sets=[train_set])
    import pdb; pdb.set_trace()