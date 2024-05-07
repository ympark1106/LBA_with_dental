import os
import json as json_module
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor

from pycocotools.coco import COCO

root_path = "/home/gpu/Workspace/youmin/Learning-by-Asking/new_panorama_coco_dataset/images" 
json_path = "/home/gpu/Workspace/youmin/Learning-by-Asking/new_panorama_coco_dataset/annotations/instances.json"

def collate_fn(samples):
    
    images = []
    targets = []

    for image, target in samples:

        images.append(image)
        targets.append(target)
   

    return images, targets

class CocoDataset(Dataset):
    def __init__(self, json, root, train=False):
        self.root= root_path
        self.coco = COCO(json_path)
        self.image_ids = list(self.coco.imgs.keys())[204:]
        # print(self.image_ids)
        self.class_cate = [None]
        self.class_cate += [i['name'] for i in self.coco.cats.values()]
        self.class_id = [None]
        self.class_id += [i['id']+1 for i in self.coco.cats.values()]
        # print(self.class_id)
        self.cate2clsid = {cls_id:idx for idx, cls_id in enumerate(self.class_id)}
        self.clsid2cate = {v:k for k,v, in self.cate2clsid.items()}

        if train:
            self.transform = presets.DetectionPresetTrain(
                        data_augmentation="fixedscale", backend="pil", use_v2=False)
            
        else:
            self.transform = presets.DetectionPresetTrain(
                        data_augmentation="val", backend="pil", use_v2=False)
        
    

    def __len__(self):
        return len(self.image_ids)


    def get_item_by_id(self, image_id):
        if image_id in self.image_ids:
            index = self.image_ids.index(image_id)
            return self.__getitem__(index)
        else:
            raise ValueError(f"Image ID {image_id} not found in dataset.")


    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.coco.imgs[image_id]
        # print(image_id)
        path = image_info['file_name']
        image_path = os.path.join(self.root, image_info['file_name'])
        image = Image.open(image_path).convert("RGB")
        with open(json_path, 'r') as f:
            data = json_module.load(f)
        teeth_ids = [category['id'] for category in data['categories'] if '#' in category['name']]
        # print("teeth category_ids:", teeth_ids)

        ann = self.coco.loadAnns(self.coco.getAnnIds(image_id))
        ann = [obj for obj in ann if obj['category_id'] in teeth_ids]
        bboxes = [obj['bbox'] for obj in ann]
        bboxes = torch.as_tensor(bboxes, dtype=torch.float32).reshape(-1, 4)
        bboxes[:, 2:] += bboxes[:, :2]

        classes = [obj['category_id']+1 for obj in ann]
        # print(classes)
        classes = torch.tensor(classes, dtype=torch.int64)

        target = {}
        target["bboxes"] = bboxes
        target["labels"] = classes

        if self.transform:
            image = self.transform(image)


        return image, target


if __name__ == "__main__":
    s = CocoDataset(root = "/home/gpu/Workspace/youmin/Learning-by-Asking/new_panorama_coco_dataset/images", json = "/home/gpu/Workspace/youmin/Learning-by-Asking/new_panorama_coco_dataset/annotations/instances.json")

    i = 0
    # print(s.__getitem__(i)[1])
    # print()






