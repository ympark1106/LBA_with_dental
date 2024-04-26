import os
import json
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor

from pycocotools.coco import COCO

class TeethDataset(Dataset):
    def __init__(self, annotation_file, root_dir, transform=None, min_img_id=205):
        self.root_dir = root_dir
        self.transform = transform
        self.coco = COCO(annotation_file)
        self.image_ids = [img_id for img_id in self.coco.getImgIds() if img_id >= min_img_id]
        
        catIds = self.coco.getCatIds(supNms=['#'])
        self.teeth_ann_ids = self.coco.getAnnIds(catIds=catIds, iscrowd=None)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        image_info = self.coco.loadImgs(img_id)[0]
        image_path = os.path.join(self.root_dir, image_info['file_name'])
        image = Image.open(image_path).convert("RGB")
        
        ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=self.teeth_ann_ids, iscrowd=None)
        annotations = self.coco.loadAnns(ann_ids)
        bboxes = [ann['bbox'] for ann in annotations]  

        if self.transform:
            image = self.transform(image)

        bboxes = torch.as_tensor(bboxes, dtype=torch.float32)

        return image, bboxes

def split_dataset(annotation_file, root_dir, transform, test_size=0.15, val_size=0.15):
    dataset = TeethDataset(annotation_file, root_dir, transform=transform)
    train_data, test_data = train_test_split(dataset, test_size=test_size + val_size)
    test_data, val_data = train_test_split(test_data, test_size=val_size / (test_size + val_size))
    return train_data, val_data, test_data


def get_dataloaders(train_data, val_data, test_data, batch_size=4):
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


root_dir = "/home/gpu/Workspace/youmin/Learning-by-Asking/new_panorama_coco_dataset/images"  
annotation_file = "/home/gpu/Workspace/youmin/Learning-by-Asking/new_panorama_coco_dataset/annotations/instances.json"  
transform = ToTensor()  

train_data, val_data, test_data = split_dataset(annotation_file, root_dir, transform)
train_loader, val_loader, test_loader = get_dataloaders(train_data, val_data, test_data, batch_size=4)

