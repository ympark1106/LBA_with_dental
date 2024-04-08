import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os

import numpy as np
from PIL import Image

from pycocotools.coco import COCO
from pycocotools import mask as pymask


from loader import presets
from loader import transforms as T



def collate_fn(samples):
    
    images = []
    targets = []

    for image, target in samples:

        images.append(image)
        targets.append(target)
   

    return images, targets
    
    

class CocoDataset(data.Dataset):
    def __init__(self, root, json, train=False):
        self.root = root
        self.coco = COCO(json)
        self.train = train

        self.class_cate = [None]
        self.class_cate += [i['name'] for i in self.coco.cats.values()]
        
        
        self.class_id = [None]
        self.class_id += [i['id']+1 for i in self.coco.cats.values()]

        self.cate2clsid = {cls_id:idx for idx, cls_id in enumerate(self.class_id)}
        self.clsid2cate = {v:k for k,v, in self.cate2clsid.items()}
        

        img_indices = list(self.coco.imgs.keys())[205:]
        # self.img_indices = img_indices
        self.disease_label_list = [34, 35, 36, 37, 38, 39, 40, 41, 42]
        
        disease_indice = []
        # print(len(self.img_indices))
        for image_id in img_indices:
            anno = self.coco.loadAnns(self.coco.getAnnIds(image_id))
            
            img_meta = self.coco.imgs[image_id]
            w = img_meta['width']
            h = img_meta['height']
        
            anno = [obj for obj in anno if obj["iscrowd"] == 0]
                    
            pass_indice = [idx for idx, obj in  enumerate(anno) if obj["category_id"]+1 in self.disease_label_list]
            
            boxes = [obj["bbox"] for obj in anno]
            boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
            
            boxes[:, 2:] += boxes[:, :2]
            boxes[:, 0::2].clamp_(min=0, max=w)
            boxes[:, 1::2].clamp_(min=0, max=h)
            
            boxes = boxes[pass_indice]
            # print(boxes)
            # print()
            if boxes.tolist() == []:
                # print(boxes)
                continue

            keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
            if True not in keep:
                continue

            disease_indice.append(image_id)

        self.img_indices = disease_indice
        
        data_len = len(disease_indice)
        train_len = int(data_len * 0.7)
      
        
        
        
        # 4418 images
  
  
        if train:
            self.img_indices = self.img_indices[:train_len]
            self.transform = presets.DetectionPresetTrain(
                        data_augmentation="fixedscale", backend="pil", use_v2=False)
            
        else:
            self.img_indices = self.img_indices[train_len:]
            self.transform = presets.DetectionPresetTrain(
                        data_augmentation="val", backend="pil", use_v2=False)
            

   
    
    def convert_coco_poly_to_mask(self, segmentations, height, width):
        masks = []
        for polygons in segmentations:
            if polygons == []:
                mask = torch.zeros((height, width), dtype=torch.uint8)
                
            else:
                rles = pymask.frPyObjects(polygons, height, width)
                mask = pymask.decode(rles)
                if len(mask.shape) < 3:
                    mask = mask[..., None]
                mask = torch.as_tensor(mask, dtype=torch.uint8)
                mask = mask.any(dim=2)
    
            masks.append(mask)
            
        if masks:
            masks = torch.stack(masks, dim=0)
        else:
            masks = torch.zeros((0, height, width), dtype=torch.uint8)
        return masks
        

    def __getitem__(self, index):

        image_id = self.img_indices[index]
        # image_id = 299875487
        img_meta = self.coco.imgs[image_id]
        # print(image_id)
        path = img_meta['file_name']
        image = Image.open(os.path.join(self.root, 'images', path)).convert('RGB')
        w, h = image.size


        anno = self.coco.loadAnns(self.coco.getAnnIds(image_id))
        anno = [obj for obj in anno if obj["iscrowd"] == 0]
        
        
        pass_indice = [idx for idx, obj in enumerate(anno) if obj["category_id"]+1 in self.disease_label_list]
        
    
        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)
        
        classes = [obj["category_id"]+1 for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)
        
        
        # new_classes = []
        # for cls_idx in classes:
        #     # if cls_idx in self.disease_label_list:
        #     new_classes.append(cls_idx - 33)
                
        #     # else:
        #     #     new_classes.append(1)
        # classes = torch.tensor(new_classes, dtype=torch.int64)
        # print(classes)
        

        segmentations = [obj["segmentation"] for obj in anno]
        
        segmentations = [seg for idx, seg in enumerate(segmentations) if idx in pass_indice]
        boxes = boxes[pass_indice]
        classes = classes[pass_indice] - 33
 
        # classes = torch.ones_like(classes)
       
        keep_0 = [idx for idx, seg in enumerate(segmentations) if seg != []]
        segmentations = [seg for idx, seg in enumerate(segmentations) if idx in keep_0]
        boxes = boxes[keep_0]
        classes = classes[keep_0]

        
        keep_1 = [idx for idx, seg in enumerate(segmentations) if len(seg[0]) > 5]
        segmentations = [seg for idx, seg in enumerate(segmentations) if idx in keep_1]
        boxes = boxes[keep_1]
        classes = classes[keep_1]
        
        
        masks = self.convert_coco_poly_to_mask(segmentations, h, w)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        # print(boxes)
        # print(keep)
        boxes = boxes[keep]
        classes = classes[keep]
        masks = masks[keep]
        

            
        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["masks"] = masks
        
        if self.train:
            image, target = self.transform(image, target)
            return image, target

        else:
            image, target = self.transform(image, target)
            print(path)
            raw_image = Image.open(os.path.join(self.root, 'images', path)).convert('RGB')
        
            
            return image, target, raw_image 
        
      

    def __len__(self):
        return len(self.img_indices)



if __name__ == "__main__":
    s = CocoDataset(root="/mnt/d/Datasets/new_panorama_coco_dataset", 
                    json="/mnt/d/Datasets/new_panorama_coco_dataset/annotations/instances.json")

    i = 1000
    print(s.__getitem__(i)[1])
    print()