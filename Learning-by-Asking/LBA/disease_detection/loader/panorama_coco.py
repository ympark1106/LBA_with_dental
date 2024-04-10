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
        self.class_id += [i['id'] for i in self.coco.cats.values()]

        # print(self.class_id)
        # print(self.class_cate)

        self.remove_cate_id = self.class_filtering()
        self.class_cate = [cate for idx, cate in enumerate(self.class_cate) if self.class_id[idx] not in self.remove_cate_id]
        self.class_id = [idx for idx in self.class_id if idx not in self.remove_cate_id]
        
        # print(self.class_id)
        # print(self.class_cate)
       

        abstract = False
        if abstract:
            self.class_cate
            self.class_id

            self.cate2clsid = {0:None}
            for cate, cls_id in zip(self.class_cate[1:], self.class_id[1:]):
                if "#" in cate:
                    self.cate2clsid[cls_id] = 1

        
                elif cate == "보철&수복":
                    self.cate2clsid[cls_id] = 2

                elif cate == "근관치료":
                    self.cate2clsid[cls_id] = 3
                    
                elif cate == "치아우식증":
                    self.cate2clsid[cls_id] = 4
                    
                elif cate == "Lt.N":
                    self.cate2clsid[cls_id] = 5
                    
                elif cate == "Rt.N":
                    self.cate2clsid[cls_id] = 6
            
                
        
        else:
            self.cate2clsid = {cls_id:idx for idx, cls_id in enumerate(self.class_id)}
            self.clsid2cate = {v:k for k,v, in self.cate2clsid.items()}
        
        # print(self.class_id)
        # # print(self.clsid2cate)
        # exit()
        # print(self.cate2clsid)
        # exit()
        
        # print(self.remove_cate_id)
        
        img_indices = list(self.coco.imgs.keys())
        remove_cate_id = ["teeth3"]
        self.img_indices = [img_idx for img_idx in img_indices if img_idx not in remove_cate_id]
        
        
    
        if train:
            self.img_indices = self.img_indices[:400]
            self.transform = presets.DetectionPresetTrain(
                        data_augmentation="fixedscale", backend="pil", use_v2=False)
            
        else:
            self.img_indices = self.img_indices[400:]
            self.transform = presets.DetectionPresetTrain(
                        data_augmentation="val", backend="pil", use_v2=False)
          

    def class_filtering(self):
        
        remove_cate_id = []
        
        empty_class = ["C.CI", "잔존치근", "C.CII", "C.CIII", "치근단염증",
                            "함치성낭","치관주위염","과잉치","잔류낭","치근단낭"]   
        
        
        for i in self.coco.cats.values():
            if "@" in i['name']:
                remove_cate_id.append(i["id"])
                
            elif "teeth3" == i['name']:
                remove_cate_id.append(i["id"])
            
            elif "?" == i['name']:
                remove_cate_id.append(i["id"])
                
            elif "s" == i['name']:
                remove_cate_id.append(i["id"])
                
            elif "검사완료" == i['name']:
                remove_cate_id.append(i["id"])
                
            elif "검사완료" == i['name']:
                remove_cate_id.append(i["id"])
                
            elif "FD" == i['name']:
                remove_cate_id.append(i["id"])
                
            elif "#18-1" == i['name']:
                remove_cate_id.append(i["id"])
                
            elif "Mandibe" == i['name']:
                remove_cate_id.append(i["id"])

            # 신경
            # elif "Rt.N" == i['name']:
            #     remove_cate_id.append(i["id"])
                
            # elif "Lt.N" == i['name']:
            #     remove_cate_id.append(i["id"])
            
            # 하악
            elif "U.Bone" == i['name']:
                remove_cate_id.append(i["id"])
                
            elif "L.Bone" == i['name']:
                remove_cate_id.append(i["id"])
            
            # 상악
            elif "Rt.Sinus" == i['name']:
                remove_cate_id.append(i["id"])
                
            elif "Lt.Sinus" == i['name']:
                remove_cate_id.append(i["id"])
                
            elif i['name'] in empty_class:
                remove_cate_id.append(i["id"])
                
             
                
                
        return remove_cate_id
    
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
        print(image_id)
        path = img_meta['file_name']
        # print(os.path.join(self.root, path))
        image = Image.open(os.path.join(self.root, path)).convert('RGB')
        w, h = image.size


        anno = self.coco.loadAnns(self.coco.getAnnIds(image_id))
        anno = [obj for obj in anno if obj["iscrowd"] == 0]

        
        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)
        
        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)


        
        segmentations = [obj["segmentation"] for obj in anno]
        masks = self.convert_coco_poly_to_mask(segmentations, h, w)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep] 
        masks = masks[keep]
        
        keep2 = [idx for idx, cat in enumerate(classes) if cat not in self.remove_cate_id]
        
        boxes = boxes[keep2]
        classes = classes[keep2]
        masks = masks[keep2]

        classes = torch.tensor([self.cate2clsid[cat.item()] for cat in classes])
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["masks"] = masks

        image, target = self.transform(image, target)
        
        if self.train:
            return image, target

        else:
            raw_image = Image.open(os.path.join(self.root, path)).convert('RGB')
        
            return image, target, raw_image
        
      

    def __len__(self):
        return len(self.img_indices)



if __name__ == "__main__":
    s = CocoDataset(root="/home/gpu/Workspace/youmin/Teeth_Image_Segmentation/new_panorama_coco_dataset/images", 
                    json="/home/gpu/Workspace/youmin/Teeth_Image_Segmentation/new_panorama_coco_dataset/annotations/instances.json")

    i = 0
    print(s.__getitem__(i)[1])
    print()