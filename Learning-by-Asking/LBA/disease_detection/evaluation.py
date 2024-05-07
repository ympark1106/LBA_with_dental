import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torchmetrics.detection import MeanAveragePrecision

from loader.panorama_coco import CocoDataset
from models.maskrcnn import dental

import warnings
warnings.filterwarnings(action='ignore')


def main(args):

    GPU_NUM = args.gpu_num    
    args.device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    
    root = "/home/gpu/Workspace/youmin/Learning-by-Asking/new_panorama_coco_dataset/images" 
    json_path = "/home/gpu/Workspace/youmin/Learning-by-Asking/new_panorama_coco_dataset/annotations/instances.json"
    dataset = CocoDataset(root=root, json=json_path, train=False)

    
    num_classes = len(dataset.class_cate)
    model = dental(num_classes=num_classes).to(args.device)
    
    ckp_path = "checkpoints/Fri_Nov_24_11-35-03_2023/epoch20.pth"
    state_dict = torch.load(ckp_path, map_location=args.device)
    model.load_state_dict(state_dict)
    
    # sam_model = sam_load(args.device)
    
    metric = MeanAveragePrecision(iou_type="segm", class_metrics=True)
    
    with torch.no_grad():
        model.eval()
        
        for i in tqdm(range(dataset.__len__())):
            
            image, target, raw_image = dataset.__getitem__(i)
            
            image = image.to(args.device)
            image = image.unsqueeze(0)
            target = [target]
            target = [{k: v.to(args.device) for k, v in t.items()} for t in target]

            output = model(image)
        
 
            scores = output[0]["scores"].detach()
            labels = output[0]["labels"].detach()
            masks = output[0]["masks"].detach().squeeze(1)
            
            # keep_idx = [idx for idx, sc in enumerate(scores) if sc >= 0.5]
            
            # if len(keep_idx) == 0:
            #     continue
            
            # scores  = scores[keep_idx]
            # masks  = masks[keep_idx]
            # labels = labels[keep_idx]
            
            # print(image.shape)
            # exit()
            # h, w, _  = image.shape
            
            
            
            th = 0.5
            masks[masks>=th] = 1
            masks[masks<th] = 0
            
            preds = [dict(masks=masks.type(torch.bool), scores=scores, labels=labels)]
    
            metric.update(preds, target)
            
        from pprint import pprint
        pprint(metric.compute())

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-num", type=int, default=0, help="gpu id number")
    
    parser.add_argument("--dataset-root", type=str, default="/mnt/d/Datasets", help="dataset name")
    parser.add_argument("--dataset-name", type=str, default="coco", help="dataset name")
    
    parser.add_argument("--save", action='store_true')
    parser.add_argument("--save-dir", type=str, default="checkpoints")
 
    args = parser.parse_args()


    main(args)