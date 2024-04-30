import torch
import cv2
import os
import numpy as np
import argparse

import matplotlib.pyplot as plt
import matplotlib.patches as patches

# from torchvision.ops.boxes import masks_to_boxes

# from loader.panorama_coco import CocoDataset
from loader import panorama_loader_t
from loader import panorama_coco
from torchvision.utils import _log_api_usage_once

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch




def show_anns(bboxes, cate, labels):
    ax = plt.gca()
    ax.set_autoscale_on(False)

    # img = np.ones((800, 800, 4))  
    # img[:, :, 3] = 0  
    
    for idx, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1

        rand_color = np.random.random(3)  
        ax.add_patch(patches.Rectangle((x1, y1), w, h, linewidth=2, edgecolor=rand_color, facecolor='none')) 
        
        cate_id = labels[idx].item()  
        class_name = cate[cate_id]  
        
        print_txt = f"{class_name}"  
        ax.text(x1, y1, print_txt, backgroundcolor=rand_color, fontsize=12)   


def main(args):
    GPU_NUM = args.gpu_num    
    args.device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    
    root_path = "/home/gpu/Workspace/youmin/Teeth_Image_Segmentation/new_panorama_coco_dataset/images"
    json_path = "/home/gpu/Workspace/youmin/Teeth_Image_Segmentation/new_panorama_coco_dataset/annotations/instances.json"
    
    dataset = panorama_loader_t.CocoDataset(root=root_path, json=json_path)

    if args.image_id is not None:
        labeled_pack = dataset.get_item_by_id(args.image_id)
        print(f"Image ID: {args.image_id}")
    else:
        random_idx = np.random.randint(dataset.__len__())
        labeled_pack = dataset[random_idx]
        print(f"Random index: {random_idx}")
        
    for img_id in dataset.img_indices:  

        labeled_pack = dataset.get_item_by_id(img_id)

        with torch.no_grad():
            image, target = labeled_pack  
            bboxes = target["bboxes"]  
            labels = target["labels"]
            cate = dataset.class_cate

            raw_image = np.array(image)#.transpose((1, 2, 0)) 

            plt.figure(figsize=(20, 20))
            plt.imshow(raw_image)
            show_anns(bboxes, cate, labels)

            plt.axis('off')
            plt.title(f"Image ID: {img_id}")  
            if args.save:
                save_path = os.path.join(args.save_dir, f"vis_{img_id}.jpg")  
                plt.savefig(save_path)
                plt.close()  
                print(f"Image ID: {img_id} processed and saved to {save_path}")
            else:
                plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-num", type=int, default=0, help="gpu id number")
    
    parser.add_argument("--dataset-root", type=str, default="/mnt/d/Datasets", help="dataset name")
    parser.add_argument("--dataset-name", type=str, default="coco", help="dataset name")
    parser.add_argument("--image-id", type=int, help="Image ID to visualize")
    
    parser.add_argument("--save", action='store_true')
    parser.add_argument("--save-dir", type=str, default="checkpoints")
 
    args = parser.parse_args()

    main(args)