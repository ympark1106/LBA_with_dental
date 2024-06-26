import torch
import cv2
import os
import numpy as np
import argparse

import matplotlib.pyplot as plt
import matplotlib.patches as patches

# from loader.panorama_loader_only_d import CocoDataset
from loader.panorama_loader import CocoDataset
# from loader.panorama_loader_only_d import CocoDataset



from torchvision.utils import _log_api_usage_once
def masks_to_boxes(masks: torch.Tensor) -> torch.Tensor:
    """
    Compute the bounding boxes around the provided masks.

    Returns a [N, 4] tensor containing bounding boxes. The boxes are in ``(x1, y1, x2, y2)`` format with
    ``0 <= x1 < x2`` and ``0 <= y1 < y2``.

    Args:
        masks (Tensor[N, H, W]): masks to transform where N is the number of masks
            and (H, W) are the spatial dimensions.

    Returns:
        Tensor[N, 4]: bounding boxes
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(masks_to_boxes)
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device, dtype=torch.float)

    n = masks.shape[0]

    bounding_boxes = torch.zeros((n, 4), device=masks.device, dtype=torch.float)
    
    for index, mask in enumerate(masks):
        y, x = torch.where(mask != 0)
        
        bounding_boxes[index, 0] = torch.min(x)
        bounding_boxes[index, 1] = torch.min(y)
        bounding_boxes[index, 2] = torch.max(x)
        bounding_boxes[index, 3] = torch.max(y)

    return bounding_boxes

def show_anns(masks, labels, category, bboxes):

    ax = plt.gca()
    ax.set_autoscale_on(False)


    img = np.ones((masks[0].shape[0], masks[0].shape[1], 4))
    img[:,:,3] = 0

    
    # print(output)
    # print("aaa",len(output), len(anns))
    # exit()

    # labels += [0] * (200 - len(labels))     
    
    print("inst num : " + str(len(labels)))
    

    for idx, lb in enumerate(labels):
    
        # if lb < 34:
        #     continue
        mk = masks[idx]

        # if lb == 33:
        #     continue
        # x1,y1,x2,y2 = masks_to_boxes(mk.unsqueeze(0))[0]
        
        x1,y1,x2,y2 = bboxes[idx]
        # print(x1,y1,x2,y2, category[lb])
        # print(torch.sum(torch.abs(bboxes[idx]-bboxes[6])), category[lb])
        x, y = int(x1), int(y1)
        w, h = int(x2-x1), int(y2- y1)
        
        mk = mk.type(torch.bool)
    
        rand_color = np.random.random(3)
        color_mask = np.concatenate([rand_color, [0.35]])

        img[mk] = color_mask
        
        ax.add_patch(patches.Rectangle(xy=(x,y),width=w,height=h, color=rand_color, fill=False))
        
        # print_txt = "class : {}".format(category[lb+33])
        print_txt = "class : {}".format(category[lb])
        ax.text(x, y, print_txt, backgroundcolor=rand_color)
        
    ax.imshow(img)


def main(args):

    GPU_NUM = args.gpu_num    
    args.device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    

    
    root = "panorama_coco_2014/coco"
    json_path = "panorama_coco_2014/coco/annotations/instances_with_tagv2.json"
    
    
    dataset = CocoDataset(root=root, json=json_path, train=False)
    
    category = dataset.class_cate

    random_idx = np.random.randint(dataset.__len__())
    # random_idx = 5362
    print(random_idx)
    
   
    labeled_pack = dataset.__getitem__(random_idx)

    with torch.no_grad():


        # image = np.array(labeled_image)
        # masks = np.array(labeled_pseudo)
        
    
        image, target, raw_image = labeled_pack


        masks = target["masks"]
        labels = target["labels"]
        
        bboxes = target["boxes"]
        # exit()
        
        
        raw_image = np.array(raw_image)
        
        # raw_image = raw_image.transpose((1,2,0))
        # print(raw_image.shape)
        # exit()
        
        plt.rcParams["font.family"] = 'NanumGothicCoding'
        plt.figure(figsize=(20,20))
        plt.imshow(raw_image)
        show_anns(masks, labels, category, bboxes)
        plt.axis('off')
        # plt.savefig("vis.jpg")
        plt.savefig("vis_ori.jpg", bbox_inches='tight')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-num", type=int, default=0, help="gpu id number")
    
    parser.add_argument("--dataset-root", type=str, default="/mnt/d/Datasets", help="dataset name")
    parser.add_argument("--dataset-name", type=str, default="coco", help="dataset name")
    
    parser.add_argument("--save", action='store_true')
    parser.add_argument("--save-dir", type=str, default="checkpoints")
 
    args = parser.parse_args()


    main(args)