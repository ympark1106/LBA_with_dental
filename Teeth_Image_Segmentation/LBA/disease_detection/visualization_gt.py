import torch
import cv2
import os
import numpy as np
import argparse

import matplotlib.pyplot as plt
import matplotlib.patches as patches

# from torchvision.ops.boxes import masks_to_boxes

from loader.panorama_coco import CocoDataset
# from loader.panorama_coco_old import CocoDataset


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

    
def show_anns(masks, cate, labels):

    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((masks[0].shape[0], masks[0].shape[1], 4))
    img[:,:,3] = 0

    
    # print(output)
    # print("aaa",len(output), len(anns))
    # exit()

    # labels += [0] * (200 - len(labels))     
    

    for idx, lb in enumerate(labels):
    
        
        mk = masks[idx]

        # x1,y1,x2,y2 = masks_to_boxes(mk.unsqueeze(0))[0]
        # x, y = int(x1), int(y1)
        # w, h = int(x2-x1), int(y2- y1)
        
        mk = mk.type(torch.bool)
    
        rand_color = np.random.random(3)
        color_mask = np.concatenate([rand_color, [0.35]])

        img[mk] = color_mask

        
        # ax.add_patch(patches.Rectangle(xy=(x,y),width=w,height=h, color=rand_color, fill=False))
        
        # print_txt = "class : {}".format(class_name)
        # ax.text(x, y, print_txt, backgroundcolor=rand_color)
        
        
    ax.imshow(img)


def main(args):

    GPU_NUM = args.gpu_num    
    args.device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    
    # root = "/mnt/d/Datasets/panorama_dataset_old/images"
    # json_path = "/mnt/d/Datasets/panorama_dataset_old/annotations/instances.json"
    
    
    root = "/home/gpu/Workspace/youmin/Teeth_Image_Segmentation/panorama_dataset/Group1/images"
    json_path = "/home/gpu/Workspace/youmin/Teeth_Image_Segmentation/panorama_dataset/Group1/annotations/instances.json"
    
    
    dataset = CocoDataset(root=root, json=json_path, train=False)
    
    #random_idx = np.random.randint(dataset.__len__())

    # random_idx = 337
    # random_idx = 272
    random_idx = 21
    # random_idx = 351
    # random_idx = 316
    print(random_idx)


    labeled_pack = dataset.__getitem__(random_idx)

    with torch.no_grad():


        # image = np.array(labeled_image)
        # masks = np.array(labeled_pseudo)
        
    
        image, target, _ = labeled_pack

        masks = target["masks"]
        labels = target["labels"]
        cate = dataset.class_cate
        # exit()
        
        
        raw_image = np.array(image)
        raw_image = raw_image.transpose((1,2,0))
        # print(raw_image.shape)
        # exit()
        
        
        plt.figure(figsize=(20,20))
        plt.imshow(raw_image)
        show_anns(masks, cate, labels)
        plt.axis('off')
        # plt.savefig("vis.jpg")
        plt.savefig("vis_ori.jpg")

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-num", type=int, default=0, help="gpu id number")
    
    parser.add_argument("--dataset-root", type=str, default="/mnt/d/Datasets", help="dataset name")
    parser.add_argument("--dataset-name", type=str, default="coco", help="dataset name")
    
    parser.add_argument("--save", action='store_true')
    parser.add_argument("--save-dir", type=str, default="checkpoints")
 
    args = parser.parse_args()


    main(args)