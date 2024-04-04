import torch
import cv2
import os
import numpy as np
import argparse
import json

import matplotlib.pyplot as plt
import matplotlib.patches as patches

# from torchvision.ops.boxes import masks_to_boxes

from loader.panorama_coco import CocoDataset
# from loader.panorama_coco_old import CocoDataset


from torchvision.utils import _log_api_usage_once
def masks_to_boxes(masks: torch.Tensor) -> torch.Tensor:

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
    
    for idx, lb in enumerate(labels):
        mk = masks[idx]
        mk = mk.type(torch.bool)
    
        rand_color = np.random.random(3)
        color_mask = np.concatenate([rand_color, [0.35]])

        img[mk] = color_mask

        cate_id = labels[idx].item()
        class_name = cate[cate_id]
        
        x1, y1, x2, y2 = masks_to_boxes(mk.unsqueeze(0))[0]
        x, y = int(x1), int(y1)
        w, h = int(x2 - x1), int(y2 - y1)

        ax.add_patch(patches.Rectangle(xy=(x, y), width=w, height=h, color=rand_color, fill=False))
        print_txt = "{}".format(class_name)
        ax.text(x, y, print_txt, backgroundcolor=rand_color)
        
    ax.imshow(img)


def main(args):
    GPU_NUM = args.gpu_num    
    args.device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    
    root = "/home/gpu/Workspace/youmin/Teeth_Image_Segmentation/panorama_dataset/Group1/images"
    json_path = "/home/gpu/Workspace/youmin/Teeth_Image_Segmentation/panorama_dataset/Group1/annotations/instances.json"
    
    dataset = CocoDataset(root=root, json=json_path, train=False)

    # Load the JSON file
    with open(args.json_path, 'r') as f:
        annotations = json.load(f)

    # Specify the desired image_id
    desired_image_id = 299875888

    # Check if the desired image_id exists in the dataset
    image_ids = [img_info['id'] for img_info in annotations['images']]
    if desired_image_id not in image_ids:
        print(f"Image with image_id {desired_image_id} not found in the dataset.")
        return

    # Find the index of the desired image_id
    random_idx = image_ids.index(desired_image_id)

    labeled_pack = dataset.__getitem__(random_idx)

    with torch.no_grad():
        image, target, _ = labeled_pack
        masks = target["masks"]
        labels = target["labels"]
        cate = dataset.class_cate

        raw_image = np.array(image)
        raw_image = raw_image.transpose((1, 2, 0))

        plt.figure(figsize=(20, 20))
        plt.imshow(raw_image)
        show_anns(masks, cate, labels)

        plt.axis('off')
        plt.savefig("vis_desired_image.jpg")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-num", type=int, default=0, help="gpu id number")
    parser.add_argument("--json-path", type=str, default="/home/gpu/Workspace/youmin/Teeth_Image_Segmentation/panorama_dataset/Group1/annotations/instances.json", help="path to the annotations JSON file")

    args = parser.parse_args()

    main(args)


