import torch
import cv2
import os
import numpy as np
import argparse


import torch.nn.functional as F

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from loader.panorama_coco import CocoDataset
from models.maskrcnn import dental

from torchvision.ops.boxes import masks_to_boxes

import warnings
warnings.filterwarnings(action='ignore')

def show_anns(masks, scores, labels):

    ax = plt.gca()
    ax.set_autoscale_on(False)

    # print(masks)
    # exit()
    img = np.ones((masks[0].shape[0], masks[0].shape[1], 4))
    img[:,:,3] = 0

    
    # print(output)
    # print("aaa",len(output), len(anns))
    # exit()
    for idx, (mk, sc, lb) in enumerate(zip(masks, scores, labels)):

        class_name = lb
        pred_score = sc
        
        if pred_score < 0.4:
            continue
        
        # mk = mk.detach().cpu()
        th = 0.5
        mk[mk>=th] = 1
        mk[mk<th] = 0
        
        x1,y1,x2,y2 = masks_to_boxes(mk.unsqueeze(0))[0]
        
        x, y = int(x1), int(y1)
        w, h = int(x2-x1), int(y2- y1)
        
        m = mk.type(torch.bool)
        # m = mk.astype(np.bool_)
                
        rand_color = np.random.random(3)
        color_mask = np.concatenate([rand_color, [0.35]])
        
        img[m] = color_mask
        
        ax.add_patch(patches.Rectangle(xy=(x,y),width=w,height=h, color=rand_color, fill=False))
        
        # print_txt = "class : {}, score : {}%".format(class_name, int(pred_score*100))
        print_txt = "{}".format(class_name)
        ax.text(x, y, print_txt, backgroundcolor=rand_color)
        # break
        
    ax.imshow(img)


def main(args):

    GPU_NUM = args.gpu_num    
    args.device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    
    root = "/home/gpu/Workspace/youmin/Teeth_Image_Segmentation/panorama_dataset/Group1/images"
    json_path = "/home/gpu/Workspace/youmin/Teeth_Image_Segmentation/panorama_dataset/Group1/annotations/instances.json"
    
    dataset = CocoDataset(root=root, json=json_path, train=False)
    
    clsid2cate = dataset.clsid2cate
    
    total_cate = [None]
    total_cate += [i['name'] for i in dataset.coco.cats.values()]
    

    
    #random_idx = np.random.randint(dataset.__len__())
    # random_idx = 2111
    #print(random_idx)
    
    num_classes = len(dataset.class_cate)
    model = dental(num_classes=num_classes).to(args.device)
    
    
    ckp_path = "/home/gpu/Workspace/youmin/Teeth_Image_Segmentation/LBA/checkpoints/Thu_Dec__7_21-35-14_2023/epoch49.pth"
    state_dict = torch.load(ckp_path, map_location=args.device)
    model.load_state_dict(state_dict)
    
    #image, target, raw_image = dataset.__getitem__(random_idx)

    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    for idx in range(len(dataset)):
        image, target, raw_image = dataset.__getitem__(idx)

        with torch.no_grad():
            model.eval()

            image = image.to(args.device)
            image = image.unsqueeze(0)
            target = [target]
            target = [{k: v.to(args.device) for k, v in t.items()} for t in target]

            output = model(image)

            scores = output[0]["scores"].detach().cpu().numpy()
            labels = output[0]["labels"].detach().cpu().numpy()

            labels = [total_cate[clsid2cate[lb]] for lb in labels]

            masks = output[0]["masks"].squeeze(1).detach().cpu()

            raw_image = np.array(raw_image)
            h, w, _ = raw_image.shape

            masks = F.interpolate(masks.unsqueeze(0), (h, w))
            masks = masks.squeeze(0)

            plt.figure(figsize=(20, 20))
            plt.imshow(raw_image)
            show_anns(masks, scores, labels)

            plt.axis('off')
            save_path = os.path.join(save_dir, f"vis_pred_{idx}.jpg")
            plt.savefig(save_path)
            plt.close()


    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-num", type=int, default=0, help="gpu id number")
    
    parser.add_argument("--dataset-root", type=str, default="/mnt/d/Datasets", help="dataset name")
    parser.add_argument("--dataset-name", type=str, default="coco", help="dataset name")
    
    parser.add_argument("--save", action='store_true')
    parser.add_argument("--save-dir", type=str, default="checkpoints")
 
    args = parser.parse_args()

    args.save_dir = "vis_pred_1"
    main(args)