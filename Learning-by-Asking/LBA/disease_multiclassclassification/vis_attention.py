import os
import sys
import argparse
import cv2
import random
import colorsys
import requests
from io import BytesIO

import skimage.io
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as pth_transforms
import numpy as np
from PIL import Image
# from ensemble.model_dino_vit14b_rein_ensemble import CustomDINOV2
from dinov2.dinov2.models.vision_transformer import vit_small, vit_base, vit_large


if __name__ == '__main__':
    image_size = (952, 952)
    output_dir = '.'
    patch_size = 16

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = vit_base(
            patch_size=16,
            img_size=526,
            init_values=1.0,
            #ffn_layer="mlp",
            block_chunks=0
    )

    # model = CustomDINOV2(num_classes=9, depth=12, embed_dim=768, patch_size=16)

    model_save_path = '/home/gpu/Workspace/youmin/Learning-by-Asking/LBA/disease_multiclassclassification/checkpoints/disease_center/9_saved_dinovit14b_rein_ensemble_0514/epoch_36_valloss_1.6036205207224654_valacc_0.7647058823529411.pth'

    model.load_state_dict(torch.load(model_save_path)['model1_state_dict'])
    
    # model.load_state_dict(torch.load('dinov2_vitb14_pretrain.pth'))
    for p in model.parameters():
        p.requires_grad = False
    model.to(device)
    model.eval()

    img = Image.open('/home/gpu/Workspace/youmin/Learning-by-Asking/LBA/cropped_disease_images/cropped_K00_images/60_1471.0_613.0.png')
    img = img.convert('RGB')
    transform = pth_transforms.Compose([
        pth_transforms.Resize(image_size),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    img = transform(img)
    print(img.shape)

    # make the image divisible by the patch size
    w, h = img.shape[1] - img.shape[1] % patch_size, img.shape[2] - img.shape[2] % patch_size
    img = img[:, :w, :h].unsqueeze(0)

    w_featmap = img.shape[-2] // patch_size
    h_featmap = img.shape[-1] // patch_size

    print(img.shape)

    attentions = model.get_last_selfattention(img.to(device))

    nh = attentions.shape[1] # number of head

    # we keep only the output patch attention
    # for every patch
    attentions = attentions[0, :, 0, 1:].reshape(nh, -1)
    # weird: one pixel gets high attention over all heads?
    print(torch.max(attentions, dim=1)) 
    attentions[:, 283] = 0 

    attentions = attentions.reshape(nh, w_featmap, h_featmap)
    attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0].cpu().numpy()

    # save attentions heatmaps
    os.makedirs(output_dir, exist_ok=True)

    for j in range(nh):
        fname = os.path.join(output_dir, "attn-head" + str(j) + ".png")
        plt.imsave(fname=fname, arr=attentions[j], format='png')
        print(f"{fname} saved.")
