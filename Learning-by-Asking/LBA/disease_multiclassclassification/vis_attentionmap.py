import pandas as pd
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as pth_transforms
import numpy as np
from PIL import Image
from ensemble.model_dino_vit14b_rein_ensemble import CustomDINOV2

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

patch_size = 16

model1 = CustomDINOV2(num_classes=9, depth=12, embed_dim=768, patch_size=16).to(device)
model2 = CustomDINOV2(num_classes=9, depth=12, embed_dim=768, patch_size=16).to(device)

model_save_path = '/home/gpu/Workspace/youmin/Learning-by-Asking/LBA/disease_multiclassclassification/checkpoints/disease_center/9_saved_dinovit14b_rein_ensemble_0514/epoch_36_valloss_1.6036205207224654_valacc_0.7647058823529411.pth'

checkpoint = torch.load(model_save_path)
model1.load_state_dict(checkpoint['model1_state_dict']) 
model2.load_state_dict(checkpoint['model2_state_dict'])

model1.eval()
model2.eval()


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image

def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out


image_path = '/home/gpu/Workspace/youmin/Learning-by-Asking/LBA/cropped_disease_images/cropped_K00_images/60_1471.0_613.0.png'
image_file = image_path
image_files = [image_path]
image = load_image(image_file)

transform = pth_transforms.Compose([
    pth_transforms.ToTensor(),
    pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
img = transform(image)
print(img.shape)

w, h = img.shape[1] - img.shape[1] % patch_size, img.shape[2] - img.shape[2] % patch_size
img = img[:, :w, :h].unsqueeze(0)

w_featmap = img.shape[-2] // patch_size
h_featmap = img.shape[-1] // patch_size


attentions = model1.get_last_selfattention(img) 
print(attentions.shape)

attentions = model1.get_last_selfattention(img) 
print(img.shape)
print(attentions.shape)
plt.imshow(attentions[0][0].reshape(901,901))
plt.axis("off")
plt.show()

nh = attentions.shape[1] 

attentions = attentions[0, :, 0, 1:].reshape(nh, -1)
print(attentions.shape)

val, idx = torch.sort(attentions)
val /= torch.sum(val, dim=1, keepdim=True)
cumval = torch.cumsum(val, dim=1)

threshold = 0.6 
th_attn = cumval > (1 - threshold)
idx2 = torch.argsort(idx)
for head in range(nh):
    th_attn[head] = th_attn[head][idx2[head]]
    
th_attn = th_attn.reshape(nh, w_featmap//2, h_featmap//2).float()


th_attn = nn.functional.interpolate(th_attn.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0].cpu().numpy()

attentions = attentions.reshape(nh, w_featmap//2, h_featmap//2)
attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0].cpu().numpy()
attentions_mean = np.mean(attentions, axis=0)

print(attentions.shape)