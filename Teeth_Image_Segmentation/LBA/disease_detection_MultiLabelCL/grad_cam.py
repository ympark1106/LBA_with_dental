from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
# from torchvision.models import resnet50
from model import resnet50
import torchvision
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os

path = "/home/gpu/Workspace/youmin/Teeth_Image_Segmentation/LBA/cropped_images/cropped_K01_images/4566_23_1919.0_644.0.png"

rgb_img = Image.open(path).convert('RGB')
rgb_img = np.array(rgb_img)
rgb_img = (rgb_img - np.min(rgb_img)) / (np.max(rgb_img) - np.min(rgb_img))


model = resnet50()
target_layers = [model.layer4[-1]]
input_tensor = torchvision.transforms.functional.to_tensor(rgb_img).unsqueeze(0).float()

cam = GradCAM(model=model, target_layers=target_layers)

targets = [ClassifierOutputTarget(2)]

# grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
# grayscale_cam = grayscale_cam[0, :]
# visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

for class_idx in range(9):  # 9개 클래스에 대해 반복
    targets = [ClassifierOutputTarget(class_idx)]  # 각 클래스 인덱스에 맞게 Target 설정
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

model_outputs = cam.outputs

save_path = "/home/gpu/Workspace/youmin/Teeth_Image_Segmentation/LBA/disease_detection_MultiLabelCL/grad_cam"
base_name = os.path.basename(path)  
# save_filename = f"cam_{base_name}"  
save_filename = f"Class {class_idx} {base_name}"  
full_path = os.path.join(save_path, save_filename)

# plt.figure(figsize=(10, 10))
# plt.imshow(visualization)
# plt.axis('off') 
# plt.savefig(full_path, bbox_inches='tight', pad_inches=0)
# plt.close()  

plt.figure(figsize=(10, 10))
plt.imshow(visualization)
plt.axis('off')
# plt.title(f"Class {class_idx} {base_name}")
plt.savefig(full_path, bbox_inches='tight', pad_inches=0)
plt.close()