from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50
import torchvision
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os

path = "/home/gpu/Workspace/youmin/Teeth_Image_Segmentation/LBA/cropped_K01_images/209_31_decay_teeth_653.0_747.0.png"

rgb_img = Image.open(path).convert('RGB')
rgb_img = np.array(rgb_img)
rgb_img = (rgb_img - np.min(rgb_img)) / (np.max(rgb_img) - np.min(rgb_img))


model = resnet50()
target_layers = [model.layer4[-1]]
input_tensor = torchvision.transforms.functional.to_tensor(rgb_img).unsqueeze(0).float()


cam = GradCAM(model=model, target_layers=target_layers)

targets = [ClassifierOutputTarget(281)]

grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
grayscale_cam = grayscale_cam[0, :]
visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

model_outputs = cam.outputs

save_path = "/home/gpu/Workspace/youmin/Teeth_Image_Segmentation/LBA/disease_detection/grad_cam"
base_name = os.path.basename(path)  
save_filename = f"cam_{base_name}"  
full_path = os.path.join(save_path, save_filename)

plt.figure(figsize=(10, 10))
plt.imshow(visualization)
plt.axis('off') 
plt.savefig(full_path, bbox_inches='tight', pad_inches=0)
plt.close()  
