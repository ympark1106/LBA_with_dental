from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from model import ResNet50  
from model_dino_vit14b import CustomDINOV2
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch
import os

path = "/home/gpu/Workspace/youmin/Teeth_Image_Segmentation/LBA/cropped_images/duplicated_margin150/cropped_K09_images/237_31_506.0_552.0.png"

rgb_img = Image.open(path).convert('RGB')
rgb_img = np.array(rgb_img) / 255.0  # 이미지를 [0, 1] 범위로 정규화

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CustomDINOV2(num_classes=5).to(device)  

model_save_path = '/home/gpu/Workspace/youmin/Teeth_Image_Segmentation/LBA/disease_detection_MultiLabelCL/saved_dinovit14b_margin150_5class_0401/model_epoch_15_valloss_0.5420217642128297_valacc_0.5530085959885387.pth'  
checkpoint = torch.load(model_save_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

target_layers = [model.layers[-1].blocks[-1].norm1]  

input_tensor = TF.to_tensor(rgb_img).unsqueeze(0).to(device).float()

cam = GradCAM(model=model, target_layers=target_layers)

save_path = "/home/gpu/Workspace/youmin/Teeth_Image_Segmentation/LBA/disease_detection_MultiLabelCL/grad_cam"
if not os.path.exists(save_path):
    os.makedirs(save_path)

base_name = os.path.basename(path)

for class_idx in range(5):  # 9개 클래스에 대해 반복
    targets = [ClassifierOutputTarget(class_idx)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    save_filename = f"Class_{class_idx}_{base_name}"
    full_path = os.path.join(save_path, save_filename)

    plt.figure(figsize=(10, 10))
    plt.imshow(visualization)
    plt.axis('off')
    plt.savefig(full_path, bbox_inches='tight', pad_inches=0)
    plt.close()
