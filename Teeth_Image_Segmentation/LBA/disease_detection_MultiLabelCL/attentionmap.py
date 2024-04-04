import torch
import os
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from model_dino_vit14b import CustomDINOV2
from torchvision.models import vit_b_16, ViT_B_16_Weights




def preprocess_image(image_path):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path)
    image_tensor = preprocess(image).unsqueeze(0) 
    return image_tensor

def visualize_and_save_attention_maps(model, image_path, save_dir):
    image_tensor = preprocess_image(image_path)
    attentions = model.get_intermediate_layers(image_tensor, n=1)[0]
    attentions = attentions[0]  # 첫번째 레이어의 attention을 가져옵니다.
    nh = attentions.shape[1]  # attention heads의 수

    for i in range(nh):
        class_dir = os.path.join(save_dir, f'class_{i+1}')
        os.makedirs(class_dir, exist_ok=True)
        
        attention = attentions[0, i].mean(0).detach().numpy()
        
        plt.imshow(attention, cmap='viridis')
        plt.axis('off')
        plt.savefig(os.path.join(class_dir, f'{os.path.basename(image_path).split(".")[0]}_attention_map.png'))
        plt.close()

def main(model_path, image_path, save_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CustomDINOV2(num_classes=5).to(device)  
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    visualize_and_save_attention_maps(model, image_path, save_dir)

image_path = '/home/gpu/Workspace/youmin/Teeth_Image_Segmentation/LBA/cropped_images/duplicated_margin150/cropped_K09_images/237_31_506.0_552.0.png'
save_dir = '/home/gpu/Workspace/youmin/Teeth_Image_Segmentation/LBA/disease_detection_MultiLabelCL/attentionmap'
model_path = '/home/gpu/Workspace/youmin/Teeth_Image_Segmentation/LBA/disease_detection_MultiLabelCL/saved_dinovit14b_margin150_5class_0401/model_epoch_15_valloss_0.5420217642128297_valacc_0.5530085959885387.pth'
main(model_path, image_path, save_dir)
