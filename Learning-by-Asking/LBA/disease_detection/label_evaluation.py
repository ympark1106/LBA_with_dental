import torch
from torch.utils.data import DataLoader
from Teeth_Image_Segmentation.LBA.disease_detection.loader1 import TeethDataset  
from model import ResNet50  
from torchvision import transforms
from sklearn.metrics import classification_report

# disease_images_dir = '/home/gpu/Workspace/youmin/Teeth_Image_Segmentation/LBA/cropped_K02_images'
# normal_images_dir = '/home/gpu/Workspace/youmin/Teeth_Image_Segmentation/LBA/not_K02_images'
disease_images_dir = '/home/gpu/Workspace/youmin/Teeth_Image_Segmentation/LBA/cropped_K01_images'
normal_images_dir = '/home/gpu/Workspace/youmin/Teeth_Image_Segmentation/LBA/not_K01_images'

def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    print(classification_report(all_labels, all_preds, target_names=['K01', 'Other']))

transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
dataset = TeethDataset(disease_images_dir, normal_images_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet50().to(device)

# model.load_state_dict(torch.load('path/to/model.pth'))

evaluate_model(model, dataloader, device)
