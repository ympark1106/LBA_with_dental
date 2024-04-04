import torch
from torch.utils.data import DataLoader
from loader import TeethDataset  
from model import ResNet50  
from torchvision import transforms
from sklearn.metrics import classification_report

K01_images_dir = '/home/gpu/Workspace/youmin/Teeth_Image_Segmentation/LBA/cropped_K01_images' # 34 cropped images
K02_images_dir = '/home/gpu/Workspace/youmin/Teeth_Image_Segmentation/LBA/cropped_K02_images' # 35 cropped images
other_images_dir = '/home/gpu/Workspace/youmin/Teeth_Image_Segmentation/LBA/not_K01_K02_images' # 34, 35 제외한 치아 cropped images

def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.sigmoid(outputs) > 0.5
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    print(classification_report(all_labels, all_preds, target_names=['K01', 'K02', 'Other']))

transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
dataset = TeethDataset(K01_images_dir, K02_images_dir,other_images_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet50().to(device)

# model.load_state_dict(torch.load('path/to/model.pth'))

evaluate_model(model, dataloader, device)
