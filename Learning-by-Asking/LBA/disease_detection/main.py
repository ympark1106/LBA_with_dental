import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from Teeth_Image_Segmentation.LBA.disease_detection.loader1 import TeethDataset
from Teeth_Image_Segmentation.LBA.disease_detection.loader1 import transforms 
from model import ResNet50 
# from utils import EarlyStopping  
from torch.utils.data import random_split
# from evaluation import evaluate_model

from grad_cam import GradCAM
import matplotlib.pyplot as plt
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

disease_images_dir = '/home/gpu/Workspace/youmin/Teeth_Image_Segmentation/LBA/cropped_K01_images'
normal_images_dir = '/home/gpu/Workspace/youmin/Teeth_Image_Segmentation/LBA/not_K01_images'

dataset = TeethDataset(disease_images_dir, normal_images_dir, transform=transform)

train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model = ResNet50().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    corrects = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            running_loss += loss.item() * images.size(0)
            corrects += torch.sum(preds == labels.data)
            total += labels.size(0)

    final_loss = running_loss / total
    final_acc = corrects.double() / total
    return final_loss, final_acc

model_save_directory = '/home/gpu/Workspace/youmin/Teeth_Image_Segmentation/LBA/disease_detection/K01_saved_models'
if not os.path.exists(model_save_directory):
    os.makedirs(model_save_directory)

for epoch in range(30):
    model.train()
    running_loss = 0.0
    corrects = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        corrects += torch.sum(preds == labels.data)
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = corrects.double() / total

    val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)

    model_save_name = f'model_epoch_{epoch+1}.pth'
    model_save_path = os.path.join(model_save_directory, model_save_name)
    torch.save({
        'model_state_dict': model.state_dict(),
        'epoch_loss': epoch_loss,
        'epoch_acc': epoch_acc,
        'val_loss': val_loss,
        'val_acc': val_acc
    }, model_save_path)

    print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")

final_model_path = os.path.join(model_save_directory, 'final_model.pth')
torch.save(model.state_dict(), final_model_path)
print(f"Model saved to {final_model_path}")

test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")






