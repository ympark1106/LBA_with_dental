import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, ColorJitter, RandomHorizontalFlip, RandomRotation
from loader import loader_dino   # 수정
from LBA.disease_multilabelclassification.model import model_dino_vit14b_rein   # 수정
from torchmetrics.classification import MultilabelConfusionMatrix
# from utils import EarlyStopping  
from torch.utils.data import random_split
# from evaluation import evaluate_model
import matplotlib.pyplot as plt
from PIL import Image

device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")

transform = Compose([
    Resize((224, 224)),  
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


parent_dir = '/home/gpu/Workspace/youmin/Learning-by-Asking/LBA/cropped_images/margin150'
categories = ['cropped_K00_images', 'cropped_K01_images', 'cropped_K02_images', 'cropped_K03_images', 'cropped_K04_images', 
                    'cropped_K05_images', 'cropped_K07_images', 'cropped_K08_images', 'cropped_K09_images'] # 9개의 카테고리
# categories = ['cropped_K01_images', 'cropped_K02_images', 
#                     'cropped_K05_images', 'cropped_K08_images', 'cropped_K09_images'] # 5개의 카테고리

split_ratios = {'train': 0.7, 'val': 0.15, 'test': 0.15}

train_files, train_labels, val_files, val_labels, test_files, test_labels = loader_dino.split_data(parent_dir, categories, split_ratios)

train_dataset = loader_dino.TeethDataset(train_files, train_labels, transform, augment=True)
val_dataset = loader_dino.TeethDataset(val_files, val_labels, transform, augment=False)
test_dataset = loader_dino.TeethDataset(test_files, test_labels, transform, augment=False)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model = model_dino_vit14b_rein.CustomDINOV2(num_classes=9, depth=12, embed_dim=384, patch_size=16).to(device)  


criterion = nn.BCEWithLogitsLoss()
optimizer = Adam(model.parameters(), lr=0.0001)
# optimizer = Adam(model.parameters(), lr=0.000005)
confmat = MultilabelConfusionMatrix(num_labels=9) # 수정 필요

def evaluate_model(model, dataloader, criterion, device, confmat=None):
    model.eval()
    running_loss = 0.0
    corrects = 0
    total = 0

    if confmat is not None:
        confmat.reset()

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            preds = torch.sigmoid(outputs) > 0.5

            running_loss += loss.item() * images.size(0)
            corrects += ((preds == labels).float().mean(dim=1) == 1).sum().item()
            total += labels.size(0)

            if confmat is not None:
                confmat.update(preds.int().clone().detach().cpu(), labels.int().to(device='cpu'))

    final_loss = running_loss / total
    final_acc = float(corrects) / total
    return final_loss, final_acc

def print_confusion_matrix(confmat):
    cm = confmat.compute()
    print(cm)
    confmat.reset() 

model_save_directory = '/home/gpu/Workspace/youmin/Learning-by-Asking/LBA/disease_multilabelclassification/checkpoints/saved_dinovit14b_0.00005_rein_0417_1' # 'saved_dinovit14b
if not os.path.exists(model_save_directory):
    os.makedirs(model_save_directory)

for epoch in range(300):
    model.train()
    running_loss = 0.0
    corrects = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        labels = labels.to(dtype=torch.float)

        optimizer.zero_grad()

        outputs = model(images)
        preds = torch.sigmoid(outputs) > 0.5
        loss = criterion(outputs, labels)
        loss.backward()
        # print(loss)
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        corrects += ((preds == labels).float().mean(dim=1) == 1).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = float(corrects) / total

    val_loss, val_acc = evaluate_model(model, val_loader, criterion, device, confmat)

  
    val_confmat = MultilabelConfusionMatrix(num_labels=9)
    val_loss, val_acc = evaluate_model(model, val_loader, criterion, device, val_confmat)
    print("Validation Confusion Matrix:")
    print_confusion_matrix(val_confmat)

    model_save_name = f'model_epoch_{epoch+1}_valloss_{val_loss}_valacc_{val_acc}.pth'
    model_save_path = os.path.join(model_save_directory, model_save_name)
    torch.save({
        'model_state_dict': model.state_dict(),
        'epoch_loss': epoch_loss,
        'epoch_acc': epoch_acc,
        'val_loss': val_loss,
        'val_acc': val_acc
    }, model_save_path)

    # confmat.update(preds.int().clone().detach().cpu(), labels.int().to(device='cpu'))
    print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")

# s = confmat.compute()
# print(s)


final_model_path = os.path.join(model_save_directory, 'final_model.pth')
torch.save(model.state_dict(), final_model_path)
print(f"Model saved to {final_model_path}")

test_confmat = MultilabelConfusionMatrix(num_labels=9)
test_loss, test_acc = evaluate_model(model, test_loader, criterion, device, test_confmat)
print("Test Confusion Matrix:")
print_confusion_matrix(test_confmat)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

