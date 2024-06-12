import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, ColorJitter, RandomHorizontalFlip, RandomRotation
import loader_dino   # 수정
import model_dino_vit14b_linearprobe   # 수정
from torchmetrics.classification import MulticlassConfusionMatrix
# from utils import EarlyStopping  
from torch.utils.data import random_split
# from evaluation import evaluate_model
import matplotlib.pyplot as plt
from PIL import Image
import glob
import natsort

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

transform = Compose([
    Resize((224, 224)),  
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

parent_dir = '/home/gpu/Workspace/youmin/Learning-by-Asking/LBA/cropped_disease_images'
categories = ['cropped_K00_images', 'cropped_K01_images', 'cropped_K02_images', #'cropped_K03_images', 
'cropped_K04_images', 'cropped_K05_images', 
'cropped_K07_images', 'cropped_K08_images', 
'cropped_K09_images', 'cropped_normal_images'] # 9개의 카테고리


split_ratios = {'train': 0.7, 'val': 0.15, 'test': 0.15}

train_files, train_labels, val_files, val_labels, test_files, test_labels = loader_dino.split_data(parent_dir, categories, split_ratios)

train_dataset = loader_dino.TeethDataset(train_files, train_labels, transform, augment=True)
val_dataset = loader_dino.TeethDataset(val_files, val_labels, transform, augment=False)
test_dataset = loader_dino.TeethDataset(test_files, test_labels, transform, augment=False)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

file_label_dic = {}
file_path_dic = {}
category_count = {category: 0 for category in categories} 
categories_sorted = natsort.natsorted(categories, reverse=True)
for idx, category in enumerate(categories_sorted):
    cat_paths = natsort.natsorted(glob.glob(os.path.join(parent_dir, category, '*.png')))
    for file_path in cat_paths:
        file_name = os.path.basename(file_path)
        if file_name not in file_label_dic:
            file_label_dic[file_name] = -1  
        if file_label_dic[file_name] == -1:  
            file_label_dic[file_name] = idx
            category_count[category] += 1 

model = model_dino_vit14b_linearprobe.CustomDINOV2(num_classes=9).to(device)  

max_epoch = 200

class_counts = torch.zeros(len(categories), dtype=torch.int64)
for _, labels in train_loader:
    for label in labels:
        class_counts[label] += 1
# print(class_counts)

max_count = float(class_counts.max())
weights = max_count / class_counts
weights_tensor = weights.float().to(device)

criterion = nn.CrossEntropyLoss(weight=weights_tensor)
lr_decay = [int(0.5*max_epoch), int(0.75*max_epoch), int(0.9*max_epoch)]
# lr_decay = [int(0.375*max_epoch), int(0.5*max_epoch), int(0.75*max_epoch)]
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay = 1e-5)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, lr_decay)
# criterion = nn.CrossEntropyLoss()
# optimizer = Adam(model.classifier.parameters(), lr=0.001)
confmat = MulticlassConfusionMatrix(num_classes=9) # 수정 필요

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
            preds = outputs.argmax(dim=1)


            running_loss += loss.item() * images.size(0)
            corrects += (preds == labels).sum().item()
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

model_save_directory = '/home/gpu/Workspace/youmin/Learning-by-Asking/LBA/disease_multiclassclassification/checkpoints/disease_center/9_saved_dino_linearprobe_weight_0514' # 'saved_dinovit14b
if not os.path.exists(model_save_directory):
    os.makedirs(model_save_directory)

for epoch in range(max_epoch):
    model.train()
    running_loss = 0.0
    corrects = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        preds = outputs.argmax(dim=1)
        loss = criterion(outputs, labels)
        loss.backward()
        # print(loss)
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        corrects += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = float(corrects) / total

    val_loss, val_acc = evaluate_model(model, val_loader, criterion, device, confmat)
  
    val_confmat = MulticlassConfusionMatrix(num_classes=9)
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

    scheduler.step()

    # confmat.update(preds.int().clone().detach().cpu(), labels.int().to(device='cpu'))
    print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")
    print(scheduler.get_last_lr())

# s = confmat.compute()
# print(s)


final_model_path = os.path.join(model_save_directory, 'final_model.pth')
torch.save(model.state_dict(), final_model_path)
print(f"Model saved to {final_model_path}")

test_confmat = MulticlassConfusionMatrix(num_classes=9)
test_loss, test_acc = evaluate_model(model, test_loader, criterion, device, test_confmat)
print("Test Confusion Matrix:")
print_confusion_matrix(test_confmat)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

