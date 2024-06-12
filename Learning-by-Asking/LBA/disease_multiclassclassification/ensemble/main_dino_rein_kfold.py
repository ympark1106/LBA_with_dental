import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torchmetrics.classification import MulticlassConfusionMatrix
import loader_dino_cv
import model_dino_vit14b_rein
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

transform = Compose([
    Resize((224, 224)),  
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

parent_dir = '/home/gpu/Workspace/youmin/Learning-by-Asking/LBA/cropped_disease_images'
categories = ['cropped_K00_images', 'cropped_K01_images', 'cropped_K02_images', #'cropped_K03_images', 
'cropped_K04_images', 'cropped_K05_images', 
'cropped_K07_images', 'cropped_K08_images', 
'cropped_K09_images', 'cropped_normal_images']
split_ratios = {'train': 0.7, 'val': 0.15, 'test': 0.15}

train_files, train_labels, val_files, val_labels, test_files, test_labels = loader_dino_cv.split_data(parent_dir, categories, split_ratios)

all_train_files = np.array(train_files + val_files)
all_train_labels = np.array(np.concatenate([train_labels, val_labels]))


k = 5
skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
fold_results = []

test_dataset = loader_dino_cv.TeethDataset(test_files, test_labels, transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

confmat = MulticlassConfusionMatrix(num_classes=9)



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

            features = model.forward_features(images)
            features = features[:, 0, :]
            outputs = model.linear(features)
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


model_save_directory = '/home/gpu/Workspace/youmin/Learning-by-Asking/LBA/disease_multiclassclassification/checkpoints/disease_center/9_saved_dinovit14b_rein_kfold_5'

if not os.path.exists(model_save_directory):
    os.makedirs(model_save_directory)


models = []

for fold, (train_idx, val_idx) in enumerate(skf.split(all_train_files, all_train_labels)):
    print(f"Training fold {fold+1}/{k}")

    train_files, train_labels = all_train_files[train_idx], all_train_labels[train_idx]
    val_files, val_labels = all_train_files[val_idx], all_train_labels[val_idx]

    train_dataset = loader_dino_cv.TeethDataset(train_files, train_labels, transform, augment=True)
    val_dataset = loader_dino_cv.TeethDataset(val_files, val_labels, transform)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    ####################model initialization####################

    model = model_dino_vit14b_rein.CustomDINOV2(num_classes=9, depth=12, embed_dim=768, patch_size=16).to(device)
     
    class_counts = torch.zeros(len(categories), dtype=torch.int64)
    for _, labels in train_loader:
        for label in labels:
            class_counts[label] += 1
    max_count = float(class_counts.max())
    weights = max_count / class_counts
    weights_tensor = weights.float().to(device)

    optimizer = Adam(model.parameters(), lr=1e-3)   
    criterion = nn.CrossEntropyLoss(weight= weights_tensor)


    for epoch in range(32):
        model.train()
        running_loss = 0.0
        corrects = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            features = model.forward_features(images)
            features = features[:, 0, :]

            outputs = model.linear(features)
            preds = outputs.argmax(dim=1)

            loss = criterion(outputs, labels)
            loss.backward()
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
        print(f"Fold {fold+1}, Epoch {epoch+1}, Loss:{epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}, Val_Loss: {val_loss:.4f}, Val_Accuracy: {val_acc:.4f}")

        model_save_name = f'fold_{fold+1}_epoch_{epoch+1}_valloss_{val_loss}_valacc_{val_acc}.pth'
        model_save_path = os.path.join(model_save_directory, model_save_name)
        torch.save({
            'model_state_dict': model.state_dict(),
            'epoch_loss': epoch_loss,
            'epoch_acc': epoch_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        }, model_save_path)

    fold_results.append((val_loss, val_acc))
    models.append(model)


test_dataset = loader_dino_cv.TeethDataset(test_files, test_labels, transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
test_predictions = [model(test_loader) for model in models]  

average_predictions = torch.mean(torch.stack(test_predictions), dim=0)

avg_loss, avg_accuracy = np.mean(fold_results, axis=0)

print(f"Average Validation Loss: {avg_loss:.4f}, Average Accuracy: {avg_accuracy:.4f}")

test_loss, test_accuracy = evaluate_model(model, test_loader, criterion, device)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")







