import os
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torchmetrics.classification import MulticlassConfusionMatrix

import loader_dino
import model_dino_vit14b_rein_ensemble


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = Compose([
    Resize((224, 224)),  
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

model_save_directory = '/home/gpu/Workspace/youmin/Learning-by-Asking/LBA/disease_multiclassclassification/checkpoints/disease_center/9_saved_dinovit14b_rein_ensemble_0514'
if not os.path.exists(model_save_directory):
    os.makedirs(model_save_directory)

parent_dir = '/home/gpu/Workspace/youmin/Learning-by-Asking/LBA/cropped_disease_images'
categories = ['cropped_K00_images', 'cropped_K01_images', 'cropped_K02_images', #'cropped_K03_images', 
'cropped_K04_images', 'cropped_K05_images', 
'cropped_K07_images', 'cropped_K08_images', 
'cropped_K09_images', 'cropped_normal_images'] 

split_ratios = {'train': 0.7, 'val': 0.15, 'test': 0.15}
train_files, train_labels, val_files, val_labels, test_files, test_labels = loader_dino.split_data(parent_dir, categories, split_ratios)

train_dataset = loader_dino.TeethDataset(train_files, train_labels, transform, augment=True)
val_dataset = loader_dino.TeethDataset(val_files, val_labels, transform)
test_dataset = loader_dino.TeethDataset(test_files, test_labels, transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)
test_loader = DataLoader(test_dataset, batch_size=16)


max_epoch = 200

model1 = model_dino_vit14b_rein_ensemble.CustomDINOV2(num_classes=9, depth=12, embed_dim=768, patch_size=16).to(device)
model2 = model_dino_vit14b_rein_ensemble.CustomDINOV2(num_classes=9, depth=12, embed_dim=768, patch_size=16).to(device)

optimizer1 = Adam(model1.parameters(), lr=1e-3, weight_decay=1e-5)
optimizer2 = Adam(model2.parameters(), lr=1e-3, weight_decay=1e-5)

lr_decay = [int(0.5*max_epoch), int(0.75*max_epoch), int(0.9*max_epoch)]

scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer1, lr_decay)
scheduler2 = torch.optim.lr_scheduler.MultiStepLR(optimizer2, lr_decay)

class_counts = torch.zeros(len(categories), dtype=torch.int64)
for _, labels in train_loader:
    for label in labels:
        class_counts[label] += 1

max_count = float(class_counts.max())
weights = max_count / class_counts
weights_tensor = weights.float().to(device)

criterion = nn.CrossEntropyLoss(weight=weights_tensor)

confmat = MulticlassConfusionMatrix(num_classes=9)



def evaluate_model(model1, model2, dataloader, criterion, device, confmat=None):
    model1.eval()
    model2.eval()
    running_loss = 0.0
    corrects = 0
    total = 0

    if confmat is not None:
        confmat.reset()

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            features1 = model1.forward_features(images)
            features1 = features1[:, 0, :]
            output1 = model1.linear(features1)

            features2 = model2.forward_features(images)
            features2 = features2[:, 0, :]
            output2 = model2.linear(features2)

            output = (output1 + output2) / 2

            loss = criterion(output, labels)
            _, predicted = output[:len(labels)].max(1)

            running_loss += loss.item() * images.size(0)
            corrects += (predicted == labels).sum().item()
            total += labels.size(0)

            if confmat is not None:
                confmat.update(predicted.int().clone().detach().cpu(), labels.int().to(device='cpu'))

    final_loss = running_loss / total
    final_acc = float(corrects) / total
    return final_loss, final_acc



def print_confusion_matrix(confmat):
    cm = confmat.compute()
    print(cm)
    confmat.reset() 


model_save_directory = '/home/gpu/Workspace/youmin/Learning-by-Asking/LBA/disease_multiclassclassification/checkpoints/disease_center/9_saved_dinovit14b_rein_boosting_0524' # 'saved_dinovit14b
if not os.path.exists(model_save_directory):
    os.makedirs(model_save_directory)

for epoch in range(max_epoch):
    model1.train()
    model2.train()
    running_loss = 0.0
    corrects = 0
    total = 0


    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        features1 = model1.forward_features(images)
        features1 = features1[:, 0, :]
        output1 = model1.linear(features1)

        features2 = model2.forward_features(images)
        features2 = features2[:, 0, :]
        output2 = model2.linear(features2)

        optimizer1.zero_grad()
        loss1 = criterion(output1, labels)
        loss1.backward()
        optimizer1.step()

        optimizer2.zero_grad()
        loss2 = criterion(output2, labels)
        
        with torch.no_grad():
            errors = (output1.argmax(1) != labels).float() * 2.0  
        loss2 = (loss2 * errors).mean()  

        loss2.backward()
        optimizer2.step()

        _, predicted = output2[:len(labels)].max(1)
        corrects += (predicted == labels).sum().item()
        running_loss += loss2.item() * images.size(0)
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = corrects / total

    val_loss, val_acc = evaluate_model(model1, model2, val_loader, criterion, device, confmat)

    val_confmat = MulticlassConfusionMatrix(num_classes=9)
    val_loss, val_acc = evaluate_model(model1, model2, val_loader, criterion, device, val_confmat)
    print("validation Confusion Matrix: ")
    print_confusion_matrix(val_confmat)

    model_save_name = f'epoch_{epoch+1}_valloss_{val_loss}_valacc_{val_acc}.pth'
    model_save_path = os.path.join(model_save_directory, model_save_name)
    torch.save({
        'model1_state_dict': model1.state_dict(), 
        'model2_state_dict': model2.state_dict(),
        'epoch_loss': epoch_loss,
        'epoch_acc': epoch_acc,
        'val_loss': val_loss,
        'val_acc': val_acc
    }, model_save_path)

    scheduler1.step()
    scheduler2.step()

    print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")
    print(scheduler2.get_last_lr())


final_model_path = os.path.join(model_save_directory, 'final_model.pth')
torch.save(model1.state_dict(), final_model_path)
torch.save(model2.state_dict(), final_model_path)

print(f"Model saved at {final_model_path}")







