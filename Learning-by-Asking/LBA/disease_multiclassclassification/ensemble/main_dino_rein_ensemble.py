import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import loader_dino  
import model_dino_vit14b_rein_ensemble 
from torchmetrics.classification import MulticlassConfusionMatrix

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

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
model3 = model_dino_vit14b_rein_ensemble.CustomDINOV2(num_classes=9, depth=12, embed_dim=768, patch_size=16).to(device)

optimizer1 = Adam(model1.parameters(), lr=1e-3, weight_decay=1e-5)
optimizer2 = Adam(model2.parameters(), lr=1e-3, weight_decay=1e-5)
optimizer3 = Adam(model2.parameters(), lr=1e-3, weight_decay=1e-5)

lr_decay = [int(0.5*max_epoch), int(0.75*max_epoch), int(0.9*max_epoch)]
scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer1, lr_decay)
scheduler2 = torch.optim.lr_scheduler.MultiStepLR(optimizer2, lr_decay)
scheduler3 = torch.optim.lr_scheduler.MultiStepLR(optimizer2, lr_decay)

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
    for model in [model1, model2]:
        model.eval()
    total_loss = 0
    total_corrects = 0
    total_samples = 0

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

            total_loss += loss.item() * images.size(0)
            total_samples += labels.size(0)

            _, predicted = output[:len(labels)].max(1)
            total_corrects += (predicted == labels).sum().item()

            total_loss += loss.item() * images.size(0)
            total_corrects += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            if confmat is not None:
                confmat.update(predicted.int().clone().detach().cpu(), labels.int().to(device='cpu'))

    avg_loss = total_loss / total_samples
    accuracy = total_corrects / total_samples
    return avg_loss, accuracy


def print_confusion_matrix(confmat):
    cm = confmat.compute()
    print(cm)
    confmat.reset() 


##train##
for epoch in range(max_epoch):
    model1.train()
    model2.train()
    total_loss1 = 0.0
    total_loss2 = 0.0
    total_loss3 = 0.0
    total_corrects1 = 0
    total_corrects2 = 0
    total_corrects3 = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        # optimizer.zero_grad()

        features1 = model1.forward_features(images)
        features1 = features1[:, 0, :]
        output1 = model1.linear(features1)

        features2 = model2.forward_features(images)
        features2 = features2[:, 0, :]
        output2 = model2.linear(features2)

        # preds = (output1+output2).max(1).indices
        # linear_acc = (preds == labels)
        
        # loss1 = linear_acc * criterion(output1, labels)
        loss1 = criterion(output1, labels)
        optimizer1.zero_grad()
        loss1.backward()
        optimizer1.step()

        loss2 = criterion(output2, labels)
        optimizer2.zero_grad()
        loss2.backward()
        optimizer2.step()

        total_loss1 += loss1.item() * images.size(0)
        total_loss2 += loss2.item() * images.size(0)
        total += labels.size(0)

        _, predicted = output1[:len(labels)].max(1)
        total_corrects1 += (predicted == labels).sum().item()

        _, predicted = output2[:len(labels)].max(1)
        total_corrects2 += (predicted == labels).sum().item()

    avg_loss1 = total_loss1 / total
    avg_loss2 = total_loss2 / total
    avg_loss = (avg_loss1 + avg_loss2) / 2

    avg_accuracy1 = total_corrects1 / total
    avg_accuracy2 = total_corrects2 / total
    avg_acc = (avg_accuracy1 + avg_accuracy2) / 2
        # loss = loss1.mean() + loss2.mean()
        # loss.backward()
        # optimizer1.step()
        # optimizer2.step()

    val_loss, val_acc = evaluate_model(model1, model2, model3, val_loader, criterion, device, confmat)
  
    val_confmat = MulticlassConfusionMatrix(num_classes=9) # 수정
    val_loss, val_acc = evaluate_model(model1, model2, model3, val_loader, criterion, device, val_confmat)
    print("Validation Confusion Matrix:")
    print_confusion_matrix(val_confmat)

    model_save_name = f'epoch_{epoch+1}_valloss_{val_loss}_valacc_{val_acc}.pth'
    model_save_path = os.path.join(model_save_directory, model_save_name)
    torch.save({
        'model1_state_dict': model1.state_dict(),
        'model2_state_dict': model2.state_dict(),
        'epoch_loss': avg_loss,
        'epoch_acc': avg_acc,
        'val_loss': val_loss,
        'val_acc': val_acc
    }, model_save_path)

    scheduler1.step()
    scheduler2.step()


    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")
    print(scheduler1.get_last_lr())


final_model_path = os.path.join(model_save_directory, 'final_model.pth')
torch.save(model1.state_dict(), final_model_path)
torch.save(model2.state_dict(), final_model_path)
torch.save(model3.state_dict(), final_model_path)
print(f"Model saved to {final_model_path}")

test_confmat = MulticlassConfusionMatrix(num_classes=9)
test_loss, test_acc = evaluate_model(model1, model2, test_loader, criterion, device, test_confmat)
print("Test Confusion Matrix:")
print_confusion_matrix(test_confmat)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")



