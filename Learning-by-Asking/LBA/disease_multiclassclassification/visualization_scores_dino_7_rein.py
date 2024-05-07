import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
import loader_dino
import model_dino_vit14b_rein
import model_dino_vit14b_rein_1
from torchmetrics.classification import MulticlassConfusionMatrix

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model = model_dino_vit14b_rein_1.CustomDINOV2(num_classes=7).to(device)
# model = model_resnet50_9.ResNet50(num_classes=9).to(device)
test_loader = loader_dino.test_loader

model_save_path = '/home/gpu/Workspace/youmin/Learning-by-Asking/LBA/disease_multiclassclassification/checkpoints/7_saved_dinovit14b_rein_weight_0501/model_epoch_132_valloss_4.977275657653808_valacc_0.47719298245614034.pth'
checkpoint = torch.load(model_save_path)
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()
all_preds = []
all_labels = []

model.eval()
all_preds = []
all_labels = []

confmat = MulticlassConfusionMatrix(num_classes=7).to(device)
confmat.reset()

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        features = model.forward_features(images)
        features = features[:, 0, :]
        outputs = model.linear(features)
        preds = outputs.argmax(dim=1) 
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())  

        confmat.update(preds, labels)

all_preds = np.concatenate(all_preds, axis=0)
all_labels = np.concatenate(all_labels, axis=0)

test_confmat = confmat.compute()
print("Test Confusion Matrix:")
print(test_confmat)

precision_macro = precision_score(all_labels, all_preds, average='macro')
recall_macro = recall_score(all_labels, all_preds, average='macro')
f1_macro = f1_score(all_labels, all_preds, average='macro')
precision_micro = precision_score(all_labels, all_preds, average='micro')
recall_micro = recall_score(all_labels, all_preds, average='micro')
f1_micro = f1_score(all_labels, all_preds, average='micro')
accuracy = accuracy_score(all_labels, all_preds)

metrics_df_macro = pd.DataFrame({
    'Precision': [precision_macro],
    'Recall': [recall_macro],
    'F1-Score': [f1_macro],
    'Accuracy': [accuracy]
}, index=['Macro Average'])
# metrics_df_micro = pd.DataFrame({
#     'Precision': [precision_micro],
#     'Recall': [recall_micro],
#     'F1-Score': [f1_micro],
#     'Accuracy': [accuracy]
# }, index=['Micro Average'])

print(metrics_df_macro)
# print(metrics_df_micro)


