import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
import loader_dino_9
import model_dino_vit14b, model_resnet50_9, model_dino_vit14b_rein, model_dino_vit14b_linearprobe
from torchmetrics.classification import MulticlassConfusionMatrix

device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
model = model_dino_vit14b_rein.CustomDINOV2(num_classes=9).to(device)
# model = model_resnet50_9.ResNet50(num_classes=9).to(device)
test_loader = loader_dino_9.test_loader

model_save_path = '/home/gpu/Workspace/youmin/Learning-by-Asking/LBA/disease_multiclassclassification/checkpoints/9_saved_dinovit14b_rein_0.00001_weight_0425/model_epoch_161_valloss_1.7631923389434814_valacc_0.33666666666666667.pth'
checkpoint = torch.load(model_save_path)
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()
all_preds = []
all_labels = []

model.eval()
all_preds = []
all_labels = []

confmat = MulticlassConfusionMatrix(num_classes=9).to(device)
confmat.reset()

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
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


