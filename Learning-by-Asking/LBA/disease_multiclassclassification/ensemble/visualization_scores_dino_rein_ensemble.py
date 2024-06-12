import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn.parallel import DataParallel
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import loader_dino  # Ensure this is linked correctly
import model_dino_vit14b_rein_ensemble  # Ensure correct model import
from torchmetrics.classification import MulticlassConfusionMatrix
from torchmetrics.classification import MulticlassCalibrationError
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import pandas as pd


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model1 = model_dino_vit14b_rein_ensemble.CustomDINOV2(num_classes=9, depth=12, embed_dim=768, patch_size=16).to(device)
model2 = model_dino_vit14b_rein_ensemble.CustomDINOV2(num_classes=9, depth=12, embed_dim=768, patch_size=16).to(device)

model_save_path = '/home/gpu/Workspace/youmin/Learning-by-Asking/LBA/disease_multiclassclassification/checkpoints/disease_center/9_saved_dinovit14b_rein_ensemble_0513/epoch_37_valloss_1.3360934615949123_valacc_0.773109243697479.pth'
checkpoint = torch.load(model_save_path)
model1.load_state_dict(checkpoint['model1_state_dict']) 
model2.load_state_dict(checkpoint['model2_state_dict'])

model1.eval()
model2.eval()

test_loader = loader_dino.test_loader  

all_preds = []
all_labels = []
confmat = MulticlassConfusionMatrix(num_classes=9).to(device)
confmat.reset()

calibration_error = MulticlassCalibrationError(num_classes=9, n_bins=20).to(device)

ece_metric = MulticlassCalibrationError(num_classes=9, n_bins=20, norm='l1').to(device)
# mcce_metric = MulticlassCalibrationError(num_classes=9, n_bins=15, norm='l2').to(device)
mce_metric = MulticlassCalibrationError(num_classes=9, n_bins=20, norm='max').to(device)

ece_metric.reset()
# mcce_metric.reset()
mce_metric.reset()

# Test the models
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        
        features1 = model1.forward_features(images)
        features1 = features1[:, 0, :]
        output1 = model1.linear(features1)

        features2 = model2.forward_features(images)
        features2 = features2[:, 0, :]
        output2 = model2.linear(features2)

        output = (output1 + output2) / 2  
        probabilities = torch.softmax(output, dim=1)
        _, preds = output.max(1)

        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        confmat.update(preds, labels)
        ece_metric.update(probabilities, labels)
        # mcce_metric.update(probabilities, labels)
        mce_metric.update(probabilities, labels)

all_preds = np.concatenate(all_preds)
all_labels = np.concatenate(all_labels)
test_confmat = confmat.compute()

# Display the confusion matrix
print("Test Confusion Matrix:")
print(test_confmat)


ece = ece_metric.compute().item()
# mcce = mcce_metric.compute().item()
mce = mce_metric.compute().item()

print(f"Expected Calibration Error (ECE): {ece:.4f}")
# print(f"Mean Calibration Error (MCCE): {mcce:.4f}")
print(f"Maximum Calibration Error (MCE): {mce:.4f}")


precision_per_class = precision_score(all_labels, all_preds, average=None)
recall_per_class = recall_score(all_labels, all_preds, average=None)
f1_per_class = f1_score(all_labels, all_preds, average=None)


metrics_df_per_class = pd.DataFrame({
    'Precision': precision_per_class,
    'Recall': recall_per_class,
    'F1-Score': f1_per_class
}, index=[f'Class {i}' for i in range(9)])  

print("Metrics per Class:")
print(metrics_df_per_class)

# Calculate and print the performance metrics
precision_macro = precision_score(all_labels, all_preds, average='macro')
recall_macro = recall_score(all_labels, all_preds, average='macro')
f1_macro = f1_score(all_labels, all_preds, average='macro')
accuracy = accuracy_score(all_labels, all_preds)

metrics_df_macro = pd.DataFrame({
    'Precision': [precision_macro],
    'Recall': [recall_macro],
    'F1-Score': [f1_macro],
    'Accuracy': [accuracy]
}, index=['Macro Average'])

print(metrics_df_macro)



