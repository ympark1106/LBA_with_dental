import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
import loader_dino
import model_dino_vit14b_rein
from torchmetrics.classification import MulticlassConfusionMatrix
from torchmetrics.classification import MulticlassCalibrationError

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

model = model_dino_vit14b_rein.CustomDINOV2(num_classes=9).to(device)
# model = model_resnet50_9.ResNet50(num_classes=9).to(device)
test_loader = loader_dino.test_loader

model_save_path = '/home/gpu/Workspace/youmin/Learning-by-Asking/LBA/disease_multiclassclassification/checkpoints/disease_center/9_saved_dinovit14b_rein_weight_0514/epoch_26_valloss_1.6771559279779864_valacc_0.7563025210084033.pth'

checkpoint = torch.load(model_save_path)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
all_preds = []
all_labels = []

confmat = MulticlassConfusionMatrix(num_classes=9).to(device)
confmat.reset()

calibration_error = MulticlassCalibrationError(num_classes=9, n_bins=10).to(device)

ece_metric = MulticlassCalibrationError(num_classes=9, n_bins=10, norm='l1').to(device)
# mcce_metric = MulticlassCalibrationError(num_classes=9, n_bins=15, norm='l2').to(device)
mce_metric = MulticlassCalibrationError(num_classes=9, n_bins=10, norm='max').to(device)

ece_metric.reset()
# mcce_metric.reset()
mce_metric.reset()


with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        features = model.forward_features(images)
        features = features[:, 0, :]
        outputs = model.linear(features)
        probabilities = torch.softmax(outputs, dim=1)

        preds = outputs.argmax(dim=1) 
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())  

        confmat.update(preds, labels)
        ece_metric.update(probabilities, labels)
        # mcce_metric.update(probabilities, labels)
        mce_metric.update(probabilities, labels)
       

all_preds = np.concatenate(all_preds, axis=0)
all_labels = np.concatenate(all_labels, axis=0)
test_confmat = confmat.compute()

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

print(metrics_df_macro)


