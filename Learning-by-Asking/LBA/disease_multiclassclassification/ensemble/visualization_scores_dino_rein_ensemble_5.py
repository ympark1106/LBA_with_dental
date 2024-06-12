import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
import loader_dino_cv 
import model_dino_vit14b_rein
from torchmetrics.classification import MulticlassConfusionMatrix
from torchmetrics.classification import MulticlassCalibrationError


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

num_classes = 9
num_models = 2
models = []

model_paths = [
                # '/home/gpu/Workspace/youmin/Learning-by-Asking/LBA/disease_multiclassclassification/checkpoints/disease_center/9_saved_dinovit14b_rein_weight_1/epoch_37_valloss_1.7683531128402268_valacc_0.7647058823529411.pth'
                '/home/gpu/Workspace/youmin/Learning-by-Asking/LBA/disease_multiclassclassification/checkpoints/disease_center/9_saved_dinovit14b_rein_weight_2/epoch_48_valloss_1.82105447765158_valacc_0.7521008403361344.pth'
                # ,'/home/gpu/Workspace/youmin/Learning-by-Asking/LBA/disease_multiclassclassification/checkpoints/disease_center/9_saved_dinovit14b_rein_weight_3/epoch_31_valloss_1.44916858586694_valacc_0.7563025210084033.pth'
                ,'/home/gpu/Workspace/youmin/Learning-by-Asking/LBA/disease_multiclassclassification/checkpoints/disease_center/9_saved_dinovit14b_rein_weight_4/epoch_42_valloss_1.7171947029458374_valacc_0.7605042016806722.pth'
                # ,'/home/gpu/Workspace/youmin/Learning-by-Asking/LBA/disease_multiclassclassification/checkpoints/disease_center/9_saved_dinovit14b_rein_weight_5/epoch_36_valloss_1.8047609665125859_valacc_0.7689075630252101.pth'
                # ,'/home/gpu/Workspace/youmin/Learning-by-Asking/LBA/disease_multiclassclassification/checkpoints/disease_center/9_saved_dinovit14b_rein_weight_6/epoch_28_valloss_1.689575401267835_valacc_0.7394957983193278.pth'
]



for model_path in model_paths:
    model = model_dino_vit14b_rein.CustomDINOV2(num_classes=num_classes).to(device)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    models.append(model)

test_loader = loader_dino_cv.test_loader  

confmat = MulticlassConfusionMatrix(num_classes=num_classes).to(device)
confmat.reset()

calibration_error = MulticlassCalibrationError(num_classes=9, n_bins=10).to(device)

ece_metric = MulticlassCalibrationError(num_classes=9, n_bins=10, norm='l1').to(device)
# mcce_metric = MulticlassCalibrationError(num_classes=9, n_bins=15, norm='l2').to(device)
mce_metric = MulticlassCalibrationError(num_classes=9, n_bins=10, norm='max').to(device)

ece_metric.reset()
# mcce_metric.reset()
mce_metric.reset()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        batch_logits = torch.zeros(labels.size(0), num_classes, device=device)
        
        for model in models:
            features = model.forward_features(images)
            features = features[:, 0, :]  
            outputs = model.linear(features)
            batch_logits += outputs

        batch_logits /= num_models
        # print(batch_logits)
        batch_probs = torch.softmax(batch_logits, dim=1)
        # preds = batch_probs.argmax(dim=1)
        _, preds = batch_logits.max(1)
        
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

        confmat.update(preds, labels)
        ece_metric.update(batch_probs, labels)
        # mcce_metric.update(batch_probs, labels)
        mce_metric.update(batch_probs, labels)

all_preds = np.concatenate(all_preds)
all_labels = np.concatenate(all_labels)
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
}, index=[f'Class {i}' for i in range(num_classes)])  

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


