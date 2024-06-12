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
num_folds = 5
models = []

model_paths = [
                '/home/gpu/Workspace/youmin/Learning-by-Asking/LBA/disease_multiclassclassification/checkpoints/disease_center/9_saved_dinovit14b_rein_kfold_0527/fold_1_epoch_12_valloss_1.071162117535577_valacc_0.7720588235294118.pth'  
                ,'/home/gpu/Workspace/youmin/Learning-by-Asking/LBA/disease_multiclassclassification/checkpoints/disease_center/9_saved_dinovit14b_rein_kfold_0527/fold_2_epoch_16_valloss_0.986962376107626_valacc_0.7904411764705882.pth'
                ,'/home/gpu/Workspace/youmin/Learning-by-Asking/LBA/disease_multiclassclassification/checkpoints/disease_center/9_saved_dinovit14b_rein_kfold_0527/fold_3_epoch_13_valloss_1.5587098872309681_valacc_0.7785977859778598.pth'
                ,'/home/gpu/Workspace/youmin/Learning-by-Asking/LBA/disease_multiclassclassification/checkpoints/disease_center/9_saved_dinovit14b_rein_kfold_0527/fold_4_epoch_30_valloss_1.9361113025033605_valacc_0.7822878228782287.pth'
                ,'/home/gpu/Workspace/youmin/Learning-by-Asking/LBA/disease_multiclassclassification/checkpoints/disease_center/9_saved_dinovit14b_rein_kfold_0527/fold_5_epoch_26_valloss_1.0339088641464491_valacc_0.8044280442804428.pth'
]


# model_paths = [
#                 '/home/gpu/Workspace/youmin/Learning-by-Asking/LBA/disease_multiclassclassification/checkpoints/disease_center/9_saved_dinovit14b_rein_kfold_6/fold_1_epoch_20_valloss_1.4084884169868435_valacc_0.7797356828193832.pth'
#                 ,'/home/gpu/Workspace/youmin/Learning-by-Asking/LBA/disease_multiclassclassification/checkpoints/disease_center/9_saved_dinovit14b_rein_kfold_6/fold_2_epoch_23_valloss_1.271700161550318_valacc_0.7920353982300885.pth'
#                 ,'/home/gpu/Workspace/youmin/Learning-by-Asking/LBA/disease_multiclassclassification/checkpoints/disease_center/9_saved_dinovit14b_rein_kfold_6/fold_3_epoch_22_valloss_1.68935990788855_valacc_0.8053097345132744.pth'
#                 ,'/home/gpu/Workspace/youmin/Learning-by-Asking/LBA/disease_multiclassclassification/checkpoints/disease_center/9_saved_dinovit14b_rein_kfold_6/fold_4_epoch_23_valloss_1.7673517336688505_valacc_0.7566371681415929.pth'
#                 ,'/home/gpu/Workspace/youmin/Learning-by-Asking/LBA/disease_multiclassclassification/checkpoints/disease_center/9_saved_dinovit14b_rein_kfold_6/fold_5_epoch_15_valloss_1.665187264751412_valacc_0.7654867256637168.pth'
#                 ,'/home/gpu/Workspace/youmin/Learning-by-Asking/LBA/disease_multiclassclassification/checkpoints/disease_center/9_saved_dinovit14b_rein_kfold_6/fold_6_epoch_29_valloss_0.9434536119192309_valacc_0.8053097345132744.pth'
# ]

# model_paths = [
#                 '/home/gpu/Workspace/youmin/Learning-by-Asking/LBA/disease_multiclassclassification/checkpoints/disease_center/9_saved_dinovit14b_rein_kfold_7/fold_1_epoch_19_valloss_1.2052361228090434_valacc_0.7989690721649485.pth'
#                 ,'/home/gpu/Workspace/youmin/Learning-by-Asking/LBA/disease_multiclassclassification/checkpoints/disease_center/9_saved_dinovit14b_rein_kfold_7/fold_2_epoch_27_valloss_1.3299458360064838_valacc_0.7783505154639175.pth'
#                 ,'/home/gpu/Workspace/youmin/Learning-by-Asking/LBA/disease_multiclassclassification/checkpoints/disease_center/9_saved_dinovit14b_rein_kfold_7/fold_3_epoch_24_valloss_1.5074704780192931_valacc_0.8144329896907216.pth'
#                 ,'/home/gpu/Workspace/youmin/Learning-by-Asking/LBA/disease_multiclassclassification/checkpoints/disease_center/9_saved_dinovit14b_rein_kfold_7/fold_4_epoch_32_valloss_1.856594629373315_valacc_0.7938144329896907.pth'
#                 ,'/home/gpu/Workspace/youmin/Learning-by-Asking/LBA/disease_multiclassclassification/checkpoints/disease_center/9_saved_dinovit14b_rein_kfold_7/fold_5_epoch_16_valloss_1.3279052464694707_valacc_0.7783505154639175.pth'
#                 ,'/home/gpu/Workspace/youmin/Learning-by-Asking/LBA/disease_multiclassclassification/checkpoints/disease_center/9_saved_dinovit14b_rein_kfold_7/fold_6_epoch_32_valloss_1.7800782777215394_valacc_0.7628865979381443.pth'
#                 ,'/home/gpu/Workspace/youmin/Learning-by-Asking/LBA/disease_multiclassclassification/checkpoints/disease_center/9_saved_dinovit14b_rein_kfold_7/fold_7_epoch_25_valloss_0.9982061931583548_valacc_0.8082901554404145.pth'
# ]

for model_path in model_paths:
    model = model_dino_vit14b_rein.CustomDINOV2(num_classes=num_classes).to(device)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    models.append(model)

test_loader = loader_dino_cv.test_loader  

confmat = MulticlassConfusionMatrix(num_classes=num_classes).to(device)
confmat.reset()

calibration_error = MulticlassCalibrationError(num_classes=9, n_bins=15).to(device)

ece_metric = MulticlassCalibrationError(num_classes=9, n_bins=15, norm='l1').to(device)
# mcce_metric = MulticlassCalibrationError(num_classes=9, n_bins=15, norm='l2').to(device)
mce_metric = MulticlassCalibrationError(num_classes=9, n_bins=15, norm='max').to(device)

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

        batch_logits /= num_folds
        batch_probs = torch.softmax(batch_logits, dim=1)
        preds = batch_probs.argmax(dim=1)
        # _, preds = batch_logits.max(1)
        
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

        confmat.update(preds, labels)
        ece_metric.update(batch_logits, labels)
        # mcce_metric.update(batch_logits, labels)
        mce_metric.update(batch_logits, labels)

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


