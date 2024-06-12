#REIN adapter 적용했을 때와 Linear Probing했을 때의 모델을 Ensemble

import torch
import numpy as np
import torch.nn.functional as F
import loader_dino
import model_dino_vit14b_rein, model_dino_vit14b_linearprobe
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from torchmetrics.classification import MulticlassConfusionMatrix

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model_rein = model_dino_vit14b_rein.CustomDINOV2(num_classes=9).to(device)
model_lp = model_dino_vit14b_linearprobe.CustomDINOV2(num_classes=9).to(device)

model_rein_save_path = '/home/gpu/Workspace/youmin/Learning-by-Asking/LBA/disease_multiclassclassification/checkpoints/disease_center/9_saved_dinovit14b_rein_weight_0510/model_epoch_24_valloss_1.9435066956744975_valacc_0.7436974789915967.pth'
checkpoint_rein = torch.load(model_rein_save_path)
model_rein.load_state_dict(checkpoint_rein['model_state_dict'])

model_lp_save_path = '/home/gpu/Workspace/youmin/Learning-by-Asking/LBA/disease_multiclassclassification/checkpoints/disease_center/9_saved_dino_linearprobe_weight_0507/model_epoch_143_valloss_1.3367266705056198_valacc_0.6596638655462185.pth'
checkpoint_lp = torch.load(model_lp_save_path)
model_lp.load_state_dict(checkpoint_lp['model_state_dict'])

test_loader = loader_dino.test_loader

def ensemble_predictions(outputs_rein, outputs_lp):
  
    probabilities_rein = F.softmax(outputs_rein, dim=1)
    probabilities_lp = F.softmax(outputs_lp, dim=1)
  
    final_probabilities = (probabilities_rein + probabilities_lp) / 2
    _, predicted = torch.max(final_probabilities, 1)
    return predicted

model_rein.eval()
model_lp.eval()
all_preds = []
all_labels = []

confmat = MulticlassConfusionMatrix(num_classes=9).to(device)
confmat.reset()

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        # Extract features for REIN model and predict
        features_rein = model_rein.forward_features(images)
        features_rein = features_rein[:, 0, :]  # Using [CLS] token's output
        outputs_rein = model_rein.linear(features_rein)

        # Predict using Linear Probing model
        outputs_lp = model_lp(images)

        # Ensemble predictions using soft voting
        preds = ensemble_predictions(outputs_rein, outputs_lp)
        
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
accuracy_macro = accuracy_score(all_labels, all_preds)

metrics_df_macro = pd.DataFrame({
    'Precision': [precision_macro],
    'Recall': [recall_macro],
    'F1-Score': [f1_macro],
    'Accuracy': [accuracy_macro]
}, index=['Macro Average'])

print(metrics_df_macro)
