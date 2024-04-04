import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
from model_densenet import DenseNet121
import pandas as pd
from loader_label_5_order import test_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DenseNet121().to(device)

model_save_path = '/home/gpu/Workspace/youmin/Teeth_Image_Segmentation/LBA/disease_detection_MultiLabelCL/saved_densenet121_margin90_5class_0402/model_epoch_30.pth'  # 최고 성능
checkpoint = torch.load(model_save_path)
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()  
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:  
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        preds = torch.sigmoid(outputs) > 0.5
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

all_preds = np.concatenate(all_preds, axis=0)
all_labels = np.concatenate(all_labels, axis=0)

precisions = precision_score(all_labels, all_preds, average=None)
recalls = recall_score(all_labels, all_preds, average=None)
f1_scores = f1_score(all_labels, all_preds, average=None)
accuracies = [accuracy_score(all_labels[:, i], all_preds[:, i]) for i in range(all_preds.shape[1])]

labels = [f'Label {i}' for i in range(all_preds.shape[1])]
print("Label-wise Performance Metrics:")
exact_match_ratio = np.all(all_preds == all_labels, axis=1).mean()
print(f"Exact Match Ratio: {exact_match_ratio:.4f}")
metrics_df = pd.DataFrame({'Precision': precisions, 'Recall': recalls, 'F1-Score': f1_scores}, index=labels)
metrics_df['Accuracy'] = accuracies
# metrics_df['multi-label Accuracy'] = [exact_match_ratio] * len(metrics_df)  
print(metrics_df)
x = np.arange(len(labels))

plt.figure(figsize=(18, 6))
width = 0.15  
plt.bar(x - width*1.5, precisions, width, label='Precision')
plt.bar(x - width/2, recalls, width, label='Recall')
plt.bar(x + width/2, f1_scores, width, label='F1-Score')
plt.bar(x + width*1.5, accuracies, width, label='Accuracy', color='green')  

plt.xticks(x, labels)
plt.xlabel('Labels')
plt.ylabel('Score')
plt.legend()
plt.title('Precision, Recall, F1-Score, and Accuracy for Each Label')
plt.savefig('metrics_scores_0326.png')
plt.close()


# 마이크로 평균 및 매크로 평균 계산
precision_micro = precision_score(all_labels, all_preds, average='micro')
recall_micro = recall_score(all_labels, all_preds, average='micro')
f1_micro = f1_score(all_labels, all_preds, average='micro')

precision_macro = precision_score(all_labels, all_preds, average='macro')
recall_macro = recall_score(all_labels, all_preds, average='macro')
f1_macro = f1_score(all_labels, all_preds, average='macro')

print("Micro Average Performance Metrics:")
print(f"Micro Precision: {precision_micro:.4f}")
print(f"Micro Recall: {recall_micro:.4f}")
print(f"Micro F1-Score: {f1_micro:.4f}\n")

print("Macro Average Performance Metrics:")
print(f"Macro Precision: {precision_macro:.4f}")
print(f"Macro Recall: {recall_macro:.4f}")
print(f"Macro F1-Score: {f1_macro:.4f}")

# 기존의 성능 메트릭 데이터프레임에 마이크로 평균 및 매크로 평균 추가
metrics_df.loc['Micro Average'] = [precision_micro, recall_micro, f1_micro, np.nan] # Accuracy는 마이크로 평균에서 의미가 없으므로 NaN으로 설정
metrics_df.loc['Macro Average'] = [precision_macro, recall_macro, f1_macro, np.nan]

# print(metrics_df)

