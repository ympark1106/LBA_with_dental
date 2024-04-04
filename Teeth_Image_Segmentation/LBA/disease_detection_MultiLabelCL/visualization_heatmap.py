import torch
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix
from model import ResNet50
# from model_densenet import DenseNet121
from loader_label_9_order import test_loader  # 수정
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet50().to(device)
# model = DenseNet121().to(device)


model_save_path = '/home/gpu/Workspace/youmin/Teeth_Image_Segmentation/LBA/disease_detection_MultiLabelCL/saved_resnet50_0328/model_epoch_30.pth'  
checkpoint = torch.load(model_save_path, map_location=device)
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

cm = multilabel_confusion_matrix(all_labels, all_preds)

def plot_confusion_matrix_heatmap(cm, class_names, filename='confusion_matrix_heatmap.png'):

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix Heatmap')
    plt.savefig(filename)
    plt.close()

# 클래스 이름 목록
class_names = ['K00', 'K01', 'K02', 'K03', 'K04', 'K05', 'K07', 'K08', 'K09'] 

for i, cn in enumerate(class_names):
    plot_confusion_matrix_heatmap(cm[i], ['False', 'True'], filename=f'cm_heatmap_label_{i+1}.png')

