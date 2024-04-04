import numpy as np
import glob
import os
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from PIL import Image
import torch
import natsort

class TeethDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        image = Image.open(img_path).convert('RGB') 
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32)
    

def split_data(parent_dir, categories, split_ratios):
    file_label_dic = {}
    file_path_dic = {}
    file_dic = {}

    for idx, category in enumerate(categories):
        globals()[f"{category}_path"] = natsort.natsorted(glob.glob(os.path.join(parent_dir, category, '*.png')))
        for file_path in globals()[f"{category}_path"]:
            file_name = os.path.basename(file_path)

            if file_name not in file_label_dic:
                file_label_dic[file_name] = np.zeros(len(categories))
            file_label_dic[file_name][idx] = 1

            if file_name not in file_path_dic:
                file_path_dic[file_name] = []
            file_path_dic[file_name].append(file_path)

    for file_name, paths in file_path_dic.items():
        selected_path = paths[0] # 중복된 파일들 중 첫번째 파일만 선택
        file_dic[selected_path] = file_label_dic[file_name]

    print(len(file_dic))

    for category in categories:
        globals()[f"{category}_file_dic"] = {}
        for file_path, label in file_dic.items():
            if category in file_path:  
                globals()[f"{category}_file_dic"][file_path] = label

    for category in categories:
        print(f"{category}_file_dic: {len(globals()[f'{category}_file_dic'])}")
        # print(f"{category}_file_dic: {globals()[f'{category}_file_dic']}")

    train_files, val_files, test_files = [], [], []
    train_labels, val_labels, test_labels = [], [], []

    for category in categories:
        n_files = len(globals()[f"{category}_file_dic"])
        n_train = int(n_files * split_ratios['train'])
        n_val = int(n_files * split_ratios['val'])
        n_test = n_files - n_train - n_val
        print(f"{category}: {n_files}, {n_train}, {n_val}, {n_test}")

        i = 0
        for i in range(len(globals()[f"{category}_file_dic"])):
            if i < n_train:
                train_files.append(list(globals()[f"{category}_file_dic"].keys())[i])
                train_labels.append(list(globals()[f"{category}_file_dic"].values())[i])
                i+=1
            elif i < n_train + n_val:
                val_files.append(list(globals()[f"{category}_file_dic"].keys())[i])
                val_labels.append(list(globals()[f"{category}_file_dic"].values())[i])
                i+=1
            else:
                test_files.append(list(globals()[f"{category}_file_dic"].keys())[i])
                test_labels.append(list(globals()[f"{category}_file_dic"].values())[i])
                i+=1


    print(len(train_files), len(val_files), len(test_files))
    # print(train_files)
    # print(val_files)
    # print(test_files)
    return train_files, np.array(train_labels), val_files, np.array(val_labels), test_files, np.array(test_labels)


parent_dir = '/home/gpu/Workspace/youmin/Teeth_Image_Segmentation/LBA/cropped_images'
categories = ['cropped_K00_images', 'cropped_K01_images', 'cropped_K02_images', 'cropped_K03_images', 'cropped_K04_images', 
              'cropped_K05_images', 'cropped_K07_images', 'cropped_K08_images', 'cropped_K09_images']
split_ratios = {'train': 0.7, 'val': 0.15, 'test': 0.15}

train_files, train_labels, val_files, val_labels, test_files, test_labels = split_data(parent_dir, categories, split_ratios)
all_files = np.array(train_files + val_files)
all_labels = np.concatenate([train_labels, val_labels], axis=0)

mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)

count_arrays_with_multiple_ones = sum(arr.sum() == 1 for arr in train_labels)
print(f"train라벨 1개: {count_arrays_with_multiple_ones}")
count_arrays_with_multiple_two = sum(arr.sum() == 2 for arr in train_labels)
print(f"train라벨 2개: {count_arrays_with_multiple_two}")
count_arrays_with_multiple_three = sum(arr.sum() == 3 for arr in train_labels)
print(f"train라벨 3개: {count_arrays_with_multiple_three}")
count_arrays_with_multiple_four = sum(arr.sum() >= 4 for arr in train_labels)
print(f"train라벨 4개 이상: {count_arrays_with_multiple_four}")

count_arrays_with_multiple_ones = sum(arr.sum() == 1 for arr in val_labels)
print(f"val라벨 1개: {count_arrays_with_multiple_ones}")
count_arrays_with_multiple_two = sum(arr.sum() == 2 for arr in val_labels)
print(f"val라벨 2개: {count_arrays_with_multiple_two}")
count_arrays_with_multiple_three = sum(arr.sum() == 3 for arr in val_labels)
print(f"val라벨 3개: {count_arrays_with_multiple_three}")
count_arrays_with_multiple_four = sum(arr.sum() >= 4 for arr in val_labels)
print(f"val라벨 4개 이상: {count_arrays_with_multiple_four}")

count_arrays_with_multiple_ones = sum(arr.sum() == 1 for arr in test_labels)
print(f"test라벨 1개: {count_arrays_with_multiple_ones}")
count_arrays_with_multiple_two = sum(arr.sum() == 2 for arr in test_labels)
print(f"test라벨 2개: {count_arrays_with_multiple_two}")
count_arrays_with_multiple_three = sum(arr.sum() == 3 for arr in test_labels)
print(f"test라벨 3개: {count_arrays_with_multiple_three}")
count_arrays_with_multiple_four = sum(arr.sum() >= 4 for arr in test_labels)
print(f"test라벨 4개 이상: {count_arrays_with_multiple_four}")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

for fold, (train_idx, val_idx) in enumerate(mskf.split(all_files, all_labels)):
    train_fold_files = all_files[train_idx]
    val_fold_files = all_files[val_idx]
    train_fold_labels = all_labels[train_idx]
    val_fold_labels = all_labels[val_idx]

    train_dataset_fold = TeethDataset(train_fold_files, train_fold_labels, transform)
    val_dataset_fold = TeethDataset(val_fold_files, val_fold_labels, transform)
    train_loader_fold = DataLoader(train_dataset_fold, batch_size=32, shuffle=True)
    val_loader_fold = DataLoader(val_dataset_fold, batch_size=32, shuffle=False)

# train_dataset = TeethDataset(train_files, train_labels, transform)
# val_dataset = TeethDataset(val_files, val_labels, transform)
test_dataset = TeethDataset(test_files, test_labels, transform)

# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    
