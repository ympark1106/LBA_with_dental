import numpy as np
import glob
import os
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
import natsort
from sklearn.model_selection import KFold



class TeethDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None, augment=False):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform
        self.augment = augment
        self.augmentation_transforms = transforms.Compose([
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(20),
        ])
    
    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.augment:
            image = self.augmentation_transforms(image)
            
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)



transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def split_data(parent_dir, categories, split_ratios):
    train_files, val_files, test_files = [], [], []
    train_labels, val_labels, test_labels = [], [], []

    for category in categories:
        files = natsort.natsorted(glob.glob(os.path.join(parent_dir, category, '*.png')))
        n_files = len(files)
        n_train = int(n_files * split_ratios['train'])
        n_val = int(n_files * split_ratios['val'])
        n_test = n_files - n_train - n_val
        category_idx = categories.index(category)

        train_files.extend(files[:n_train])
        train_labels.extend([category_idx] * n_train)
        val_files.extend(files[n_train:n_train + n_val])
        val_labels.extend([category_idx] * n_val)
        test_files.extend(files[n_train + n_val:])
        test_labels.extend([category_idx] * n_test)
        # print(test_labels)
        print(f"Category: {category} | Total: {n_files} | Train: {n_train} | Val: {n_val} | Test: {n_test}")

    return train_files, np.array(train_labels), val_files, np.array(val_labels), test_files, np.array(test_labels) 



parent_dir = '/home/gpu/Workspace/youmin/Learning-by-Asking/LBA/cropped_disease_images'
categories = ['cropped_K00_images', 'cropped_K01_images', 'cropped_K02_images',
 # 'cropped_K03_images', 
 'cropped_K04_images', 
'cropped_K05_images', 'cropped_K07_images', 'cropped_K08_images',
                'cropped_K09_images', 'cropped_normal_images']

              
split_ratios = {'train': 0.7, 'val': 0.15, 'test': 0.15}

train_files, train_labels, val_files, val_labels, test_files, test_labels = split_data(parent_dir, categories, split_ratios)

test_dataset = TeethDataset(test_files, test_labels, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

print(f"test 라벨 개수: {len(test_labels)}")




all_train_files = train_files + val_files
all_train_labels = np.concatenate([train_labels, val_labels])

k = 5

kfold = KFold(n_splits=k, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(kfold.split(all_train_files)):
    fold_train_files = [all_train_files[i] for i in train_idx]
    fold_train_labels = all_train_labels[train_idx]
    fold_val_files = [all_train_files[i] for i in val_idx]
    fold_val_labels = all_train_labels[val_idx]

    train_dataset = TeethDataset(fold_train_files, fold_train_labels, transform=transform, augment=True)
    val_dataset = TeethDataset(fold_val_files, fold_val_labels, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)

    print(f"Fold {fold+1}: Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

test_dataset = TeethDataset(test_files, test_labels, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)
print(f"Test samples: {len(test_dataset)}")




