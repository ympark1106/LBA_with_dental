import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import glob

# 정상 이미지 없이 질병 카테고리 9개 분류

class TeethDataset(Dataset):
    def __init__(self, transform=None):
        self.parent_dir = '/home/gpu/Workspace/youmin/Teeth_Image_Segmentation/LBA/cropped_images'
        self.categories = ['cropped_K00_images', 'cropped_K01_images', 'cropped_K02_images', 'cropped_K03_images', 'cropped_K04_images', 
                           'cropped_K05_images', 'cropped_K07_images', 'cropped_K08_images', 'cropped_K09_images'] # 9개의 카테고리
        
        count = 0

        for category in self.categories:
            category_path = os.path.join(self.parent_dir, category)
            print(category, len(glob.glob(f"{category_path}/*.png")))
            count += len(glob.glob(f"{category_path}/*.png"))
        print(count)


            # if len(glob.glob(f"{category_path}/*.png")) < 300:
            #     print(f"{category} 카테고리의 이미지 수가 300개 미만입니다.")

        self.files = []
        self.labels = []  
        self.file_label_dic = {}
        self.file_path_dic = {}
        self.file_dic = {}

        # min_size = 300

        for idx, category in enumerate(self.categories):
            category_path = os.path.join(self.parent_dir, category)
            for file_path in glob.glob(f"{category_path}/*.png"):
                file_name = os.path.basename(file_path)

                if file_name not in self.file_label_dic:
                    self.file_label_dic[file_name] = np.zeros(len(self.categories))
                self.file_label_dic[file_name][idx] = 1

                if file_name not in self.file_path_dic:
                    self.file_path_dic[file_name] = []
                self.file_path_dic[file_name].append(file_path)

        

        for file_name, paths in self.file_path_dic.items():
            selected_path = paths[0]
            self.file_dic[selected_path] = self.file_label_dic[file_name]

        # print(self.file_dic)

        print(len(self.file_path_dic))
        print(len(self.file_dic))
        # print(self.file_label_dic)
        # print(self.file_path_dic)        
        
        # print(self.file_dic)

        # self.balanced_file_label_dic = {}
        # category_files = [[] for _ in range(len(self.categories))] # 9개의 카테고리에 대한 파일 리스트

        # for file_path, labels in self.file_dic.items(): # 파일명과 라벨을 가져옴
        #     # print(file_path, labels)
        #     for idx, label in enumerate(labels):
        #         if label == 1 and len(category_files[idx]) <= min_size:  
        #             category_files[idx].append(file_path)
        #             # print(category_files[idx])

        # for idx, files in enumerate(category_files):
        #     for file_path in files:
        #         if file_path not in self.balanced_file_label_dic: 
        #             self.balanced_file_label_dic[file_path] = self.file_dic[file_path]


        
        self.files = list(self.file_dic.keys())
        self.labels = list(self.file_dic.values())

        # self.file_names = [os.path.basename(file) for file in self.files]
        # print(self.file_names)

        # self.unique_file_names = []
        # for v in self.file_names:
        #     if v not in self.unique_file_names:
        #         self.unique_file_names.append(v)
        # print(self.unique_file_names)

        # print("unique_file_names :", len(list(set(self.file_names))))

        # print(self.labels)
        
        # print(self.files)
        print(len(self.files))
        
        count_arrays_with_multiple_ones = sum(arr.sum() == 1 for arr in self.labels)
        print(f"라벨 1개: {count_arrays_with_multiple_ones}")
        count_arrays_with_multiple_two = sum(arr.sum() == 2 for arr in self.labels)
        print(f"라벨 2개: {count_arrays_with_multiple_two}")
        count_arrays_with_multiple_three = sum(arr.sum() == 3 for arr in self.labels)
        print(f"라벨 3개: {count_arrays_with_multiple_three}")
        count_arrays_with_multiple_four = sum(arr.sum() >= 4 for arr in self.labels)
        print(f"라벨 4개 이상: {count_arrays_with_multiple_four}")

        self.transform = transform


    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image_path = self.files[idx]
        image = Image.open(image_path)
        label = torch.tensor(self.labels[idx], dtype=torch.float)

        
        if self.transform:
            image = self.transform(image)

        return image, label


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset = TeethDataset(transform=transform)

train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# print(len(train_dataset), len(val_dataset), len(test_dataset))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)