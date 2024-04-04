import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from collections import defaultdict
from PIL import Image
import glob
import natsort

# train, val, test에 대해 각 클래스 별로 순서대로 구성

class TeethDataset(Dataset):
    def __init__(self, transform=None):
        self.parent_dir = '/home/gpu/Workspace/youmin/Teeth_Image_Segmentation/LBA/cropped_images'
        self.categories = ['cropped_K00_images', 'cropped_K01_images', 'cropped_K02_images', 'cropped_K03_images', 'cropped_K04_images', 
                           'cropped_K05_images', 'cropped_K07_images', 'cropped_K08_images', 'cropped_K09_images'] # 9개의 카테고리
        
        min_size = 300

        folders = ['cropped_K00_images', 'cropped_K01_images', 'cropped_K02_images', 'cropped_K03_images', 'cropped_K04_images', 
                   'cropped_K05_images', 'cropped_K07_images', 'cropped_K08_images', 'cropped_K09_images']

        file_count = defaultdict(int)

        totals = []

        for folder in folders:
            path = f'/home/gpu/Workspace/youmin/Teeth_Image_Segmentation/LBA/cropped_images/{folder}' 
            for filename in os.listdir(path):
                file_count[filename] += 1
                
        '''
        class 별로 image 개수가 매우 상이하기 때문에 multi-label을 가진 image가 적게 나올텐데 이에 대해 어떻게 해결?
        위 문제는 추후에 해결하도록 하고 일단 먼저 실험 ㄱㄱ

        전체 데이터셋 수 설정
        train -> val -> test
        
        '''

        train_size = int(min_size * 0.7)
        val_size = int(min_size * 0.15)
        test_size = min_size - train_size - val_size

        # train_paths = []
        # val_paths = []
        # test_paths = []

        for category in self.categories:
            category_path = os.path.join(self.parent_dir, category)
            file_paths = natsort.natsorted(glob.glob(f"{category_path}/*.png"))
            # print(file_paths)
            globals()[f"{category}_path_train"] = []
            globals()[f"{category}_path_val"] = []
            globals()[f"{category}_path_test"] = []
        

            i = 0
            if len(glob.glob(f"{category_path}/*.png")) > min_size:
                for i in range(train_size):
                    globals()[f"{category}_path_train"].append(file_paths[i])
                    i += 1
                for i in range(val_size):
                    globals()[f"{category}_path_val"].append(file_paths[i+train_size])
                    i += 1
                for i in range(test_size):
                    globals()[f"{category}_path_test"].append(file_paths[i+train_size+val_size])
                    i += 1
            else:
                for i in range(int(len(glob.glob(f"{category_path}/*.png"))*0.7)):
                    globals()[f"{category}_path_train"].append(glob.glob(f"{category_path}/*.png")[i])   
                    i += 1
                for i in range(int(len(glob.glob(f"{category_path}/*.png"))*0.15)):
                    globals()[f"{category}_path_val"].append(glob.glob(f"{category_path}/*.png")[i+int(len(glob.glob(f"{category_path}/*.png"))*0.7)])   
                    i += 1
                for i in range(int(len(glob.glob(f"{category_path}/*.png"))*0.15)):
                    globals()[f"{category}_path_test"].append(glob.glob(f"{category_path}/*.png")[i+int(len(glob.glob(f"{category_path}/*.png"))*0.85)])   
                    i += 1

            # train_paths += globals()[f"{category}_path_train"]
            # val_paths += globals()[f"{category}_path_val"]
            # test_paths += globals()[f"{category}_path_test"]

            # print(globals()[f"{category}_path_train"])
            # print(globals()[f"{category}_path_val"])
            # print(globals()[f"{category}_path_test"])
            # print(len(globals()[f"{category}_path_train"]))
            # print(len(globals()[f"{category}_path_val"]))
            # print(len(globals()[f"{category}_path_test"]))
            print(len(globals()[f"{category}_path_train"]) + len(globals()[f"{category}_path_val"]) + len(globals()[f"{category}_path_test"]))

            
        self.files = []
        self.labels = []  

        self.train_label_dic = {}
        self.train_path_dic = {}
        self.train_dic = {}
        self.val_label_dic = {}
        self.val_path_dic = {}
        self.val_dic = {}
        self.test_label_dic = {}
        self.test_path_dic = {}
        self.test_dic = {}

# train dataset
        for idx, category in enumerate(self.categories):
            for file_path in globals()[f"{category}_path_train"]:
                file_name = os.path.basename(file_path)

                if file_name not in self.train_label_dic:
                    self.train_label_dic[file_name] = np.zeros(len(self.categories))
                self.train_label_dic[file_name][idx] = 1

                if file_name not in self.train_path_dic:
                    self.train_path_dic[file_name] = []
                self.train_path_dic[file_name].append(file_path)

        for file_name, paths in self.val_path_dic.items():
            selected_path = paths[0] # 중복된 파일들 중 첫번째 파일만 선택
            self.train_dic[selected_path] = self.train_label_dic[file_name]

# val dataset
        for idx, category in enumerate(self.categories):
            for file_path in globals()[f"{category}_path_val"]:
                file_name = os.path.basename(file_path)

                if file_name not in self.val_label_dic:
                    self.val_label_dic[file_name] = np.zeros(len(self.categories))
                self.val_label_dic[file_name][idx] = 1

                if file_name not in self.val_path_dic:
                    self.val_path_dic[file_name] = []
                self.val_path_dic[file_name].append(file_path)

        for file_name, paths in self.val_path_dic.items():
            selected_path = paths[0] # 중복된 파일들 중 첫번째 파일만 선택
            self.val_dic[selected_path] = self.train_label_dic[file_name]

# test dataset
        for idx, category in enumerate(self.categories):
            for file_path in globals()[f"{category}_path_test"]:
                file_name = os.path.basename(file_path)

                if file_name not in self.test_label_dic:
                    self.test_label_dic[file_name] = np.zeros(len(self.categories))
                self.test_label_dic[file_name][idx] = 1

                if file_name not in self.test_path_dic:
                    self.test_path_dic[file_name] = []
                self.test_path_dic[file_name].append(file_path)

        for file_name, paths in self.val_path_dic.items():
            selected_path = paths[0] # 중복된 파일들 중 첫번째 파일만 선택
            self.test_dic[selected_path] = self.train_label_dic[file_name]

        # print(self.file_dic)

        
        self.train_files = list(self.train_dic.keys())
        self.train_labels = list(self.train_dic.values())
        self.val_files = list(self.val_dic.keys())
        self.val_labels = list(self.val_dic.values())
        self.test_files = list(self.test_dic.keys())
        self.test_labels = list(self.test_dic.values())

        self.total_files = self.train_files + self.val_files + self.test_files
        self.total_labels = self.train_labels + self.val_labels + self.test_labels

        
        # count_arrays_with_multiple_ones = sum(arr.sum() == 1 for arr in self.labels)
        # print(f"라벨 1개: {count_arrays_with_multiple_ones}")
        # count_arrays_with_multiple_two = sum(arr.sum() == 2 for arr in self.labels)
        # print(f"라벨 2개: {count_arrays_with_multiple_two}")
        # count_arrays_with_multiple_three = sum(arr.sum() == 3 for arr in self.labels)
        # print(f"라벨 3개: {count_arrays_with_multiple_three}")
        # count_arrays_with_multiple_four = sum(arr.sum() >= 4 for arr in self.labels)
        # print(f"라벨 4개 이상: {count_arrays_with_multiple_four}")

        self.transform = transform


    def __len__(self):
        return len(self.total_files)

    def __getitem__(self, idx):
        image_path = self.total_files[idx]
        image = Image.open(image_path)
        label = torch.tensor(self.total_labels[idx], dtype=torch.float)

        
        if self.transform:
            image = self.transform(image)

        return image, label


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset = TeethDataset(transform=transform)

# train, val, test로 나누는 부분 수정해야함

train_size = len(TeethDataset.train_files)
val_size = len(TeethDataset.val_files)
test_size = len(TeethDataset.test_files)
train_dataset = TeethDataset.trai

print((len(dataset), len(train_dataset), len(val_dataset), len(test_dataset)))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


