import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from collections import defaultdict
from PIL import Image
import glob

# 정상 이미지 없이 질병 카테고리 9개 분류

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
                

        files_more_than_once = [file for file, count in file_count.items() if count >= 2]
        # print(f"2번 이상 등장하는 파일의 개수: {len(files_more_than_once)}")
        # print(files_more_than_once)
        # exit()


        for category in self.categories:
            category_path = os.path.join(self.parent_dir, category)
            file_paths = glob.glob(f"{category_path}/*.png")
            globals()[f"{category}_path"] = []
            globals()[f"{category}_path_duplicate"] = []
            globals()[f"{category}_path_no_duplicate"] = []
            globals()[f"{category}_path_no_duplicate_random"] = []
            print(category, len(glob.glob(f"{category_path}/*.png")))
        
            if len(glob.glob(f"{category_path}/*.png")) > min_size:
                for file_more_than_once in files_more_than_once:
                    for file_path in file_paths:
                        if file_more_than_once in file_path:
                            # print(file_path)
                            # print(file_more_than_once)
                            globals()[f"{category}_path_duplicate"].append(file_path)
                # print(globals()[f"{category}_path_duplicate"])
                # print(len(globals()[f"{category}_path_duplicate"]))
                if len(globals()[f"{category}_path_duplicate"]) > min_size:
                    globals()[f"{category}_path"] = random.sample(globals()[f"{category}_path_duplicate"], min_size)
                    # print(len(globals()[f"{category}_path"]))
                else:
                    globals()[f"{category}_path"].extend(globals()[f"{category}_path_duplicate"])
                    print("중복된 파일들", len(globals()[f"{category}_path"]))
                    min_size_remain = min_size - len(globals()[f"{category}_path_duplicate"])   
                    globals()[f"{category}_path_no_duplicate"] = [x for x in glob.glob(f"{category_path}/*.png") 
                                                                    if x not in globals()[f"{category}_path_duplicate"]]   
                    # print(len(globals()[f"{category}_path_no_duplicate"]))
                    globals()[f"{category}_path_no_duplicate_random"] = random.sample(globals()[f"{category}_path_no_duplicate"], min_size_remain)       
                    print("unique 파일들", len(globals()[f"{category}_path_no_duplicate_random"]))
                    globals()[f"{category}_path"] = globals()[f"{category}_path"] + globals()[f"{category}_path_no_duplicate_random"]
                    # print(len(globals()[f"{category}_path"]))

            else:
                globals()[f"{category}_path"] = glob.glob(f"{category_path}/*.png")

            totals += globals()[f"{category}_path"]

            # print(globals()[f"{category}_path"])
            print(len(globals()[f"{category}_path"]))
            # random_file_paths.extend(globals()[f"{category}_path"])
            # print(len(totals))
        # print(totals)

             
        # file_count = defaultdict(int)
        # names = []

        # for total in totals:
        #     names.append(total.split("/")[-1])
        # for name in names:
        #     for total in totals:
        #         if name in total:
        #             file_count[name] += 1
        # files_twice = [file for file, count in file_count.items() if count == 2]
        # print(f"2번 등장하는 파일의 개수: {len(files_twice)}")
        # files_three = [file for file, count in file_count.items() if count == 3]
        # print(f"3번 등장하는 파일의 개수: {len(files_three)}")
        # files_four = [file for file, count in file_count.items() if count == 4]
        # print(f"4번 이상 등장하는 파일의 개수: {len(files_four)}")


        # print(random_file_paths)

            # print(category, len(globals()[f"{category}_path"]))
            # print(len(globals()[f"{category}_path"]))
            # print(len(random_file_paths))

        self.files = []
        self.labels = []  
        self.file_label_dic = {}
        self.file_path_dic = {}
        self.file_dic = {}


        for idx, category in enumerate(self.categories):
            for file_path in globals()[f"{category}_path"]:
                file_name = os.path.basename(file_path)

                if file_name not in self.file_label_dic:
                    self.file_label_dic[file_name] = np.zeros(len(self.categories))
                self.file_label_dic[file_name][idx] = 1

                if file_name not in self.file_path_dic:
                    self.file_path_dic[file_name] = []
                self.file_path_dic[file_name].append(file_path)

        for file_name, paths in self.file_path_dic.items():
            selected_path = paths[0] # 중복된 파일들 중 첫번째 파일만 선택
            self.file_dic[selected_path] = self.file_label_dic[file_name]

        # print(len(self.file_path_dic))
        # print(len(self.file_dic))
        # print(self.file_label_dic)
        # print(self.file_path_dic)        
        
        print(self.file_dic)
    
        
        self.files = list(self.file_dic.keys())
        self.labels = list(self.file_dic.values())

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

print((len(dataset), len(train_dataset), len(val_dataset), len(test_dataset)))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)