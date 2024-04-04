import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import numpy as np
import random

class TeethDataset(Dataset):
    def __init__(self, K00_dir, K01_dir, K02_dir, K03_dir, K04_dir, K05_dir, K07_dir, K08_dir, K09_dir, other_dir, transform=None):
        self.K00_files = [os.path.join(K00_dir, file) for file in os.listdir(K00_dir)]
        self.K01_files = [os.path.join(K01_dir, file) for file in os.listdir(K01_dir)]
        self.K02_files = [os.path.join(K02_dir, file) for file in os.listdir(K02_dir)]
        self.K03_files = [os.path.join(K03_dir, file) for file in os.listdir(K03_dir)]
        self.K04_files = [os.path.join(K04_dir, file) for file in os.listdir(K04_dir)] 
        self.K05_files = [os.path.join(K05_dir, file) for file in os.listdir(K05_dir)]
        self.K07_files = [os.path.join(K07_dir, file) for file in os.listdir(K07_dir)]
        self.K08_files = [os.path.join(K08_dir, file) for file in os.listdir(K08_dir)]
        self.K09_files = [os.path.join(K09_dir, file) for file in os.listdir(K09_dir)]
        self.other_files = [os.path.join(other_dir, file) for file in os.listdir(other_dir)]

        print("K00: ", len(self.K00_files))
        print("K01: ", len(self.K01_files))
        print("K02: ", len(self.K02_files))
        print("K03: ", len(self.K03_files))
        print("K04: ", len(self.K04_files))
        print("K05: ", len(self.K05_files))
        print("K07: ", len(self.K07_files))
        print("K08: ", len(self.K08_files))
        print("K09: ", len(self.K09_files))
        print("others: ", len(self.other_files))



        # min_size = min(len(self.K01_files), len(self.K02_files), len(self.other_files))
        # print(min_size)
        
        # self.all_files = random.sample(self.K01_files, min_size) + \
        #                  random.sample(self.K02_files, min_size) + \
        #                  random.sample(self.other_files, min_size)
        
        self.all_files = self.K00_files + self.K01_files + self.K02_files + self.K03_files + self.K04_files + self.K05_files + self.K07_files + self.K08_files + self.K09_files + self.other_files

        self.labels = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]] * len(self.K00_files) + \
                        [[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]] * len(self.K01_files) + \
                        [[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]] * len(self.K02_files) + \
                        [[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]] * len(self.K03_files) + \
                        [[0, 0, 0, 0, 1, 0, 0, 0, 0, 0]] * len(self.K04_files) + \
                        [[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]] * len(self.K05_files) + \
                        [[0, 0, 0, 0, 0, 0, 1, 0, 0, 0]] * len(self.K07_files) + \
                        [[0, 0, 0, 0, 0, 0, 0, 1, 0, 0]] * len(self.K08_files) + \
                        [[0, 0, 0, 0, 0, 0, 0, 0, 1, 0]] * len(self.K09_files) + \
                        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]] * len(self.other_files)
        
        self.transform = transform
        self.augmentation_transforms = augmentation_transforms
        


    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        image_path = self.all_files[idx]
        image = Image.open(image_path)
        label = torch.tensor(self.labels[idx], dtype=torch.float)
        
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    

augmentation_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


basic_transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  
])

K00_images_dir = '/home/gpu/Workspace/youmin/Teeth_Image_Segmentation/LBA/cropped_K00_images' # 33 cropped images
K01_images_dir = '/home/gpu/Workspace/youmin/Teeth_Image_Segmentation/LBA/cropped_K01_images' # 34 cropped images
K02_images_dir = '/home/gpu/Workspace/youmin/Teeth_Image_Segmentation/LBA/cropped_K02_images' # 35 cropped images
K03_images_dir = '/home/gpu/Workspace/youmin/Teeth_Image_Segmentation/LBA/cropped_K03_images' # 36 cropped images
K04_images_dir = '/home/gpu/Workspace/youmin/Teeth_Image_Segmentation/LBA/cropped_K04_images' # 37 cropped images
K05_images_dir = '/home/gpu/Workspace/youmin/Teeth_Image_Segmentation/LBA/cropped_K05_images' # 38 cropped images
K07_images_dir = '/home/gpu/Workspace/youmin/Teeth_Image_Segmentation/LBA/cropped_K07_images' # 39 cropped images
K08_images_dir = '/home/gpu/Workspace/youmin/Teeth_Image_Segmentation/LBA/cropped_K08_images' # 40 cropped images
K09_images_dir = '/home/gpu/Workspace/youmin/Teeth_Image_Segmentation/LBA/cropped_K09_images' # 41 cropped images
others_images_dir = '/home/gpu/Workspace/youmin/Teeth_Image_Segmentation/LBA/others_images' 


dataset = TeethDataset(K00_images_dir, K01_images_dir, K02_images_dir, K03_images_dir, K04_images_dir, K05_images_dir, K07_images_dir, K08_images_dir, K09_images_dir, others_images_dir, transform=transform)
# print(len(dataset))

train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# print(len(train_dataset), len(val_dataset), len(test_dataset))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


