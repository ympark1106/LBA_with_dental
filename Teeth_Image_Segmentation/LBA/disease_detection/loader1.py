import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image

class TeethDataset(Dataset):
    def __init__(self, decay_dir, normal_dir, transform=None):
        self.decay_files = [os.path.join(decay_dir, file) for file in os.listdir(decay_dir)]
        self.normal_files = [os.path.join(normal_dir, file) for file in os.listdir(normal_dir)]
        self.all_files = self.decay_files + self.normal_files
        self.labels = [1] * len(self.decay_files) + [0] * len(self.normal_files)
        self.transform = transform

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        image_path = self.all_files[idx]
        image = Image.open(image_path)
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  
])

disease_images_dir = '/home/gpu/Workspace/youmin/Teeth_Image_Segmentation/LBA/cropped_K02_images'
normal_images_dir = '/home/gpu/Workspace/youmin/Teeth_Image_Segmentation/LBA/not_K02_images'

dataset = TeethDataset(disease_images_dir, normal_images_dir, transform=transform)

train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# print(len(train_dataset), len(val_dataset), len(test_dataset))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
