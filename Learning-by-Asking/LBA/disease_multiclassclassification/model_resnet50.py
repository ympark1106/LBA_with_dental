import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.models import resnet50, ResNet50_Weights

class ResNet50(nn.Module):
    def __init__(self, num_classes=9): # 수정 필요
        super(ResNet50, self).__init__()
        weights = ResNet50_Weights.DEFAULT  
        self.model = resnet50(weights=weights)
        
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)
    
model = ResNet50()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
