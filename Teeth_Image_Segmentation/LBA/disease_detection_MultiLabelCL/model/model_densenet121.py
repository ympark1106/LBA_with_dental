import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.models import densenet121, DenseNet121_Weights 
from torchsummary import summary

class DenseNet121(nn.Module):
    def __init__(self, num_classes=5):
        super(DenseNet121, self).__init__()
        weights = DenseNet121_Weights.DEFAULT
        self.model = densenet121(weights=weights)
        num_ftrs = self.model.classifier.in_features
        self.model.classifier = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x

model = DenseNet121()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)


