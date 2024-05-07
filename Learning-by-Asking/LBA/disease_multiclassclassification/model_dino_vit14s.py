import torch
import torch.nn as nn
dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')  # backbone

class CustomDINOV2(nn.Module):
    def __init__(self, num_classes=9): # 수정
        super(CustomDINOV2, self).__init__()
        self.transformer = dinov2_vits14
        self.classifier = nn.Linear(384, num_classes)  

    def forward(self, x):
        x = self.transformer(x)
        x = self.classifier(x)
        return x

