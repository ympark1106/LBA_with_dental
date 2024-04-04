import torch
import torch.nn as nn
dinov2_vitl14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14') # backbone

class CustomDINOV2(nn.Module):
    def __init__(self, num_classes=5): # 수정
        super(CustomDINOV2, self).__init__()
        self.transformer = dinov2_vitl14
        self.classifier = nn.Linear(1024, num_classes)  

    def forward(self, x):
        x = self.transformer(x)
        x = self.classifier(x)
        return x

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = CustomDINOV2(num_classes=5).to(device)
# criteria = nn.BCEWithLogitsLoss()
criteria = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.000005)