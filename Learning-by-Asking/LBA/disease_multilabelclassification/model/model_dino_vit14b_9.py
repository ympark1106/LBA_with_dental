import torch
import torch.nn as nn
dinov2_vitb14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14') # backbone
# dinov2_vitb14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14') # backbone


class CustomDINOV2(nn.Module):
    def __init__(self, num_classes=9): # 수정
        super(CustomDINOV2, self).__init__()
        self.transformer = dinov2_vitb14
        self.classifier = nn.Linear(768, num_classes)  

    def forward(self, x):
        x = self.transformer(x)
        x = self.classifier(x)
        return x
    
    def get_last_selfattention(self, x):
        return self.transformer.get_intermediate_layers(x)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model = CustomDINOV2(num_classes=9).to(device)
criteria = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for name, param in model.named_parameters():
    if "transformer" in name:
        param.requires_grad = False
    print(name, param.requires_grad)
# print(model)
