import torch
import torch.nn as nn
dinov2_vitb14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14') # backbone
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
class CustomDINOV2(nn.Module):
    def __init__(self, num_classes=9): # 클래스 수 지정
        global hi
        super(CustomDINOV2, self).__init__()
        self.transformer = dinov2_vitb14
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.transformer(x)
        x = self.classifier(x)
        return x

# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# model = CustomDINOV2(num_classes=9).to(device)
# criterion = nn.CrossEntropyLoss() 
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# for name, param in model.named_parameters():
#     if "transformer" in name:
#         param.requires_grad = False
#     print(name, param.requires_grad)

# print(CustomDINOV2())

model = CustomDINOV2(num_classes=9).to(device)

def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

trainable_params = count_trainable_parameters(model)
print(f"Trainable parameters: {trainable_params}")