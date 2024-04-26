import torch
import torch.nn as nn
from reins import Reins
from utils_rein import set_requires_grad, set_train

# dinov2_vits14_reg = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')  # backbone
dinov2_vitb14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')

class CustomDINOV2(nn.Module):
    def __init__(self, num_classes=9, depth=12, embed_dim=384, patch_size=16): # 수정
        super(CustomDINOV2, self).__init__()
        # self.transformer = dinov2_vits14_reg
        self.transformer = dinov2_vitb14
        self.linear = nn.Linear(768, num_classes)  
        self.reins = Reins(
            num_layers = depth,
            embed_dims = embed_dim,
            patch_size = patch_size,
        )

    def forward(self, x):
        x = self.transformer(x)
        x = self.linear(x)
        return x
    
    def train(self, mode: bool = True):
        if not mode:
            return super().train(mode)
        set_requires_grad(self, ["reins", "linear"])
        set_train(self, ["reins", "linear"])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = CustomDINOV2(num_classes=9).to(device)
criteria = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)