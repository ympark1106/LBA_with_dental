import torch
import torch.nn as nn
from reins import Reins
from utils_rein import set_requires_grad, set_train

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
dinov2_vitb14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').to(device)

class CustomDINOV2(nn.Module):
    def __init__(self, num_classes=9, depth=12, embed_dim=768, patch_size=16):
        super(CustomDINOV2, self).__init__()
        # self.transformer = dinov2_vitb14
        self.reins = Reins(
            num_layers = depth,
            embed_dims = embed_dim,
            patch_size = patch_size,
        )

        self.linear = nn.Linear(embed_dim, num_classes) 

    def forward_features(self, x, masks=None):
        B, _, h, w = x.shape
        H, W = h // self.reins.patch_size, w // self.reins.patch_size
        x = dinov2_vitb14.prepare_tokens_with_masks(x, masks)
        outs = []
        for idx, blk in enumerate(dinov2_vitb14.blocks):
            x = blk(x)
            x = self.reins.forward(
                x,
                idx,
                batch_first=True,
                has_cls_token=True,
            )
        return x
    
    def forward(self, x):
        x = self.forward_features(x)
        x = self.linear(x)
        return x
    
    def train(self, mode: bool = True):
        if not mode:
            return super().train(mode)
        set_requires_grad(self, ["reins", "linear"])
        set_train(self, ["reins", "linear"])

model = CustomDINOV2(num_classes=9, depth=12, embed_dim=768, patch_size=16).to(device)

def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

trainable_params = count_trainable_parameters(model)
print(f"Trainable parameters: {trainable_params}")
