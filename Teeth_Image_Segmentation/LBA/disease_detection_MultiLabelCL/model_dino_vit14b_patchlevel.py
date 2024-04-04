import torch
import torch.nn as nn
import torch.nn.functional as F

dinov2_vitb14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')  # backbone

import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomDINOV2(nn.Module):
    def __init__(self, num_classes, num_patches, hidden_dim):
        super().__init__()
        self.dino_backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14') 
        self.num_classes = num_classes
        self.num_patches = num_patches
        self.hidden_dim = hidden_dim
        self.patch_classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        features = self.dino_backbone.get_intermediate_layers(x, n=1)[0]
        
        patch_logits = self.patch_classifier(features[:, 0:, :]) # Skip the [CLS] token
        patch_probs = F.softmax(patch_logits, dim=-1)
        
        image_probs = patch_probs.mean(dim=1)
        return image_probs

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = CustomDINOV2(num_classes=5, num_patches=196, hidden_dim=768).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
