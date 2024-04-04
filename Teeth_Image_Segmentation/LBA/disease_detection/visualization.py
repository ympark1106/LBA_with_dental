import torch
import torch.nn.functional as F
from torchvision.transforms import Compose, Normalize, ToPILImage
import numpy as np
import matplotlib.pyplot as plt

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activation = None

        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activation = output

    def save_gradient(self, module, input_grad, output_grad):
        self.gradients = output_grad[0]

    def __call__(self, x):
        self.model.eval()
        output = self.model(x)
        self.model.zero_grad()
        target_class = output.argmax(dim=1).item()
        output[0, target_class].backward()

        gradients = self.gradients
        activations = self.activation
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

        for i in range(activations.size(1)):
            activations[:, i, :, :] *= pooled_gradients[i]
        
        heatmap = torch.mean(activations, dim=1).squeeze()
        heatmap = np.maximum(heatmap.detach().cpu(), 0)
        heatmap /= torch.max(heatmap)

        return heatmap, target_class

def visualize_cam(heatmap, img, alpha=0.5):
    heatmap = heatmap.numpy()
    plt.imshow(img.permute(1, 2, 0))
    plt.imshow(heatmap, cmap='jet', alpha=alpha, interpolation='nearest')
    plt.show()

model.eval()
target_layer = model.model.layer4[2].conv3 

grad_cam = GradCAM(model, target_layer)

images, _ = next(iter(train_loader))
img = images[0].unsqueeze(0).to(device)

# Grad-CAM 실행
heatmap, _ = grad_cam(img)

# 이미지 변환 및 시각화
unloader = Compose([Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225]), ToPILImage()])
img_show = unloader(img.squeeze().cpu())
visualize_cam(heatmap, img[0])
