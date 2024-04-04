import torch
from model import ResNet50  

model = ResNet50()

model_save_path = '/home/gpu/Workspace/youmin/Teeth_Image_Segmentation/LBA/disease_detection/K02_saved_models/model_epoch_30.pth'
checkpoint = torch.load(model_save_path)


if 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)  


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()  


if 'epoch_loss' in checkpoint and 'epoch_acc' in checkpoint:
    print(f"Train - Loss: {checkpoint['epoch_loss']}, Accuracy: {checkpoint['epoch_acc']}")
if 'val_loss' in checkpoint and 'val_acc' in checkpoint:
    print(f"Validation - Loss: {checkpoint['val_loss']}, Accuracy: {checkpoint['val_acc']}")


