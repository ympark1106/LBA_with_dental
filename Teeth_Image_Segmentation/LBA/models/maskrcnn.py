import torch.nn as nn

from torchvision.models.detection.mask_rcnn import maskrcnn_resnet50_fpn_v2, MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

class dental(nn.Module):
    def __init__(self, num_classes):
        super(dental, self).__init__()
        
        # MaskRCNN_ResNet50_FPN_Weights.DEFAULT
        self.model = maskrcnn_resnet50_fpn_v2(weights="DEFAULT",
                                              trainable_backbone_layers = 5)
        
        # get number of input features for the classifier
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # now get the number of input features for the mask classifier
        in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        # and replace the mask predictor with a new one
        self.model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask,
            hidden_layer,
            num_classes,
        )
        
    def forward(self, x, target=None):
        
        output = self.model(x, target)
        
        return output