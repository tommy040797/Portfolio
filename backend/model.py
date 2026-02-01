import torch
import torch.nn as nn
from torchvision import models

class BinaryResNet18(nn.Module):
    def __init__(self, pretrained=True, freeze_backbone=False):
        super(BinaryResNet18, self).__init__()
        
        # Load Pretrained ResNet18
        try:
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            self.model = models.resnet18(weights=weights)
        except AttributeError:
            self.model = models.resnet18(pretrained=pretrained)

        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
        
        
        num_ftrs = self.model.fc.in_features # 512 in resnet18
        
        #break down pretrained model to output
        self.model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.model(x)

if __name__ == '__main__':
    # Test the model
    model = BinaryResNet18()
    print(model)
    
    # Test with dummy input
    x = torch.randn(2, 3, 224, 224)
    output = model(x)
    print(f"Output Shape: {output.shape}") # Should be [2, 1]
