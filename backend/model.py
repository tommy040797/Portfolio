import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os

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



# Objekt um model in Memory zu halten um Analysezeiten in den Griff zu bekommen
class SkinClassifier:
    def __init__(self, model_path='First_Valid_Model.pth', threshold=0.3):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.threshold = threshold
        
        # Transforms (standard ResNet18 normalization, wie im Originalmodell)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        
        
        self.model = BinaryResNet18(pretrained=False)
        if not os.path.exists(model_path):
            # Try absolute path relative to this file
            base_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(base_dir, model_path)
            
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model weights not found at {model_path}")

        # Load Model   
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded successfully on {self.device}")

    def predict_image(self, image_path):
        if not os.path.exists(image_path):
            return {"error": f"Image not found at {image_path}"}

        try:
            image = Image.open(image_path).convert('RGB')
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)

            # Inference
            with torch.no_grad():
                output = self.model(input_tensor)
                prob = torch.sigmoid(output).item()
            
            prediction = "Malignant" if prob > self.threshold else "Benign"
            confidence = prob if prob > self.threshold else (1 - prob)
            
            return {
                "class": prediction,
                "confidence": f"{confidence*100:.2f}%",
                "raw_prob": prob
            }
        except Exception as e:
            return {"error": str(e)}

