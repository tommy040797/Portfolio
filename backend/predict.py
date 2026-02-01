import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import argparse
from model import BinaryResNet18

class SkinClassifier:
    def __init__(self, model_path='First_Valid_Model.pth', threshold=0.3):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.threshold = threshold
        
        # Transforms (standard ResNet18 normalization)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        
        # Load Model
        self.model = BinaryResNet18(pretrained=False)
        if not os.path.exists(model_path):
            # Try absolute path relative to this file
            base_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(base_dir, model_path)
            
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model weights not found at {model_path}")
            
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

def cli_predict(image_path, model_path='First_Valid_Model.pth', threshold=0.3):
    try:
        classifier = SkinClassifier(model_path, threshold)
        result = classifier.predict_image(image_path)
        
        if "error" in result:
            print(f"Error: {result['error']}")
            return

        print("\n" + "="*30)
        print(f"Prediction Result")
        print("="*30)
        print(f"Image:      {os.path.basename(image_path)}")
        print(f"Class:      {result['class']}")
        print(f"Confidence: {result['confidence']}")
        print(f"Raw Prob:   {result['raw_prob']:.4f}")
        print("-" * 30)
    except Exception as e:
        print(f"Fatal Error: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict single image class')
    parser.add_argument('--image', type=str, required=True, help='Path to image file')
    parser.add_argument('--model', type=str, default='First_Valid_Model.pth', help='Path to model weights')
    parser.add_argument('--threshold', type=float, default=0.3, help='Classification threshold')
    
    args = parser.parse_args()
    predict(args.image, args.model, args.threshold)
