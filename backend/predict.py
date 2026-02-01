import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import argparse
from model import BinaryResNet18

def predict(image_path, model_path='First_Valid_Model.pth', threshold=0.3):
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Transforms (see here https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    # Load Image
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device) # Add batch dimension [1, 3, 224, 224]

    # Load Model
    model = BinaryResNet18(pretrained=False) # No need for ImageNet weights here as we load ours
    
    if not os.path.exists(model_path):
        print(f"Error: Model weights not found at {model_path}")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Inference
    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.sigmoid(output).item()
    
    # Result
    prediction = "Malignant" if prob > threshold else "Benign"
    confidence = prob if prob > threshold else (1 - prob)
    
    print("\n" + "="*30)
    print(f"Prediction Result")
    print("="*30)
    print(f"Image:      {os.path.basename(image_path)}")
    print(f"Class:      {prediction}")
    print(f"Confidence: {confidence*100:.2f}%")
    print(f"Raw Prob:   {prob:.4f}")
    print("-" * 30)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict single image class')
    parser.add_argument('--image', type=str, required=True, help='Path to image file')
    parser.add_argument('--model', type=str, default='First_Valid_Model.pth', help='Path to model weights')
    parser.add_argument('--threshold', type=float, default=0.3, help='Classification threshold')
    
    args = parser.parse_args()
    predict(args.image, args.model, args.threshold)
