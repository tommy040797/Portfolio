import sys
import json
import os
import subprocess
import re
from PIL import Image

def process_image(image_path):
    try:
        # 1. Basic Image Info (Pillow)
        with Image.open(image_path) as img:
            base_info = {
                "width": img.width,
                "height": img.height,
                "format": img.format,
                "mode": img.mode,
            }

        # 2. Call predict.py (Subprocess)
        # Note: predict.py prints to stdout, it doesn't return JSON
        pred_process = subprocess.run(
            [sys.executable, "predict.py", "--image", image_path],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        
        prediction_data = {
            "class": "Unknown",
            "confidence": "0%",
            "raw_output": pred_process.stdout
        }

        if pred_process.returncode == 0:
            # Parse the text output using regex
            # Class:      Malignant
            # Confidence: 99.23%
            class_match = re.search(r"Class:\s+(.+)", pred_process.stdout)
            conf_match = re.search(r"Confidence:\s+(.+)", pred_process.stdout)
            
            if class_match:
                prediction_data["class"] = class_match.group(1).strip()
            if conf_match:
                prediction_data["confidence"] = conf_match.group(1).strip()

        # 3. Combine results
        result = {
            "status": "success",
            **base_info,
            "prediction": prediction_data,
            "message": f"Analyse abgeschlossen für: {os.path.basename(image_path)}"
        }
        return result
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"status": "error", "message": "No image path provided"}))
        sys.exit(1)
    
    image_path = sys.argv[1]
    result = process_image(image_path)
    print(json.dumps(result))
