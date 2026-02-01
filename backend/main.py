from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import subprocess
import os
import uuid
import json
import shutil
import sys

app = FastAPI()

# Allow CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(os.path.dirname(BASE_DIR), "frontend")
UPLOAD_DIR = os.path.join(BASE_DIR, "temp_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Serve the landing page at /
@app.get("/")
async def read_landing():
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail=f"Landing page index.html not found at {index_path}")
    return FileResponse(index_path)

# Serve the scanner page at /scanner
@app.get("/scanner")
async def read_scanner():
    scanner_path = os.path.join(FRONTEND_DIR, "scanner.html")
    if not os.path.exists(scanner_path):
        raise HTTPException(status_code=404, detail=f"Scanner page scanner.html not found at {scanner_path}")
    return FileResponse(scanner_path)

# Serve uploaded images for preview under /scanner/previews
app.mount("/scanner/previews", StaticFiles(directory=UPLOAD_DIR), name="previews")

from predict import SkinClassifier
from PIL import Image

# Initialize the classifier globally (loads model once on startup)
# This keeps the model in RAM, making inferences significantly faster (especially on RPi)
classifier = SkinClassifier()

async def cleanup_temp_file(path: str, delay: int = 60):
    """Deletes a file after a specified delay in seconds."""
    await asyncio.sleep(delay)
    try:
        if os.path.exists(path):
            os.remove(path)
            print(f"Cleaned up temp file: {path}")
    except Exception as e:
        print(f"Error cleaning up {path}: {e}")

@app.post("/scanner/process-image")
async def process_image(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    # Generate a unique filename
    file_id = str(uuid.uuid4())
    extension = os.path.splitext(file.filename)[1]
    temp_file_path = os.path.join(UPLOAD_DIR, f"{file_id}{extension}")

    try:
        # Save the uploaded file
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 1. Extract Image Metadata directly (fast)
        with Image.open(temp_file_path) as img:
            base_info = {
                "width": img.width,
                "height": img.height,
                "format": img.format,
                "mode": img.mode,
            }

        # 2. Use the in-memory classifier (lightning fast after initial load)
        prediction = classifier.predict_image(temp_file_path)

        if "error" in prediction:
            raise HTTPException(status_code=500, detail=prediction["error"])

        # Schedule cleanup in the background
        background_tasks.add_task(cleanup_temp_file, temp_file_path)

        # Assemble final response
        return {
            "status": "success",
            **base_info,
            "prediction": prediction,
            "preview_url": f"/scanner/previews/{file_id}{extension}",
            "message": f"Analyse abgeschlossen für: {os.path.basename(temp_file_path)}"
        }

    except Exception as e:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=80)
