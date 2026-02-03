from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
import os
import uuid
import shutil
import asyncio
from model import SkinClassifier
from PIL import Image    
import uvicorn

app = FastAPI()

# Allow CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], #actual domain name
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(os.path.dirname(BASE_DIR), "frontend")
COMPONENTS_DIR = os.path.join(FRONTEND_DIR, "components")
UPLOAD_DIR = os.path.join(BASE_DIR, "temp_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize the classifier globally (loads model once on startup)
classifier = SkinClassifier()

#es gibt wohl keine möglichkeit mehr header und footer mit include einzubinden, aber ich will auch einen header der nicht in jedem html manuell bearbeitet werden muss.
def get_page_with_components(page_name):
    page_path = os.path.join(FRONTEND_DIR, page_name)
    header_path = os.path.join(COMPONENTS_DIR, "header.html")
    footer_path = os.path.join(COMPONENTS_DIR, "footer.html")
    
    with open(page_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    if os.path.exists(header_path):
        with open(header_path, "r", encoding="utf-8") as f:
            header = f.read()
        content = content.replace("<!-- HEADER_PLACEHOLDER -->", header)
        
        # probleme mit react sonst..
        safe_header = header.replace("`", "\\`")
        content = content.replace("{/* HEADER_MARKER */}", f'<div dangerouslySetInnerHTML={{{{ __html: `{safe_header}` }}}} />')

    if os.path.exists(footer_path):
        with open(footer_path, "r", encoding="utf-8") as f:
            footer = f.read()
        content = content.replace("<!-- FOOTER_PLACEHOLDER -->", footer)
        safe_footer = footer.replace("`", "\\`")
        content = content.replace("{/* FOOTER_MARKER */}", f'<div dangerouslySetInnerHTML={{{{ __html: `{safe_footer}` }}}} />')
        
    return content

# Serve the landing page at /
@app.get("/")
async def read_landing():
    try:
        content = get_page_with_components("index.html")
        return HTMLResponse(content)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Page not found")

@app.get("/scanner")
async def read_scanner():
    try:
        content = get_page_with_components("scanner.html")
        return HTMLResponse(content)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Page not found")

@app.get("/cv")
async def read_cv():
    try:
        content = get_page_with_components("cv.html")
        return HTMLResponse(content)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Page not found")

@app.get("/doc")
async def read_doc():
    try:
        content = get_page_with_components("cv.html")
        return HTMLResponse(content)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Page not found")

# bilder zur Vorschau bereitstellen
app.mount("/scanner/previews", StaticFiles(directory=UPLOAD_DIR), name="previews")

# Statische Ressourcen (PDFs, Bilder, etc.) bereitstellen
app.mount("/resource", StaticFiles(directory=os.path.join(FRONTEND_DIR, "resource")), name="static_resource")


@app.post("/scanner/process-image")
async def process_image(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    # Generate a unique filename for security reasons and attach file extension
    file_id = str(uuid.uuid4())
    extension = os.path.splitext(file.filename)[1]
    temp_file_path = os.path.join(UPLOAD_DIR, f"{file_id}{extension}")

    try:
        # Save the uploaded file
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        #debugging maybe
        with Image.open(temp_file_path) as img:
            base_info = {
                "width": img.width,
                "height": img.height,
                "format": img.format,
                "mode": img.mode,
            }

        # Use the in-memory classifier
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


async def cleanup_temp_file(path: str, delay: int = 60):
    #Deletes a file after a specified delay in seconds.
    await asyncio.sleep(delay)
    try:
        if os.path.exists(path):
            os.remove(path)
            print(f"Cleaned up temp file: {path}")
    except Exception as e:
        print(f"Error cleaning up {path}: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=80)
