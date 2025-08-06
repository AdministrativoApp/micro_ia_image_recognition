from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import io
from PIL import Image
from product_scanner import ProductScannerSQL, db_url, mask_humans  # also import mask_humans
from typing import List

app = FastAPI(
    title="Product Recognition API",
    description="Scan and add products using webcam-like logic with face/hands blurring.",
    version="1.0.0",
    root_path="/despacho" 
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "https://develop.globaldv.tech"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add root endpoint
@app.get("/")
async def root():
    return {
        "message": "Product Recognition API is running",
        "endpoints": {
            "docs": "/docs",
            "redoc": "/redoc",
            "scan": "/scan",
            "add": "/add"
        }
    }

scanner = ProductScannerSQL(db_url=db_url)
scanner.train()  # optional preload (lazy retrain inside .recognize anyway)

def read_imagefile(file_bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

@app.post("/scan")
async def scan_product(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img_array = read_imagefile(contents)

        # 🔐 Apply face/hands blur before recognition
        filtered = mask_humans(img_array)
        result = scanner.recognize(filtered)

        if result:
            name, confidence = result
            return {"recognized": True, "product": name, "confidence": round(confidence, 2)}
        else:
            return {"recognized": False, "message": "Product not recognized"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/add")
async def add_product(
    product_name: str = Form(...),
    files: List[UploadFile] = File(...)
):
    if len(files) < 5:
        raise HTTPException(status_code=400, detail="At least 5 photos are required")

    successes = 0
    errors = []

    for idx, file in enumerate(files):
        try:
            contents = await file.read()
            img_array = read_imagefile(contents)

            # 🔐 Apply blur before saving
            filtered = mask_humans(img_array)
            success = scanner.add_product(product_name, filtered)
            if success:
                successes += 1
        except Exception as e:
            errors.append(f"Image {idx + 1}: {str(e)}")

    if successes == len(files):
        return {
            "success": True,
            "message": f"✅ Product '{product_name}' stored with {successes} views"
        }
    elif successes > 0:
        return {
            "success": False,
            "message": f"⚠️ Only {successes} out of {len(files)} images were saved",
            "errors": errors
        }
    else:
        raise HTTPException(status_code=500, detail="❌ Failed to store any product images")


# Run the FastAPI app with: 
# ✅ 3. Example cURL (Frontend can mimic this)
# curl -X POST "http://localhost:8000/scan" -F "file=@your_image.jpg"
# ✅ Example cURL to Add a Product Image
# curl -X POST "http://localhost:8000/add" \
#  -F "product_name=Logitech Mouse M325" \
#  -F "file=@logitech_m325_view1.jpg"
