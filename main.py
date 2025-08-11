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
    allow_origins=["http://localhost:5173", "http://localhost:8000", "https://develop.globaldv.tech"],
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
    try:
        # Try using OpenCV first
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is not None:
            return img
        
        # Fallback to PIL if OpenCV fails
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    except Exception as e:
        raise ValueError(f"Failed to read image: {str(e)}")

@app.post("/scan")
async def scan_product(file: UploadFile = File(...)):
    try:
        # Read the uploaded file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Recognize the product
        result = scanner.recognize(img)
        if not result:
            return JSONResponse(
                content={"recognized": False, "message": "Product not recognized"},
                status_code=200
            )

        return JSONResponse(
            content={
                "recognized": True,
                "product": result["label"],
                "confidence": result["confidence"]
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/add")
async def add_product(
    product_name: str = Form(...),
    product_sku: str = Form(...),
    files: List[UploadFile] = File(...)
):
    # Input validation
    if not product_name.strip():
        raise HTTPException(status_code=400, detail="Product name is required")
    if not product_sku.strip():
        raise HTTPException(status_code=400, detail="Product SKU is required")
    if len(files) < 5:
        raise HTTPException(status_code=400, detail="At least 5 photos are required")

    successes = 0
    errors = []

    for idx, file in enumerate(files):
        try:
            # Read and decode image
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                errors.append(f"Image {idx + 1}: Invalid image format")
                continue

            # 🔐 Apply blur before saving
            try:
                filtered = mask_humans(img)
            except Exception as e:
                print(f"Warning: Face/hand detection failed for image {idx + 1}: {str(e)}")
                filtered = img  # Use original image if face/hand detection fails
            
            try:
                success = scanner.add_product(product_name, product_sku, filtered)
                if success:
                    successes += 1
                    print(f"✅ Successfully added image {idx + 1} for product {product_name} (SKU: {product_sku})")
            except Exception as e:
                error_msg = f"Failed to add image {idx + 1}: {str(e)}"
                print(f"❌ {error_msg}")
                errors.append(error_msg)

        except Exception as e:
            error_msg = f"Image {idx + 1}: {str(e)}"
            print(f"❌ {error_msg}")
            errors.append(error_msg)

    # Return appropriate response based on results
    if successes == len(files):
        return JSONResponse(
            content={
                "success": True,
                "message": f"✅ Product '{product_name}' (SKU: {product_sku}) stored with {successes} views"
            },
            status_code=200
        )
    elif successes > 0:
        return JSONResponse(
            content={
                "success": True,
                "message": f"⚠️ Partially successful: {successes} out of {len(files)} images were saved for '{product_name}' (SKU: {product_sku})",
                "errors": errors
            },
            status_code=207  # Multi-Status
        )
    else:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "❌ Failed to store any product images",
                "errors": errors
            }
        )


# Run the FastAPI app with: 
# ✅ 3. Example cURL (Frontend can mimic this)
# curl -X POST "http://localhost:8000/scan" -F "file=@your_image.jpg"
# ✅ Example cURL to Add a Product Image
# curl -X POST "http://localhost:8000/add" \
#  -F "product_name=Logitech Mouse M325" \
#  -F "file=@logitech_m325_view1.jpg"
# To run it locally:
# uvicorn main:app --reload
