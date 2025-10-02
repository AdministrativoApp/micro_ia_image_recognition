from fastapi import FastAPI, File, UploadFile, Form, HTTPException, APIRouter
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import io
from PIL import Image
from product_scanner import ProductScannerSQL, db_url  # ← removed mask_humans
from typing import List

app = FastAPI(
    title="Product Recognition API",
    description="Scan and add products using webcam-like logic (no human masking).",
    version="1.0.0"
)

# Create router with prefix
router = APIRouter(prefix="/despacho")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:8000", "https://develop.globaldv.tech"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "message": "Product Recognition API is running",
        "endpoints": {
            "docs": "/docs",
            "redoc": "/redoc",
            "scan": "/despacho/scan",
            "add": "/despacho/add"
        }
    }

scanner = ProductScannerSQL(db_url=db_url)
scanner.train()  # optional preload

def read_imagefile(file_bytes) -> np.ndarray:
    try:
        # Try OpenCV first
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is not None:
            return img
        # Fallback PIL
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    except Exception as e:
        raise ValueError(f"Failed to read image: {str(e)}")

@router.post("/scan")
async def scan_product(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

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

@router.post("/add")
async def add_product(
    product_name: str = Form(...),
    product_sku: str = Form(...),
    files: List[UploadFile] = File(...)
):
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
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                errors.append(f"Image {idx + 1}: Invalid image format")
                continue

            # No masking — store original image
            try:
                success = scanner.add_product(product_name, product_sku, img)
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
            status_code=207
        )
    else:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "❌ Failed to store any product images",
                "errors": errors
            }
        )

# Include router
app.include_router(router)

# Run with:
# uvicorn main:app --reload
# Endpoints: /despacho/scan and /despacho/add
