# main.py
import json
import os
import tempfile

import redis

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")

try:
    import torch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
except Exception:
    pass

import re
import time
import uuid
from typing import List, Optional, Dict, Any, Tuple

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder

import numpy as np
import cv2
from io import BytesIO
from PIL import Image
from difflib import SequenceMatcher
from dotenv import load_dotenv
import hashlib

load_dotenv()

MATCH_THRESHOLD = float(os.getenv("MATCH_THRESHOLD", "0.85"))
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6390")
redis_client = redis.from_url(REDIS_URL, decode_responses=False)

from product_store import (
    upsert_product_struct, get_product, list_products, search_candidates_by_tokens, ensure_tables
)
from qwen import analyze_with_qwen_vl_modelstudio

app = FastAPI(
    title="Product Recognition API",
    description="Canonical JSON + token pruning + 0.85 similarity. Enhanced with DeepSeek Vision OCR",
    version="7.0.0",
    root_path="/despacho"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ensure_tables()

# _______________________ Helper      _______________________
def compute_image_hash(image_bytes: bytes) -> str:
    return hashlib.sha256(image_bytes).hexdigest()

# ─────────────────────── image utils ───────────────────────
def read_image(file_bytes) -> Optional[np.ndarray]:
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is not None:
        return img
    try:
        image = Image.open(BytesIO(file_bytes)).convert("RGB")
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    except:
        return None

# ✅ NEW: Save bytes to temp file and return path
def save_to_temp_file(file_bytes: bytes, suffix: str = ".jpg") -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        return tmp.name

# ─────────────────────── canonicalization ───────────────────────
def _s(x: Optional[str]) -> str:
    return (x or "").strip().lower()

def _norm_text(x: Optional[str]) -> str:
    t = _s(x)
    t = re.sub(r"\s+", " ", t)
    # Normalize common variants (keep minimal)
    t = t.replace("ad₃e", "ad3e")
    t = t.replace("í","i").replace("ó","o").replace("á","a").replace("é","e").replace("ú","u")
    return t

def build_canonical(raw: Dict[str, Any], sku: Optional[str] = None) -> Dict[str, Any]:
    # ONLY use the 5 guaranteed fields
    product_name = raw.get("product_name")
    active_ingredient = raw.get("active_ingredient")
    concentration = raw.get("concentration")
    manufacturer = raw.get("manufacturer")
    all_visible_text = raw.get("all_visible_text")

    canon = {
        "sku": sku or raw.get("sku"),
        "product_name": _norm_text(product_name) or None,
        "active_ingredient": _norm_text(active_ingredient) or None,
        "concentration": _norm_text(concentration) or None,  # Keep as string (e.g., "3,15%")
        "manufacturer": _norm_text(manufacturer) or None,
        "all_visible_text": _norm_text(all_visible_text) or None,
    }
    return canon

# ─────────────────────── tokens for pruning ───────────────────────
def tokens_from_canon(c: Dict[str, Any]) -> List[str]:
    toks = []
    for field in ["product_name", "active_ingredient", "concentration", "manufacturer"]:
        v = c.get(field)
        if v:
            toks.extend([t for t in re.split(r"[^a-z0-9]+", v) if t])
    
    # Also add tokens from all_visible_text (optional but helpful)
    avt = c.get("all_visible_text")
    if avt:
        toks.extend([t for t in re.split(r"[^a-z0-9]+", avt) if len(t) > 2])
    
    # Dedupe
    seen = set()
    out = []
    for t in toks:
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out
# ─────────────────────── similarity ───────────────────────
def _fuzzy(a: Optional[str], b: Optional[str]) -> float:
    return SequenceMatcher(None, _norm_text(a), _norm_text(b)).ratio()


def score_objects(q: Dict[str, Any], ref: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
    w = {
        "product_name": 0.4,
        "active_ingredient": 0.3,
        "concentration": 0.2,
        "manufacturer": 0.1
    }
    
    parts = {
        "product_name": _fuzzy(q.get("product_name"), ref.get("product_name")),
        "active_ingredient": _fuzzy(q.get("active_ingredient"), ref.get("active_ingredient")),
        "concentration": _fuzzy(q.get("concentration"), ref.get("concentration")),
        "manufacturer": _fuzzy(q.get("manufacturer"), ref.get("manufacturer")),
    }
    
    score = sum(w[k] * parts[k] for k in w.keys())
    return score, parts

# ─────────────────────── endpoints ───────────────────────
@app.get("/")
async def root():
    return {"message": "Product Recognition API is running - Enhanced with DeepSeek Vision OCR"}

@app.post("/selftest")
async def selftest(file: UploadFile = File(...)):
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file")
    
    img_hash = compute_image_hash(contents)
    cache_key = f"ocr:{img_hash}"

    # Try cache first
    cached = redis_client.get(cache_key)
    if cached:
        raw = json.loads(cached)
        print(f"✅ OCR cache hit for {img_hash[:8]}")
    else:
        temp_path = save_to_temp_file(contents)
        try:
            raw = analyze_with_qwen_vl_modelstudio(temp_path) or {}
            # Cache for 1 hour (3600 sec)
            redis_client.setex(cache_key, 3600, json.dumps(raw, ensure_ascii=False))
            print(f"🆕 OCR cache miss for {img_hash[:8]}")
        finally:
            os.unlink(temp_path)

    return JSONResponse(content=raw)

# store a product: sku + 0–3 images (first is parsed), mandatory schema enforced
@app.post("/products/add")
async def add_product(
    sku: str = Form(...),
    files: List[UploadFile] = File(None)
):
    sku = sku.strip()
    if not sku:
        raise HTTPException(status_code=400, detail="SKU is required")

    raw = {}
    if files:
        contents = await files[0].read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty image file")
        temp_path = save_to_temp_file(contents)
        try:
            raw = analyze_with_qwen_vl_modelstudio(temp_path) or {}
        finally:
            os.unlink(temp_path)
    
    if not isinstance(raw, dict):
        raw = {}
    
    # ✅ Canonicalize + tokenize BEFORE upsert
    canon = build_canonical(raw, sku=sku)
    tokens = tokens_from_canon(canon)
    
    # ✅ Now call with 3 args
    row = upsert_product_struct(sku, canon, tokens)

    return JSONResponse(content=jsonable_encoder({"success": True, "stored": row}), status_code=200)

# read & list
@app.get("/products/read")
async def read_product(sku: str):
    row = get_product(sku)
    if not row:
        raise HTTPException(status_code=404, detail="Not found")
    return JSONResponse(content=jsonable_encoder(row), status_code=200)

@app.get("/products/list")
async def list_all(limit: int = 100, offset: int = 0):
    rows = list_products(limit=limit, offset=offset)
    return JSONResponse(content=jsonable_encoder({"items": rows, "count": len(rows)}), status_code=200)

# scan: parse image → canonicalize → prune via tokens/numbers → score → accept if ≥ threshold
@app.post("/scan")
async def scan(
    file: UploadFile = File(...),
    threshold: float = Query(MATCH_THRESHOLD, ge=0.0, le=0.999),
    max_candidates: int = Query(200, ge=1, le=2000)
):
    t0 = time.perf_counter()
    req_id = uuid.uuid4().hex

    try:
        contents = await file.read()
        if not contents:
            return JSONResponse(content=[], status_code=200, headers={
                "X-Scan-Recognized": "0",
                "X-Scan-Reason": "empty_file",
                "X-Scan-Threshold": str(threshold),
            })

        # 🔑 Compute image hash for caching
        img_hash = compute_image_hash(contents)
        cache_key = f"ocr:{img_hash}"

        # 🔄 Try Redis cache first
        cached_raw = redis_client.get(cache_key)
        if cached_raw:
            raw = json.loads(cached_raw)
            ocr_source = "cache"
        else:
            # 🖼️ Save to temp file and call Qwen
            temp_path = save_to_temp_file(contents)
            try:
                raw = analyze_with_qwen_vl_modelstudio(temp_path) or {}
                # 💾 Cache for 1 hour (3600 seconds)
                redis_client.setex(cache_key, 3600, json.dumps(raw, ensure_ascii=False))
                ocr_source = "qwen"
            finally:
                os.unlink(temp_path)

        if not isinstance(raw, dict):
            raw = {}

        # 🧼 Build canonical using ONLY the 5 fields
        q = build_canonical(raw)

        # ✅ Check for signal
        has_signal = any([
            q.get("product_name"),
            q.get("active_ingredient"),
            q.get("concentration"),
            q.get("manufacturer")
        ])
        if not has_signal:
            return JSONResponse(content=[], status_code=200, headers={
                "X-Scan-Recognized": "0",
                "X-Scan-Reason": "no_label_signal",
                "X-Scan-Threshold": str(threshold),
                "X-OCR-Source": ocr_source,
            })

        # 🔍 Tokenize & search
        query_tokens = set(tokens_from_canon(q))

        cands = search_candidates_by_tokens(
            tokens=[],
            pct=None,
            vol_ml=None,
            tablet_count=None,
            limit=max_candidates * 2
        ) or []

        # 🧹 In-memory token filtering
        filtered_cands = []
        for cand in cands:
            canon = cand.get("canon", {})
            if not canon:
                continue
            cand_tokens = set(tokens_from_canon(canon))
            if not query_tokens or not cand_tokens or (query_tokens & cand_tokens):
                filtered_cands.append(cand)
            if len(filtered_cands) >= max_candidates:
                break

        if not filtered_cands:
            return JSONResponse(content=[], status_code=200, headers={
                "X-Scan-Recognized": "0",
                "X-Scan-Reason": "no_candidates_after_pruning",
                "X-Scan-Threshold": str(threshold),
                "X-OCR-Source": ocr_source,
            })

        # 🎯 Score candidates
        best = {"sku": None, "score": 0.0, "parts": {}, "ref": None}
        for r in filtered_cands:
            ref = r.get("canon") or {}
            score, parts = score_objects(q, ref)
            if score > best["score"] or (abs(score - best["score"]) < 1e-6 and (r["sku"] or "") < (best["sku"] or "")):
                best = {"sku": r["sku"], "score": float(score), "parts": parts, "ref": ref}

        if not (best["sku"] and best["score"] >= float(threshold)):
            return JSONResponse(content=[], status_code=200, headers={
                "X-Scan-Recognized": "0",
                "X-Scan-Reason": "below_threshold",
                "X-Scan-Threshold": str(threshold),
                "X-OCR-Source": ocr_source,
            })

        return JSONResponse(
            content=jsonable_encoder({
                "recognized": True,
                "sku": best["sku"],
                "confidence": round(best["score"], 3),
                "label": {"raw": raw, "canon": q},
                "details": {
                    "threshold": float(threshold),
                    "per_field": {k: round(v, 3) for k, v in (best["parts"] or {}).items()},
                    "candidates_considered": len(filtered_cands),
                    "request_id": req_id,
                    "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                    "ocr_source": ocr_source,
                }
            }),
            status_code=200,
            headers={
                "X-Scan-Recognized": "1",
                "X-Scan-Reason": "ok",
                "X-Scan-Threshold": str(threshold),
                "X-OCR-Source": ocr_source,  # 'cache' or 'qwen'
            },
        )

    except Exception as e:
        print(f"Scan error: {str(e)}")
        return JSONResponse(
            content=[],
            status_code=200,
            headers={
                "X-Scan-Recognized": "0",
                "X-Scan-Reason": "internal_error",
                "X-Scan-Threshold": str(threshold),
                "X-OCR-Source": "error",
            },
        )
    
# uvicorn main:app --reload --port 8080
# redis-cli -p 6390 FLUSHDB