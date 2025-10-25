import os

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

load_dotenv()

MATCH_THRESHOLD = float(os.getenv("MATCH_THRESHOLD", "0.85"))

from product_store import (
    upsert_product_struct, get_product, list_products, search_candidates_by_tokens, ensure_tables
)
from product_scanner import extract_product_info  # Updated DeepSeek Vision OCR

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

# ─────────────────────── canonicalization ───────────────────────
def _s(x: Optional[str]) -> str:
    return (x or "").strip().lower()

def _norm_text(x: Optional[str]) -> str:
    t = _s(x)
    t = re.sub(r"\s+", " ", t)
    t = t.replace("ad₃e", "ad3e")
    t = t.replace("í","i").replace("ó","o").replace("á","a").replace("é","e").replace("ú","u")
    return t

def _parse_pct(x: Optional[str]) -> Optional[float]:
    if not x: return None
    m = re.search(r"(\d+(?:[.,]\d+)?)\s*%", x)
    if not m: return None
    return float(m.group(1).replace(",", "."))

def _parse_ml(x: Optional[str]) -> Optional[float]:
    if not x: return None
    xl = x.lower()
    m = re.search(r"(\d+(?:[.,]\d+)?)\s*(?:ml|mℓ)\b", xl)
    if m: return float(m.group(1).replace(",", "."))
    m = re.search(r"(\d+(?:[.,]\d+)?)\s*l\b", xl)
    if m: return float(m.group(1).replace(",", ".")) * 1000.0
    m = re.search(r"contenido\s+neto[^0-9]*?(\d+(?:[.,]\d+)?)", xl)
    if m:
        v = float(m.group(1).replace(",", "."))
        if 10 <= v <= 5000: return v
    return None

def _parse_tablets(x: Optional[str]) -> Optional[int]:
    """Parse tablet/unit count from usage_instructions or other text fields"""
    if not x: return None
    xl = x.lower()
    
    # Look for tablet counts
    tablet_patterns = [
        r"(\d+)\s*(?:comp|tabletas?|tabs?)\b",
        r"(\d+)\s*(?:unidades?|units?)\b",
        r"caja\s+de\s+(\d+)\s*(?:comp|tabletas?)",
        r"(\d+)\s*\.?\s*comp",
    ]
    
    for pattern in tablet_patterns:
        m = re.search(pattern, xl)
        if m:
            try:
                return int(m.group(1))
            except:
                continue
    return None

def build_canonical(raw: Dict[str, Any], sku: Optional[str] = None) -> Dict[str, Any]:
    # map many possible keys → mandatory schema
    name = raw.get("product_name") or raw.get("product_title")
    brand = raw.get("manufacturer") or raw.get("brand")
    active = raw.get("active_ingredient")
    formulation = raw.get("formulation") or raw.get("dosage_form")
    keywords = raw.get("keywords") or raw.get("other_details")
    
    # Enhanced extraction from new fields
    usage_instructions = raw.get("usage_instructions")
    presentation_details = raw.get("presentation_details")
    warnings = raw.get("warnings")

    pct = raw.get("percent")
    if pct is None:
        pct = _parse_pct(str(raw.get("concentration") or name or ""))

    vol = raw.get("volume_ml")
    if vol is None:
        vol = _parse_ml(str(raw.get("volume") or name or ""))
    
    # Extract tablet count from usage instructions or presentation details
    tablets = None
    if usage_instructions:
        tablets = _parse_tablets(usage_instructions)
    if tablets is None and presentation_details:
        tablets = _parse_tablets(presentation_details)

    # normalize
    name_n = _norm_text(name) or None
    brand_n = _norm_text(brand) or None
    active_n = _norm_text(active) or None
    form_n = _norm_text(formulation) or None

    if isinstance(keywords, list):
        kw = [ _norm_text(k) for k in keywords if k ]
    elif isinstance(keywords, str):
        kw = [ _norm_text(k) for k in re.split(r"[;,/|]", keywords) if k.strip() ]
    else:
        kw = []

    # Build enhanced extras with new fields
    extras = {
        "route": _norm_text(raw.get("administration_route")) or None,
        "species": _norm_text(raw.get("species")) or None,
        "color_badge": _norm_text(raw.get("color") or raw.get("badge_color")) or None,
        "language": _norm_text(raw.get("language")) or None,
        "presentation_details": _norm_text(presentation_details) or None,
        "usage_instructions": _norm_text(usage_instructions) or None,
        "warnings": _norm_text(warnings) or None
    }

    canon = {
        "sku": sku or raw.get("sku"),
        "name": name_n,
        "brand": brand_n,
        "active": active_n,
        "concentration_pct": float(pct) if isinstance(pct, (int, float, str)) and str(pct) else None,
        "formulation": form_n,
        "volume_ml": float(vol) if isinstance(vol, (int, float, str)) and str(vol) else None,
        "tablet_count": tablets,
        "keywords": (kw[:8] or None),
        "extras": extras
    }
    return canon

# ─────────────────────── tokens for pruning ───────────────────────
def tokens_from_canon(c: Dict[str, Any]) -> List[str]:
    toks = []
    for k in ("name", "brand", "active", "formulation"):
        v = c.get(k)
        if v:
            toks.extend([t for t in re.split(r"[^a-z0-9]+", v) if t])
    if c.get("keywords"):
        toks.extend([t for t in c["keywords"] if t])
    
    # Extract tokens from enhanced extras
    extras = c.get("extras", {})
    for field in ["presentation_details", "usage_instructions", "warnings"]:
        value = extras.get(field)
        if value:
            toks.extend([t for t in re.split(r"[^a-z0-9]+", value) if len(t) > 2])  # Only tokens > 2 chars
    
    # coarse numeric bins to help filter by value
    if c.get("concentration_pct") is not None:
        pct_bin = round(float(c["concentration_pct"])*20)/20.0  # ~0.05 bins
        toks.append(f"pct:{pct_bin:.2f}")
    if c.get("volume_ml") is not None:
        ml = float(c["volume_ml"])
        ml_bin = int(round(ml/25.0)*25)  # 25 mL buckets
        toks.append(f"ml:{ml_bin}")
    if c.get("tablet_count") is not None:
        tablets = int(c["tablet_count"])
        tablet_bin = int(round(tablets/10.0)*10)  # 10 tablet buckets
        toks.append(f"tabs:{tablet_bin}")
    
    # dedupe
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

def _num_sim(x: Optional[float], y: Optional[float], tol_abs: float, tol_rel: float = 0.08) -> float:
    if x is None or y is None: return 0.0
    if x == y: return 1.0
    diff = abs(x - y)
    if diff <= tol_abs: return max(0.0, 1.0 - diff / max(tol_abs, 1e-9))
    if min(abs(x), abs(y)) > 0 and (diff / max(abs(x), abs(y))) <= tol_rel:
        return max(0.0, 1.0 - (diff / max(abs(x), abs(y))))
    return 0.0

def _set_jaccard(a: Optional[List[str]], b: Optional[List[str]]) -> float:
    A = set([_norm_text(x) for x in (a or []) if x])
    B = set([_norm_text(x) for x in (b or []) if x])
    if not A and not B: return 0.0
    return len(A & B) / max(1, len(A | B))

def score_objects(q: Dict[str, Any], ref: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
    # Updated weights to include new fields
    w = {
        "name": 0.30, "brand": 0.15, "active": 0.15, 
        "concentration_pct": 0.12, "volume_ml": 0.10, "tablet_count": 0.08,
        "formulation": 0.05, "keywords": 0.05
    }
    
    parts = {
        "name": _fuzzy(q.get("name"), ref.get("name")),
        "brand": _fuzzy(q.get("brand"), ref.get("brand")),
        "active": _fuzzy(q.get("active"), ref.get("active")),
        "formulation": _fuzzy(q.get("formulation"), ref.get("formulation")),
        "concentration_pct": _num_sim(q.get("concentration_pct"), ref.get("concentration_pct"), tol_abs=0.08),
        "volume_ml": _num_sim(q.get("volume_ml"), ref.get("volume_ml"), tol_abs=30.0),
        "tablet_count": _num_sim(q.get("tablet_count"), ref.get("tablet_count"), tol_abs=5.0),
        "keywords": _set_jaccard(q.get("keywords"), ref.get("keywords")),
    }
    score = sum(w[k]*parts[k] for k in w.keys())
    return score, parts

# ─────────────────────── endpoints ───────────────────────
@app.get("/")
async def root():
    return {"message": "Product Recognition API is running - Enhanced with DeepSeek Vision OCR"}

@app.post("/selftest")
async def selftest(file: UploadFile = File(...), preview_pruning: bool = True):
    contents = await file.read()
    img = read_image(contents)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    raw = extract_product_info(img) or {}
    canon = build_canonical(raw)
    tokens = tokens_from_canon(canon)

    pruned = []
    if preview_pruning:
        pruned = search_candidates_by_tokens(
            tokens=tokens,
            pct=canon.get("concentration_pct"),
            vol_ml=canon.get("volume_ml"),
            tablet_count=canon.get("tablet_count"),  # New parameter
            limit=50
        )

    return JSONResponse(content=jsonable_encoder({
        "raw": raw,
        "canon": canon,
        "tokens": tokens,
        "pruned_candidates": [r["sku"] for r in pruned],
        "counts": {
            "tokens": len(tokens),
            "candidates": len(pruned)
        },
        "threshold": {"global_default": MATCH_THRESHOLD}
    }))

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
        img = read_image(contents)
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        raw = extract_product_info(img) or {}
    if not isinstance(raw, dict):
        raw = {}

    canon = build_canonical(raw, sku=sku)
    canon["sku"] = sku  # enforce presence
    tokens = tokens_from_canon(canon)

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
# Not recognized paths return body [] and signal via headers
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
        img = read_image(contents)
        if img is None:
            return JSONResponse(
                content=[],
                status_code=200,
                headers={
                    "X-Scan-Recognized": "0",
                    "X-Scan-Reason": "invalid_image",
                    "X-Scan-Threshold": str(threshold),
                },
            )

        # 1) Extract + canonicalize
        raw = extract_product_info(img)
        if not isinstance(raw, dict):
            raw = {}
        q = build_canonical(raw)

        has_signal = any([
            q.get("name"), q.get("brand"), q.get("active"),
            q.get("concentration_pct") is not None,
            q.get("volume_ml") is not None,
            q.get("tablet_count") is not None,  # New signal check
            bool(q.get("keywords"))
        ])
        if not has_signal:
            return JSONResponse(
                content=[],
                status_code=200,
                headers={
                    "X-Scan-Recognized": "0",
                    "X-Scan-Reason": "no_label_signal",
                    "X-Scan-Threshold": str(threshold),
                },
            )

        # 2) Prune candidates
        tokens = tokens_from_canon(q)
        cands = search_candidates_by_tokens(
            tokens,
            q.get("concentration_pct"),
            q.get("volume_ml"),
            q.get("tablet_count"),  # New parameter
            limit=max_candidates
        ) or []

        if not cands:
            return JSONResponse(
                content=[],
                status_code=200,
                headers={
                    "X-Scan-Recognized": "0",
                    "X-Scan-Reason": "no_candidates_after_pruning",
                    "X-Scan-Threshold": str(threshold),
                },
            )

        # 3) Score candidates
        best = {"sku": None, "score": 0.0, "parts": {}, "ref": None}
        for r in cands:
            ref = r.get("canon") or {}
            score, parts = score_objects(q, ref)
            if score > best["score"] or (abs(score-best["score"]) < 1e-6 and (r["sku"] or "") < (best["sku"] or "")):
                best = {"sku": r["sku"], "score": float(score), "parts": parts, "ref": ref}

        # 4) Decision; if below threshold → empty array + headers
        if not (best["sku"] and best["score"] >= float(threshold)):
            return JSONResponse(
                content=[],
                status_code=200,
                headers={
                    "X-Scan-Recognized": "0",
                    "X-Scan-Reason": "below_threshold",
                    "X-Scan-Threshold": str(threshold),
                },
            )

        # success → full payload + headers
        return JSONResponse(
            content=jsonable_encoder({
                "recognized": True,
                "sku": best["sku"],
                "confidence": round(best["score"], 3),
                "label": {"raw": raw, "canon": q},
                "details": {
                    "threshold": float(threshold),
                    "per_field": {k: round(v, 3) for k, v in (best["parts"] or {}).items()},
                    "candidates_considered": len(cands),
                    "request_id": req_id,
                    "latency_ms": round((time.perf_counter()-t0)*1000, 1)
                }
            }),
            status_code=200,
            headers={
                "X-Scan-Recognized": "1",
                "X-Scan-Reason": "ok",
                "X-Scan-Threshold": str(threshold),
            },
        )

    except Exception as e:
        # Log the error for debugging
        print(f"Scan error: {str(e)}")
        return JSONResponse(
            content=[],
            status_code=200,
            headers={
                "X-Scan-Recognized": "0",
                "X-Scan-Reason": "internal_error",
                "X-Scan-Threshold": str(threshold),
            },
        )


# uvicorn main:app --reload --port 8080