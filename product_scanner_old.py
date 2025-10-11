#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Product Image Recognition – Accuracy + Open-Set Guards

What’s new:
- OCR strengthened (targeted Tesseract pass + bottom-ROI OCR).
- Strict volume guard: if reference has volume (e.g., 500 ml), query must show and match volume.
- Higher acceptance floors: OCR and geometry.
- CLI override: allow adding a new product even after a wrong recognition.
- Require >=2 products before enabling recognition.
- Correct SKU return (no "debug").

Dependencies (common):
  pip install opencv-python-headless numpy scikit-learn python-dotenv joblib psycopg2-binary
Optional:
  pip install easyocr
  pip install faiss-cpu
  apt-get install tesseract-ocr  (or OS equivalent), plus: pip install pytesseract
"""

import os
import uuid
import cv2
import joblib
import psycopg2
import numpy as np
import time
from datetime import datetime
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from dotenv import load_dotenv
from collections import defaultdict
import base64, json, requests

# ---------------- Optional FAISS (ANN) ----------------
_HAS_FAISS = False
try:
    import faiss  # pip install faiss-cpu
    _HAS_FAISS = True
except Exception:
    _HAS_FAISS = False

# ---------------- Env / DB ----------------
load_dotenv()
USER = os.getenv('DB_USER')
PASSWORD = os.getenv('DB_PASSWORD')
DATABASE = os.getenv('DB_DATABASE')
HOST = os.getenv('DB_HOST')
PORT = os.getenv('DB_PORT')
db_url = f"postgresql://{USER}:{PASSWORD}@{HOST}:{PORT}/{DATABASE}"

# ---------------- DeepSeek config (optional) ----------------
DEEPSEEK_API_KEY  = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_MODEL    = os.getenv("DEEPSEEK_MODEL", "deepseek-vl")
DEEPSEEK_API_BASE = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com")


# =========================
# Utilities and Preprocess
# =========================
def _b64_png_from_bgr(img_bgr):
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        raise RuntimeError("encode png failed")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def _denoise_enhance(img_bgr):
    h, w = img_bgr.shape[:2]
    scale = 640.0 / max(h, w)
    if scale < 1.0:
        img_bgr = cv2.resize(img_bgr, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    img_bgr = cv2.bilateralFilter(img_bgr, 5, 25, 25)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _multicrops(img):
    h, w = img.shape[:2]
    crops = []
    cx1, cy1 = int(0.15*w), int(0.15*h)
    cx2, cy2 = int(0.85*w), int(0.85*h)
    crops.append(img[cy1:cy2, cx1:cx2])
    crops.append(img[0:int(0.55*h), int(0.20*w):int(0.80*w)])
    crops.append(img[int(0.45*h):h, int(0.20*w):int(0.80*w)])
    crops.append(img[int(0.20*h):int(0.80*h), 0:int(0.55*w)])
    crops.append(img[int(0.20*h):int(0.80*h), int(0.45*w):w])
    return [c for c in crops if c.size]


# =========================
# OCR / Numeric parsing
# =========================
NUM_PERCENT = re.compile(r"(\d+(?:\.\d+)?)\s*%")
NUM_WITH_UNIT = re.compile(r"(\d+(?:\.\d+)?)[\s\-]*(ml|mL|ML|l|L)\b")

def _normalize_text(txt: str) -> str:
    txt = txt.lower()
    txt = re.sub(r"[^a-z0-9%\. l]+", " ", txt)
    return re.sub(r"\s+", " ", txt).strip()

_HAS_EASYOCR = False
_HAS_TESS = False
_EASYOCR_READER = None
try:
    import easyocr
    _HAS_EASYOCR = True
    _EASYOCR_READER = easyocr.Reader(['en','es','pt'], gpu=False)
except Exception:
    try:
        import pytesseract
        _HAS_TESS = True
    except Exception:
        pass

def _bottom_roi(img_bgr, frac=0.30):
    h, w = img_bgr.shape[:2]
    y1 = int((1.0 - frac) * h)
    return img_bgr[y1:h, 0:w]

def _add_bottom_roi_text(img_bgr, base_text):
    # second OCR pass on bottom region where volumes often are printed
    text_parts = [base_text]
    roi = _bottom_roi(img_bgr, frac=0.35)
    roi_rgb = cv2.cvtColor(_denoise_enhance(roi), cv2.COLOR_BGR2RGB)
    if _HAS_EASYOCR and _EASYOCR_READER is not None:
        try:
            res = _EASYOCR_READER.readtext(roi_rgb, detail=0, paragraph=True)
            if res: text_parts.append(" ".join(res))
        except Exception:
            pass
    if _HAS_TESS:
        try:
            import pytesseract
            cfg = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789.%mMlL'
            text_parts.append(pytesseract.image_to_string(roi_rgb, config=cfg))
        except Exception:
            pass
    return _normalize_text(" ".join([t for t in text_parts if t]))

def extract_ocr_text(img_bgr):
    texts = []
    img_rgb = cv2.cvtColor(_denoise_enhance(img_bgr), cv2.COLOR_BGR2RGB)

    # EasyOCR paragraph
    if _HAS_EASYOCR and _EASYOCR_READER is not None:
        try:
            res = _EASYOCR_READER.readtext(img_rgb, detail=0, paragraph=True)
            if res: texts.append(" ".join(res))
        except Exception:
            pass

    # Tesseract general + targeted numeric/units
    if _HAS_TESS:
        try:
            import pytesseract
            texts.append(pytesseract.image_to_string(img_rgb))
            cfg = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789.%mMlL'
            texts.append(pytesseract.image_to_string(img_rgb, config=cfg))
        except Exception:
            pass

    base = _normalize_text(" ".join([t for t in texts if t]))
    return _add_bottom_roi_text(img_bgr, base)

def _parse_volume_ml_from_text(text: str):
    if not text: return None
    m = NUM_WITH_UNIT.search(text)
    if not m: return None
    v = float(m.group(1))
    u = m.group(2).lower()
    if u == 'l':
        v *= 1000.0
    return v

def _parse_percent_from_text(text: str):
    if not text: return None
    m = NUM_PERCENT.search(text)
    if not m: return None
    try:
        return float(m.group(1))
    except Exception:
        return None

def extract_numeric_cues(text: str):
    # normalized numeric vector used in the fine embedding
    raw_pcts = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\s*%", text)]
    percents = [p for p in raw_pcts if 0.0 <= p <= 10.0]
    sized = []
    for m in NUM_WITH_UNIT.finditer(text):
        v = float(m.group(1)); u = m.group(2).lower()
        if u == 'l': v *= 1000.0
        sized.append(v)
    max_pct = max(percents) if percents else -1.0
    max_ml = max(sized) if sized else -1.0
    return np.array([
        max_pct/10.0,          # [0..1] for 0..10%
        len(percents)/5.0,     # normalized count
        max_ml/1000.0,         # [0..1] for 0..1000 ml
        len(sized)/5.0
    ], dtype=np.float32)

def global_hsv_hist(img_bgr, bins=16):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h = hsv[...,0].flatten()
    hist, _ = np.histogram(h, bins=bins, range=(0,180), density=True)
    return hist.astype(np.float32)

def extract_dominant_colors(img_bgr, k=5):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pixels = img_rgb.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = centers.astype(np.uint8)
    unique, counts = np.unique(labels, return_counts=True)
    sorted_indices = np.argsort(counts)[::-1]
    dominant_colors = centers[sorted_indices]
    return dominant_colors.flatten().astype(np.float32)

def color_moments(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    moments = []
    for channel in range(3):
        ch = img_rgb[..., channel].flatten().astype(np.float32)
        mean = np.mean(ch)
        std = np.std(ch)
        skewness = np.mean(((ch - mean) / (std + 1e-8)) ** 3)
        moments.extend([mean, std, skewness])
    return np.array(moments, dtype=np.float32)

# =========================
# DeepSeek optional reader
# =========================
def deepseek_read_label(img_bgr):
    if not DEEPSEEK_API_KEY:
        return {"percent": None, "volume_ml": None, "badge_color": None}
    img_b64 = _b64_png_from_bgr(img_bgr)
    prompt = (
        "You are an expert label reader specializing in animal pharmaceutical and medical product labels. "
        "Your task is to accurately extract key information from medication packaging, such as bottles, boxes, or vials. "
        "Focus on printed text, numbers, and visual indicators like colored badges or dots.\n\n"
        "Extract ONLY the following information from the product label:\n"
        "- percent: The active ingredient concentration as a decimal number (e.g., 1.0 for 1%, 3.15 for 3.15%, or 0.5 for 0.5%). "
        "  Look for formats like '1%', '1.0%', '3.15%', or written as 'concentration 1%'. Return as a number or null if not found.\n"
        "- volume_ml: The product volume in milliliters as a number (e.g., 500 for 500ml, 250 for 250mL, or 1000 for 1L). "
        "  Look for formats like '500ml', '500 mL', '1L', or '1000ml'. Convert liters to ml if needed (e.g., 1L = 1000ml). Return as a number or null if not found.\n"
        "- badge_color: The color of any small badge, dot, or indicator on the label (e.g., 'red', 'blue', 'green', 'yellow'). "
        "  Look for colored elements that distinguish product variants. Return as a lowercase string (e.g., 'red') or null if no clear badge/dot is visible.\n\n"
        "Rules:\n"
        "- Only extract information that is clearly printed or visually indicated on the label.\n"
        "- Ignore background designs, logos, barcodes, batch numbers, expiration dates, or irrelevant text.\n"
        "- If a field is missing, unclear, or cannot be confidently determined, return null for that field.\n"
        "- Handle common abbreviations: '%' for percent, 'ml'/'mL' for milliliters, 'L' for liters.\n"
        "- For colors, focus on small, distinctive badges or dots (not the overall label color).\n"
        "- Return ONLY valid JSON with the exact keys: percent, volume_ml, badge_color. No extra text or explanations.\n\n"
        "Examples:\n"
        '{"percent": 1.0, "volume_ml": 500, "badge_color": "blue"}\n'
        '{"percent": 3.15, "volume_ml": null, "badge_color": "red"}\n'
        '{"percent": 0.5, "volume_ml": 250, "badge_color": null}\n'
        '{"percent": null, "volume_ml": 1000, "badge_color": "green"}\n'
        '{"percent": 2.0, "volume_ml": 500, "badge_color": null}'
    )
    url = f"{DEEPSEEK_API_BASE}/v1/chat/completions"
    headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}
    data = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": [
                {"type":"input_text","text":"Read the label and extract values."},
                {"type":"input_image","image_url": f"data:image/png;base64,{img_b64}"}
            ]}
        ],
        "temperature": 0
    }
    try:
        resp = requests.post(url, headers=headers, data=json.dumps(data), timeout=30)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        text = content if isinstance(content, str) else json.dumps(content)
        start = text.find("{"); end = text.rfind("}")
        if start == -1 or end == -1:
            return {"percent": None, "volume_ml": None, "badge_color": None}
        result = json.loads(text[start:end+1])
        return {
            "percent": float(result["percent"]) if result.get("percent") is not None else None,
            "volume_ml": float(result["volume_ml"]) if result.get("volume_ml") is not None else None,
            "badge_color": result.get("badge_color")
        }
    except Exception:
        return {"percent": None, "volume_ml": None, "badge_color": None}


# =========================
# Main Scanner Class
# =========================
class ProductScannerSQL:
    def _has_min_products(self, min_required: int = 2) -> bool:
        try:
            self.cursor.execute("SELECT COUNT(*) FROM public.products;")
            n = self.cursor.fetchone()[0]
            self.unique_labels = int(n)
            return n >= min_required
        except Exception as e:
            print(f"DB check failed: {e}")
            return False

    def __init__(self, db_url, features_dir='features'):
        self.unique_labels = 0
        self.min_labels_to_accept = 2

        # thresholds / tolerances
        self.sim_threshold_global = 0.78
        self.MIN_MARGIN = 0.10
        self.MIN_GEO = 0.40
        self.MIN_OCR = 0.30
        self.PCT_TOL = 0.06     # %
        self.ML_TOL = 60.0      # ml

        self.conn = psycopg2.connect(db_url)
        self.cursor = self.conn.cursor()
        self.feature_extractor = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
        self.features_dir = features_dir
        os.makedirs(features_dir, exist_ok=True)
        self.ensure_tables()

        # Learned in train()
        self.pca = None
        self.vectorizer = None
        self.use_pca = True
        self.coarse_dim = 128

        # Banks and caches
        self.X = None
        self.y = []
        self.file_paths = []
        self.img_path_by_index = []
        self.fine_bank = None
        self.fine_bank_norm = None
        self.tfidf_bank = None
        self.numeric_bank = None
        self.dom_colors_bank = None
        self.color_moments_bank = None
        self.badge_bank = None
        self.sku_stats = {}
        self.faiss_index = None
        self.knn = None
        self.is_trained = False
        self.labels_by_index = None

    # ---------- DB ----------
    def ensure_tables(self):
        self.cursor.execute("""
        CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
        CREATE TABLE IF NOT EXISTS products (
            id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            name TEXT NOT NULL UNIQUE,
            sku TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS product_vectors (
            id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            product_id UUID REFERENCES products(id) ON DELETE CASCADE,
            file_path TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT NOW()
        );
        """)
        self.conn.commit()

    # ---------- Vision preprocessing ----------
    def preprocess_frame(self, img_array):
        if img_array is None or img_array.size == 0:
            raise ValueError("preprocess_frame: empty image")
        if len(img_array.shape) == 2 or (img_array.shape[2] == 1):
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
        elif img_array.shape[2] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
        resized = cv2.resize(img_array, (224, 224))
        arr = image.img_to_array(resized)
        arr = np.expand_dims(arr, axis=0)
        return preprocess_input(arr)

    def _cnn_raw(self, img_array):
        processed = self.preprocess_frame(img_array)
        return self.feature_extractor.predict(processed, verbose=0)[0].astype(np.float32)

    def _crop_for_ds(self, img_bgr):
        h, w = img_bgr.shape[:2]
        y1, y2 = int(0.40*h), int(0.88*h)
        x1, x2 = int(0.45*w), int(0.97*w)
        crop = img_bgr[y1:y2, x1:x2]
        return crop if crop.size else img_bgr

    def _label_roi(self, img):
        h, w = img.shape[:2]
        x1, y1 = int(0.12*w), int(0.18*h)
        x2, y2 = int(0.88*w), int(0.90*h)
        roi = img[y1:y2, x1:x2]
        return roi if roi.size else img

    def _extract_payload(self, img_array):
        img_array = _denoise_enhance(img_array)
        cnn_global = self._cnn_raw(img_array)
        crops = _multicrops(img_array)
        if crops:
            cnn_locals = [self._cnn_raw(c) for c in crops]
            cnn_local = np.mean(np.stack(cnn_locals, axis=0), axis=0).astype(np.float32)
        else:
            cnn_local = cnn_global.copy()

        ocr_text = extract_ocr_text(img_array)
        numeric = extract_numeric_cues(ocr_text)
        global_hist = global_hsv_hist(img_array)
        dominant_colors = extract_dominant_colors(img_array, k=5)
        color_moments_feat = color_moments(img_array)
        return {"cnn": cnn_global, "cnn_local": cnn_local,
                "ocr_text": ocr_text, "numeric": numeric, "global_hist": global_hist,
                "dominant_colors": dominant_colors, "color_moments": color_moments_feat}

    def _build_vectors(self, payload):
        cnn_global = payload["cnn"]
        cnn_local  = payload["cnn_local"]
        ocr_text = payload["ocr_text"]
        numeric = payload["numeric"]
        global_hist = payload["global_hist"]
        dominant_colors = payload.get("dominant_colors", np.zeros(15, dtype=np.float32))
        color_moments_feat = payload.get("color_moments", np.zeros(9, dtype=np.float32))

        if self.pca is not None and self.use_pca:
            coarse = self.pca.transform([cnn_global]).astype(np.float32)[0]
            local_reduced = self.pca.transform([cnn_local]).astype(np.float32)[0]
        else:
            coarse = cnn_global[:self.coarse_dim].astype(np.float32)
            local_reduced = cnn_local[:self.coarse_dim].astype(np.float32)

        if self.vectorizer is not None:
            tfidf = self.vectorizer.transform([ocr_text]).astype(np.float32).toarray()[0]
        else:
            tfidf = np.zeros((256,), dtype=np.float32)

        fine = np.concatenate([
            coarse,
            0.8 * local_reduced,
            3.0 * tfidf,
            5.0 * global_hist,
            8.0 * dominant_colors,
            6.0 * color_moments_feat,
            10.0 * numeric
        ], axis=0).astype(np.float32)

        return coarse, fine

    # ---------- Storage ----------
    def add_product(self, product_name, product_sku, img_array):
        try:
            self.cursor.execute(
                "INSERT INTO products (name, sku) VALUES (%s, %s) "
                "ON CONFLICT (name) DO UPDATE SET name=EXCLUDED.name, sku=EXCLUDED.sku RETURNING id",
                (product_name, product_sku)
            )
            product_id = self.cursor.fetchone()[0]

            payload = self._extract_payload(img_array)

            # optional DeepSeek to enrich stored payload
            ds_meta = deepseek_read_label(self._crop_for_ds(img_array))
            payload["deepseek_meta"] = ds_meta

            vector_id = str(uuid.uuid4())
            feature_path = os.path.join(self.features_dir, f"{vector_id}_feature.joblib")
            image_path   = os.path.join(self.features_dir, f"{vector_id}_image.jpg")

            joblib.dump(payload, feature_path)
            cv2.imwrite(image_path, img_array)

            self.cursor.execute(
                "INSERT INTO product_vectors (id, product_id, file_path, created_at) VALUES (%s, %s, %s, %s)",
                (vector_id, product_id, feature_path, datetime.now())
            )
            self.conn.commit()
            self.is_trained = False
            print(f"Added vector to product: {product_name} (SKU: {product_sku})")
            return True
        except Exception as e:
            self.conn.rollback()
            print(f"Error adding product: {str(e)}")
            return False

    # ---------- ANN index ----------
    def _normalize(self, X):
        X = X.astype(np.float32)
        norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
        return X / norms

    def _build_faiss(self):
        if not _HAS_FAISS or self.X is None or len(self.X) == 0:
            self.faiss_index = None
            return
        Xn = self._normalize(self.X)
        self.faiss_index = faiss.IndexFlatIP(Xn.shape[1])
        self.faiss_index.add(Xn)

    # ---------- Training ----------
    def train(self):
        # block recognition pipeline until there are at least 2 products
        if not self._has_min_products(2):
            self.is_trained = False
            print("Recognition disabled: need at least 2 products in DB.")
            return False

        self.cursor.execute("""
            SELECT pv.file_path, p.name
            FROM product_vectors pv
            JOIN products p ON p.id = pv.product_id
        """)
        rows = self.cursor.fetchall()
        if not rows:
            print("No vectors available for training.")
            self.is_trained = False
            return False

        payloads, labels, paths = [], [], []
        for file_path, name in rows:
            try:
                payloads.append(joblib.load(file_path))
                labels.append(name)
                paths.append(file_path)
            except Exception as e:
                print(f"Could not load {file_path}: {e}")

        if not payloads:
            print("No product vectors found for training.")
            self.is_trained = False
            return False

        # PCA on CNN global
        cnn_bank = np.stack([p["cnn"] for p in payloads]).astype(np.float32)
        try:
            n_comp = min(self.coarse_dim, len(cnn_bank))
            self.pca = PCA(n_components=n_comp, random_state=42)
            coarse_bank = self.pca.fit_transform(cnn_bank).astype(np.float32)
            self.use_pca = True
            self.coarse_dim = n_comp
            print(f"PCA({n_comp}) fitted.")
        except Exception as e:
            print(f"PCA failed ({e}), fallback raw.")
            self.pca = None
            self.use_pca = False
            coarse_bank = cnn_bank[:, :self.coarse_dim].astype(np.float32)

        # TF-IDF
        self.vectorizer = TfidfVectorizer(
            token_pattern=r"(?u)\b[\w\.\%]+\b",
            lowercase=True,
            max_features=256,
            ngram_range=(1,2),
            min_df=1
        )
        ocr_texts = [p["ocr_text"] for p in payloads]
        if any(text.strip() for text in ocr_texts):
            try:
                _ = self.vectorizer.fit(ocr_texts)
            except ValueError as e:
                print(f"TF-IDF fitting failed: {e}. Using placeholder.")
                self.vectorizer = TfidfVectorizer(max_features=1)
                _ = self.vectorizer.fit(["placeholder"])
        else:
            print("No valid OCR text. Using placeholder.")
            self.vectorizer = TfidfVectorizer(max_features=1)
            _ = self.vectorizer.fit(["placeholder"])

        # Banks + caches
        fine_bank = []
        tfidf_bank = []
        numeric_bank = []
        dom_colors_bank = []
        color_moments_bank = []
        badge_bank = []

        for p in payloads:
            _, f = self._build_vectors(p)
            fine_bank.append(f)
            tfidf_bank.append(self.vectorizer.transform([p["ocr_text"]]).astype(np.float32).toarray()[0])
            numeric_bank.append(p["numeric"])
            dom_colors_bank.append(p.get("dominant_colors", np.zeros(15, np.float32)))
            color_moments_bank.append(p.get("color_moments", np.zeros(9, np.float32)))
            badge_bank.append(p.get("deepseek_meta", {}).get("badge_color"))

        fine_bank = np.stack(fine_bank).astype(np.float32)
        fine_bank_norm = fine_bank / (np.linalg.norm(fine_bank, axis=1, keepdims=True) + 1e-8)
        tfidf_bank = np.stack(tfidf_bank).astype(np.float32)
        numeric_bank = np.stack(numeric_bank).astype(np.float32)
        dom_colors_bank = np.stack(dom_colors_bank).astype(np.float32)
        color_moments_bank = np.stack(color_moments_bank).astype(np.float32)

        self.X = coarse_bank
        self.y = labels
        self.labels_by_index = self.y[:]
        self.file_paths = paths
        self.img_path_by_index = [fp.replace("_feature.joblib","_image.jpg") for fp in self.file_paths]
        self.fine_bank = fine_bank
        self.fine_bank_norm = fine_bank_norm
        self.tfidf_bank = tfidf_bank
        self.numeric_bank = numeric_bank
        self.dom_colors_bank = dom_colors_bank
        self.color_moments_bank = color_moments_bank
        self.badge_bank = badge_bank
        self.is_trained = True

        # Indexes
        if _HAS_FAISS:
            self._build_faiss()
            print(f"FAISS index built on {len(self.X)} vectors.")
        else:
            self.knn = NearestNeighbors(n_neighbors=max(1, min(50, len(self.X))), metric='cosine')
            self.knn.fit(self.X)
            print(f"KNN (cosine) fitted on {len(self.X)} vectors. (Install faiss-cpu for speed)")

        # Per-label centroid stats
        self.sku_stats = {}
        groups = defaultdict(list)
        for vec, label in zip(self.fine_bank_norm, self.y):
            groups[label].append(vec)
        for label, vecs in groups.items():
            A = np.stack(vecs).astype(np.float32)
            c = np.mean(A, axis=0)
            dists = 1 - (A @ c)
            self.sku_stats[label] = (c, float(np.mean(dists)), float(np.std(dists) + 1e-6))

        uniq = len(set(self.y))
        print(f"Trained {len(self.X)} vectors across {uniq} products.")

        return True

    # ---------- Geometry (AKAZE) ----------
    def _geo_score(self, img_q, img_r):
        try:
            if img_q is None or img_r is None:
                return 0.0
            img_q = self._label_roi(img_q)
            img_r = self._label_roi(img_r)
            gq = cv2.cvtColor(img_q, cv2.COLOR_BGR2GRAY)
            gr = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
            akaze = cv2.AKAZE_create()
            kq, dq = akaze.detectAndCompute(gq, None)
            kr, dr = akaze.detectAndCompute(gr, None)
            if dq is None or dr is None or len(kq) < 10 or len(kr) < 10:
                return 0.0
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            matches = bf.knnMatch(dq, dr, k=2)
            good = [m for m, n in matches if m.distance < 0.75 * n.distance] if matches else []
            denom = float(max(10, min(len(kq), len(kr))))
            return max(0.0, min(1.0, len(good) / denom))
        except Exception:
            return 0.0

    # ---------- Scoring helpers ----------
    @staticmethod
    def _cos(a, b):
        na = np.linalg.norm(a); nb = np.linalg.norm(b)
        if na == 0 or nb == 0: return 0.0
        return float(np.dot(a, b) / (na * nb))

    def _color_similarity_from_banks(self, q_payload, i):
        q_colors = q_payload.get("dominant_colors", np.zeros(15, np.float32))
        r_colors = self.dom_colors_bank[i]
        q_mom    = q_payload.get("color_moments", np.zeros(9, np.float32))
        r_mom    = self.color_moments_bank[i]
        return 0.7 * self._cos(q_colors, r_colors) + 0.3 * self._cos(q_mom, r_mom)

    def _volume_from_any(self, numeric_vec, name_text):
        v_from_numeric = numeric_vec[2] * 1000.0 if numeric_vec is not None and numeric_vec[2] > 0 else None
        v_from_name = _parse_volume_ml_from_text(name_text or "")
        return v_from_numeric if v_from_numeric is not None else v_from_name

    # ---------- Recognition ----------
    def recognize(self, img_array):
        # ensure db has >=2 products; otherwise refuse recognition
        if not self._has_min_products(2):
            return None

        if not self.is_trained and not self.train():
            return None

        # Build query vectors
        q_payload = self._extract_payload(img_array)
        q_coarse, q_fine = self._build_vectors(q_payload)
        q_coarse_n = q_coarse.astype(np.float32)
        q_coarse_n /= (np.linalg.norm(q_coarse_n) + 1e-8)
        qf = q_fine / (np.linalg.norm(q_fine) + 1e-8)
        q_tfidf = self.vectorizer.transform([q_payload["ocr_text"]]).astype(np.float32).toarray()[0]
        qn = q_payload["numeric"]

        # Optional DeepSeek numeric/badge
        ds_percent = None
        ds_volume  = None
        query_badge = None
        try:
            ds = deepseek_read_label(self._crop_for_ds(img_array))
            ds_percent = ds.get("percent")
            ds_volume  = ds.get("volume_ml")
            query_badge = ds.get("badge_color")
        except Exception:
            pass

        # ANN retrieval (top-K)
        K = min(50, len(self.X))
        if _HAS_FAISS and self.faiss_index is not None:
            sims, idx = self.faiss_index.search(q_coarse_n[None, :], K)
            cand_idx = idx[0].tolist()
            base_sims = sims[0].tolist()
        else:
            dist, idx = self.knn.kneighbors([q_coarse], n_neighbors=K)
            cand_idx = idx[0].tolist()
            base_sims = [1.0 - float(d) for d in dist[0].tolist()]

        # Re-rank with penalties
        def score_candidate(i, base_sim):
            rf = self.fine_bank_norm[i]
            sim_visual = float(np.dot(qf, rf))

            rt = self.tfidf_bank[i]
            denom_t = (np.linalg.norm(q_tfidf) + 1e-8) * (np.linalg.norm(rt) + 1e-8)
            sim_ocr = float(np.dot(q_tfidf, rt) / denom_t) if denom_t > 0 else 0.0

            rn = self.numeric_bank[i]
            denom_n = (np.linalg.norm(qn) + 1e-8) * (np.linalg.norm(rn) + 1e-8)
            sim_num = float(np.dot(qn, rn) / denom_n) if denom_n > 0 else 0.0

            sim_color = self._color_similarity_from_banks(q_payload, i)

            score = (0.35*base_sim + 0.30*sim_visual + 0.20*sim_ocr + 0.05*sim_num + 0.10*sim_color)
            return score, sim_ocr

        prelim = []
        for base, i in zip(base_sims, cand_idx):
            s, sim_ocr = score_candidate(i, base)
            prelim.append((i, s, sim_ocr))
        prelim.sort(key=lambda x: x[1], reverse=True)
        prelim = prelim[:max(30, min(100, len(prelim)))]

        # Consolidate by label: keep best view per product
        best_by_label = {}
        for i, s, sim_ocr in prelim:
            lbl = self.labels_by_index[i]
            if lbl not in best_by_label or s > best_by_label[lbl][1]:
                best_by_label[lbl] = (i, s, sim_ocr)

        # Geometry only once per label
        finals = []
        for lbl, (i, s, sim_ocr) in best_by_label.items():
            r_img_path = self.img_path_by_index[i]
            r_img = cv2.imread(r_img_path) if r_img_path and os.path.exists(r_img_path) else None
            geo = self._geo_score(img_array, r_img)
            finals.append((i, lbl, s + 0.20*geo, sim_ocr, geo))
        finals.sort(key=lambda x: x[2], reverse=True)

        if not finals:
            return None

        # Margin vs. second best label
        if len(finals) >= 2:
            margin = finals[0][2] - finals[1][2]
            if margin < self.MIN_MARGIN:
                return None

        best_idx, best_label, best_score, best_sim_ocr, best_geo = finals[0]

        # Floors: OCR and geometry must be reasonable
        if best_sim_ocr < self.MIN_OCR:
            return None
        if best_geo < self.MIN_GEO:
            return None

        # Badge color guard (if available from stored payload)
        ref_badge = self.badge_bank[best_idx] if self.badge_bank else None
        if query_badge and ref_badge and query_badge != ref_badge:
            return None

        # ---------- NUMERIC GUARD (STRICT) ----------
        # Reference values from stored numeric OR from its name text
        rn = self.numeric_bank[best_idx]
        r_volume  = self._volume_from_any(rn, self.labels_by_index[best_idx])
        r_percent = rn[0] * 10.0 if rn[0] > 0 else _parse_percent_from_text(self.labels_by_index[best_idx])

        # Query values from OCR/DS
        q_percent_ocr = qn[0] * 10.0 if qn[0] > 0 else None
        q_volume_ocr  = qn[2] * 1000.0 if qn[2] > 0 else None
        q_percent = ds_percent if ds_percent is not None else q_percent_ocr
        q_volume  = ds_volume  if ds_volume  is not None else q_volume_ocr

        # Volume check when both known
        if (r_volume is not None) and (q_volume is not None):
            if abs(q_volume - r_volume) > self.ML_TOL:
                return None

        # Percent check when both known
        if (r_percent is not None) and (q_percent is not None):
            if abs(q_percent - r_percent) > self.PCT_TOL:
                return None

        # Centroid guard
        c, mu, sigma = self.sku_stats.get(best_label, (None, 0.3, 0.1))
        if c is not None:
            dist_to_centroid = 1.0 - float(np.dot(qf, c))
            if dist_to_centroid > (mu + 3.0*sigma):
                return None

        confidence = max(0.0, min(1.0, best_score))
        if confidence < self.sim_threshold_global:
            return None

        # Fetch SKU for the recognized product
        best_file_path = self.file_paths[best_idx]
        vector_id = os.path.basename(best_file_path).replace('_feature.joblib', '')
        self.cursor.execute('''
            SELECT p.sku FROM products p 
            JOIN product_vectors pv ON p.id = pv.product_id 
            WHERE pv.id = %s
        ''', (vector_id,))
        sku_row = self.cursor.fetchone()
        best_sku = sku_row[0] if sku_row else 'Unknown'

        return {"recognized": True, "product": best_label, "sku": best_sku, "confidence": confidence}

    # ---------- Debug recognize ----------
    def debug_recognize(self, img_array):
        if not self._has_min_products(2):
            print("Not enough products in DB (<2).")
            return None
        if not self.is_trained and not self.train():
            return None
        print("DEBUG recognize...")
        q_payload = self._extract_payload(img_array)
        q_coarse, q_fine = self._build_vectors(q_payload)
        qf = q_fine / (np.linalg.norm(q_fine) + 1e-8)
        q_coarse_n = q_coarse / (np.linalg.norm(q_coarse) + 1e-8)
        q_tfidf = self.vectorizer.transform([q_payload["ocr_text"]]).astype(np.float32).toarray()[0]
        qn = q_payload["numeric"]
        
        # Optional DeepSeek numeric/badge
        ds_percent = None
        ds_volume = None
        query_badge = None
        try:
            ds = deepseek_read_label(self._crop_for_ds(img_array))
            ds_percent = ds.get("percent")
            ds_volume = ds.get("volume_ml")
            query_badge = ds.get("badge_color")
        except Exception:
            pass
    
        K = min(5, len(self.X))
        if _HAS_FAISS and self.faiss_index is not None:
            sims, idx = self.faiss_index.search(q_coarse_n[None,:].astype(np.float32), K)
            cand_idx = idx[0].tolist()
            base_sims = sims[0].tolist()
        else:
            dist, idx = self.knn.kneighbors([q_coarse], n_neighbors=K)
            cand_idx = idx[0].tolist()
            base_sims = [1.0 - float(d) for d in dist[0].tolist()]
        print(f"Found {K} candidates, best sim: {base_sims[0]:.3f}")
        prelim = []
        for base_sim, i in zip(base_sims, cand_idx):
            rf = self.fine_bank_norm[i]
            sim_visual = float(np.dot(qf, rf))
            rt = self.tfidf_bank[i]
            denom_t = (np.linalg.norm(q_tfidf) + 1e-8) * (np.linalg.norm(rt) + 1e-8)
            sim_ocr = float(np.dot(q_tfidf, rt) / denom_t) if denom_t > 0 else 0.0
            sim_color = self._color_similarity_from_banks(q_payload, i)
            s = 0.35*base_sim + 0.30*sim_visual + 0.20*sim_ocr + 0.10*sim_color + 0.05*0.0
            prelim.append((i, s))
            print(f"  Candidate: {self.y[i]} base={base_sim:.3f} visual={sim_visual:.3f} ocr={sim_ocr:.3f} color={sim_color:.3f} score={s:.3f}")
        prelim.sort(key=lambda x: x[1], reverse=True)
        finals = []
        for i, s in prelim:
            r_img = None
            pth = self.img_path_by_index[i] if i < len(self.img_path_by_index) else None
            if pth and os.path.exists(pth):
                r_img = cv2.imread(pth)
            geo = self._geo_score(img_array, r_img)
            finals.append((i, s + 0.20*geo))
            print(f"  After geometry: {self.y[i]} score={s+0.20*geo:.3f} geo={geo:.3f}")
        finals.sort(key=lambda x: x[1], reverse=True)
        best_idx = finals[0][0]
        best_score = finals[0][1]
        best_label = self.y[best_idx]
        
        # Badge color guard
        ref_badge = self.badge_bank[best_idx] if self.badge_bank else None
        if query_badge and ref_badge and query_badge != ref_badge:
            print("❌ Rejected by badge color mismatch")
            return None
        
        # ---------- NUMERIC GUARD (STRICT) ----------
        rn = self.numeric_bank[best_idx]
        r_volume = self._volume_from_any(rn, self.labels_by_index[best_idx])
        r_percent = rn[0] * 10.0 if rn[0] > 0 else _parse_percent_from_text(self.labels_by_index[best_idx])
        
        q_percent_ocr = qn[0] * 10.0 if qn[0] > 0 else None
        q_volume_ocr = qn[2] * 1000.0 if qn[2] > 0 else None
        q_percent = ds_percent if ds_percent is not None else q_percent_ocr
        q_volume = ds_volume if ds_volume is not None else q_volume_ocr
        
        # Volume check when both known
        if (r_volume is not None) and (q_volume is not None):
            if abs(q_volume - r_volume) > self.ML_TOL:
                print(f"❌ Rejected by volume mismatch: query {q_volume}ml vs ref {r_volume}ml")
                return None
        if (r_percent is not None) and (q_percent is not None):
            if abs(q_percent - r_percent) > self.PCT_TOL:
                print(f"❌ Rejected by percent mismatch: query {q_percent}% vs ref {r_percent}%")
                return None
        
        print(f"Best: {best_label} score={best_score:.3f}")
        
        # Centroid guard
        c, mu, sigma = self.sku_stats.get(best_label, (None, 0.3, 0.1))
        if c is not None:
            dist_to_centroid = 1.0 - float(np.dot(qf, c))
            threshold = max(mu + 5.0*sigma, 0.08)
            print(f"🎯 Centroid check: dist={dist_to_centroid:.3f}, threshold={threshold:.3f}, pass={dist_to_centroid <= threshold}")
            if dist_to_centroid > threshold:
                print("❌ Failed centroid validation")
                return None
        
        # Return with proper SKU lookup
        try:
            best_file_path = self.file_paths[best_idx]
            vector_id = os.path.basename(best_file_path).replace('_feature.joblib', '')
            self.cursor.execute('''
                SELECT p.sku FROM products p 
                JOIN product_vectors pv ON p.id = pv.product_id 
                WHERE pv.id = %s
            ''', (vector_id,))
            sku_row = self.cursor.fetchone()
            best_sku = sku_row[0] if sku_row else 'Unknown'
            print(f"🏷️ Retrieved SKU: {best_sku}")
        except Exception as e:
            print(f"❌ SKU lookup error: {e}")
            best_sku = 'Unknown'
        
        confidence = max(0.0, min(1.0, best_score))
        if confidence < self.sim_threshold_global:
            print(f"❌ Confidence {confidence:.3f} below threshold {self.sim_threshold_global}")
            return None
        
        return {"recognized": True, "product": best_label, "sku": best_sku, "confidence": confidence}


# =========================
# CLI / Camera flow
# =========================
def main():
    try:
        is_server = os.environ.get('DISPLAY') is None and os.name != 'nt'
        is_container = os.path.exists('/.dockerenv')
        if is_server or is_container:
            print("Headless/container environment. Use FastAPI endpoints:")
            print(" - POST /scan")
            print(" - POST /add")
            return
    except:
        pass

    scanner = ProductScannerSQL(db_url=db_url)

    def _opencv_has_gui():
        try:
            cv2.namedWindow("__test_gui__", cv2.WINDOW_NORMAL)
            cv2.imshow("__test_gui__", np.zeros((2,2,3), dtype=np.uint8))
            cv2.waitKey(1)
            cv2.destroyWindow("__test_gui__")
            return True
        except Exception:
            return False

    _GUI_AVAILABLE = _opencv_has_gui()
    if not _GUI_AVAILABLE:
        print("OpenCV GUI not available. Use FastAPI endpoints:")
        print(" - POST /scan")
        print(" - POST /add")
        return

    video_devices = [0, 2, 1]
    cap = None
    for device in video_devices:
        try:
            cap = cv2.VideoCapture(device)
            if cap.isOpened():
                print(f"Opened camera device {device}")
                break
            cap.release()
        except:
            continue
    if cap is None or not cap.isOpened():
        print("Could not access any webcam. Use FastAPI endpoints.")
        return

    print("""
    Controls:
    [s] - Scan current frame
    [a] - Add new product
    [q] - Quit
    """)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera error.")
            break

        try:
            cv2.imshow('Product Scanner', frame)
        except cv2.error:
            print("Cannot display video. Use FastAPI endpoints.")
            break

        key = cv2.waitKey(1) & 0xFF

        if key == ord('s') or key == 19:  # Ctrl+S
            result = scanner.recognize(frame)
            print(result)
            if result:
                print(f"Product recognized: {result['product']} | SKU: {result['sku']} | Confidence: {result['confidence']:.2f}")
                confirm = input("Is this correct? (y/n): ").strip().lower()
                if confirm == 'n':
                    # allow adding a new product even after a wrong recognition
                    product_name = input("Enter NEW product name (e.g., 'Ivermectina 100 ml 3.15%'): ").strip()
                    product_sku  = input("Enter NEW product SKU: ").strip()
                    if product_name and product_sku:
                        print(f"Capture at least 5 views of '{product_name}' (SKU: {product_sku}).")
                        print("Press [SPACE] to capture each photo. Press [ESC] to cancel.")
                        captured = 0
                        while captured < 5:
                            ret, frame = cap.read()
                            if not ret:
                                print("Camera error."); break
                            display = frame.copy()
                            cv2.putText(display, f"Capture {captured + 1}/5 - [SPACE]=save  [ESC]=cancel",
                                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                            cv2.imshow('Capture Product Views', display)
                            key_inner = cv2.waitKey(1)
                            if key_inner == 32:  # SPACE
                                if scanner.add_product(product_name, product_sku, frame):
                                    print(f"View {captured + 1} saved."); captured += 1
                            elif key_inner == 27:  # ESC
                                print("Capture cancelled."); captured = 0; break
                        try: cv2.destroyWindow('Capture Product Views')
                        except Exception: pass
                        if captured == 5:
                            print(f"Product '{product_name}' stored with 5 views.")
                            time.sleep(1)
            else:
                print("Product not recognized.")
                choice = input("Store this product? (y/n): ").strip().lower()
                if choice == 'y':
                    product_name = input("Enter product name (e.g., 'Ivermectina 500 ml 3.15%'): ").strip()
                    product_sku  = input("Enter product SKU: ").strip()
                    if product_name and product_sku:
                        print(f"Capture at least 5 views of '{product_name}' (SKU: {product_sku}).")
                        print("Press [SPACE] to capture each photo. Press [ESC] to cancel.")
                        captured = 0
                        while captured < 5:
                            ret, frame = cap.read()
                            if not ret:
                                print("Camera error.")
                                break
                            display = frame.copy()
                            cv2.putText(display, f"Capture {captured + 1}/5 - [SPACE]=save  [ESC]=cancel",
                                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                            cv2.imshow('Capture Product Views', display)
                            key_inner = cv2.waitKey(1)
                            if key_inner == 32:  # SPACE
                                if scanner.add_product(product_name, product_sku, frame):
                                    print(f"View {captured + 1} saved.")
                                    captured += 1
                            elif key_inner == 27:  # ESC
                                print("Capture cancelled."); captured = 0; break
                        try:
                            cv2.destroyWindow('Capture Product Views')
                        except Exception:
                            pass
                        if captured == 5:
                            print(f"Product '{product_name}' stored with 5 views.")
                            time.sleep(1)

        elif key == ord('a'):
            product_name = input("Enter product name: ").strip()
            product_sku = input("Enter product SKU: ").strip()
            if product_name and product_sku:
                if scanner.add_product(product_name, product_sku, frame):
                    print(f"Image added to product '{product_name}' (SKU: {product_sku})")
                else:
                    print("Failed to add product.")
            else:
                print("Both product name and SKU are required.")

        elif key == ord('q'):
            break

    cap.release()
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass
    print("Scanner closed.")


def is_running_in_container():
    return os.path.exists('/.dockerenv')


if __name__ == "__main__":
    if is_running_in_container():
        print("Running in Docker container")
        print("GUI features are disabled")
        print("Use FastAPI endpoints:")
        print(" - POST /scan")
        print(" - POST /add")
    else:
        main()
