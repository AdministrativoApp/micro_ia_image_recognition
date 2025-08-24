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
import mediapipe as mp
import base64, json, requests
from collections import defaultdict, Counter

# ---------------- Optional FAISS (ANN) ----------------
_HAS_FAISS = False
try:
    import faiss
    _HAS_FAISS = True
except Exception:
    _HAS_FAISS = False

# ---------------- Mediapipe (face/hands masking) ----------------
mp_face = mp.solutions.face_detection
mp_hands = mp.solutions.hands
mp_hands_detection = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_face_detection = mp.solutions.face_detection.FaceDetection(
    model_selection=1, min_detection_confidence=0.5
)

def mask_humans(img):
    if img is None or img.size == 0:
        raise ValueError("❌ mask_humans: Received empty image.")
    h, w, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    masked_img = img.copy()

    # Faces
    try:
        results = mp_face_detection.process(img_rgb)
        if results and results.detections:
            for det in results.detections:
                bbox = det.location_data.relative_bounding_box
                x1 = int(bbox.xmin * w); y1 = int(bbox.ymin * h)
                x2 = x1 + int(bbox.width * w); y2 = y1 + int(bbox.height * h)
                if x2 > x1 and y2 > y1:
                    region = masked_img[y1:y2, x1:x2]
                    if region.size > 0:
                        masked_img[y1:y2, x1:x2] = cv2.GaussianBlur(region, (55, 55), 30)
    except Exception:
        pass

    # Hands
    try:
        hand_results = mp_hands_detection.process(img_rgb)
        if hand_results and hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                x_coords = [int(lm.x * w) for lm in hand_landmarks.landmark]
                y_coords = [int(lm.y * h) for lm in hand_landmarks.landmark]
                x1 = max(min(x_coords) - 10, 0)
                y1 = max(min(y_coords) - 10, 0)
                x2 = min(max(x_coords) + 10, w)
                y2 = min(max(y_coords) + 10, h)
                if x2 > x1 and y2 > y1:
                    region = masked_img[y1:y2, x1:x2]
                    if region.size > 0:
                        masked_img[y1:y2, x1:x2] = cv2.GaussianBlur(region, (55, 55), 30)
    except Exception:
        pass

    return masked_img

# ---------------- Env / DB ----------------
load_dotenv()
USER = os.getenv('DB_USER')
PASSWORD = os.getenv('DB_PASSWORD')
DATABASE = os.getenv('DB_DATABASE')
HOST = os.getenv('DB_HOST')
PORT = os.getenv('DB_PORT')
db_url = f"postgresql://{USER}:{PASSWORD}@{HOST}:{PORT}/{DATABASE}"

# ---------------- DeepSeek config (kept; optional) ----------------
DEEPSEEK_API_KEY  = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_MODEL    = os.getenv("DEEPSEEK_MODEL", "deepseek-vl")
DEEPSEEK_API_BASE = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com")

def _b64_png_from_bgr(img_bgr):
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        raise RuntimeError("encode png failed")
    return base64.b64encode(buf.tobytes()).decode("utf-8")

def deepseek_read_label(img_bgr):
    """Kept for compatibility; pipeline does not depend on it."""
    if not DEEPSEEK_API_KEY:
        return {"percent": None, "volume_ml": None, "badge_color": None}
    img_b64 = _b64_png_from_bgr(img_bgr)
    prompt = (
        "UNIVERSAL PRODUCT LABEL READER — JSON ONLY\n"
        "\n"
        "Goal: From the given photo, read ONLY the text printed on the MAIN product’s label and return a JSON object with:\n"
        '  - percent: number|null  — concentration printed as a percent. Accept forms like "3.15%", "3,15 %", "3.15Z" (stylized %), "1 %", "70%".\n'
        "    • Use dot as decimal separator (3,15 → 3.15). Range 0–100. If multiple percents appear, pick the one that looks most central/prominent\n"
        "      or inside a circular/elliptical callout badge. If uncertain or absent, use null.\n"
        "  - volume_ml: number|null — liquid volume in milliliters (ml). Accept “500 ml”, “0.5 L”→500, “1 L”→1000, “50 cl”→500.\n"
        "    • Convert L/cl/lt to ml. Ignore mass units (g, kg, mg, oz) and counts (e.g., 12 pack). If not visible, null.\n"
        '  - badge_color: "red"|"blue"|"green"|"yellow"|"orange"|"purple"|"black"|"white"|"other"|null —\n'
        "    • Dominant color of a circular/elliptical numeric callout (often containing the percent). If no such badge, null.\n"
        "\n"
        "Rules:\n"
        "• Read the largest/closest product only; ignore background items, screens, UI, reflections, barcodes, prices, phone numbers, dates.\n"
        "• Consider labels in any language (en/es/pt etc.).\n"
        "• Do NOT guess. If a field is not clearly visible, return null.\n"
        "• Output MUST be exactly one JSON object with only these keys and simple values (no extra text).\n"
        "\n"
        "Examples:\n"
        '• Red round badge “3.15Z”, text “500 ml”  → {"percent": 3.15, "volume_ml": 500, "badge_color": "red"}\n'
        '• Blue badge “1%”, text “250ml”          → {"percent": 1,    "volume_ml": 250, "badge_color": "blue"}\n'
        '• Label shows “70% alcohol”, “1 L”       → {"percent": 70,   "volume_ml": 1000,"badge_color": "other"}\n'
        '• No percent or volume visible           → {"percent": null, "volume_ml": null,"badge_color": null}\n'
    )
    url = f"{DEEPSEEK_API_BASE}/v1/chat/completions"
    headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}
    data = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": [
                 {"type":"input_text","text":"Return JSON only."},
                {"type":"input_image","image_url": f"data:image/png;base64,{img_b64}"}
            ]}
        ],
         "temperature": 0,
         "top_p": 0.1,
         "response_format": {"type": "json_object"}  # if the API supports it; safe to include
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

# ---------------- OCR / Numeric parsing ----------------
NUM_PERCENT = re.compile(r"(\d+(?:\.\d+)?)(?:\s*(?:%|z|Z))?")
NUM_WITH_UNIT = re.compile(r"(\d+(?:\.\d+)?)[\s\-]*(ml|mL|ML|g|kg|oz|L)\b", re.I)

def _normalize_text(txt: str) -> str:
    txt = txt.lower()
    txt = re.sub(r"[^a-z0-9%\. ]+", " ", txt)
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

def extract_ocr_text(img_bgr):
    texts = []
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    if _HAS_EASYOCR and _EASYOCR_READER is not None:
        res = _EASYOCR_READER.readtext(img_rgb, detail=0, paragraph=True)
        texts = res if res else []
    elif _HAS_TESS:
        try:
            import pytesseract
            texts = [pytesseract.image_to_string(img_rgb)]
        except Exception:
            texts = []
    return _normalize_text(" ".join(texts))

def extract_numeric_cues(text: str):
    raw_pcts = [float(m.group(1)) for m in NUM_PERCENT.finditer(text.replace(" ", ""))]
    percents = [p for p in raw_pcts if 0.0 <= p <= 10.0]  # sane concentration range
    sized = [(float(m.group(1)), m.group(2).lower()) for m in NUM_WITH_UNIT.finditer(text)]
    max_pct = max(percents) if percents else -1.0
    cnt_pct = float(len(percents))
    unit2ml = {"ml":1, "l":1000}
    ml_vals = [v*unit2ml[u] for v,u in sized if u in unit2ml]
    max_ml = max(ml_vals) if ml_vals else -1.0
    cnt_sz = float(len(sized))
    return np.array([max_pct/10.0, cnt_pct/5.0, max_ml/1000.0, cnt_sz/5.0], dtype=np.float32)

def global_hsv_hist(img_bgr, bins=16):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h = hsv[...,0].flatten()
    hist, _ = np.histogram(h, bins=bins, range=(0,180), density=True)
    return hist.astype(np.float32)

# ---------- NEW: generic multi-crop utility ----------
def _multicrops(img):
    """Return generic local crops to capture small label differences."""
    h, w = img.shape[:2]
    crops = []
    # center 70%
    cx1, cy1 = int(0.15*w), int(0.15*h)
    cx2, cy2 = int(0.85*w), int(0.85*h)
    crops.append(img[cy1:cy2, cx1:cx2])
    # top center
    crops.append(img[0:int(0.55*h), int(0.20*w):int(0.80*w)])
    # bottom center
    crops.append(img[int(0.45*h):h, int(0.20*w):int(0.80*w)])
    # left-middle
    crops.append(img[int(0.20*h):int(0.80*h), 0:int(0.55*w)])
    # right-middle
    crops.append(img[int(0.20*h):int(0.80*h), int(0.45*w):w])
    # ensure non-empty
    return [c for c in crops if c.size]


def _find_badge_roi(img_bgr):
    if img_bgr is None or img_bgr.size == 0:
        return None
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    sat = hsv[...,1] > 90
    val = hsv[...,2] > 80
    red1 = (hsv[...,0] < 10) & sat & val
    red2 = (hsv[...,0] > 160) & sat & val
    blue = (hsv[...,0] > 95) & (hsv[...,0] < 135) & sat & val
    mask = (red1 | red2 | blue).astype(np.uint8) * 255

    # stronger cleanup → smoother circle edges
    k = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.GaussianBlur(mask, (7,7), 0)

    # Run Hough on the masked region only
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_and(gray, gray, mask=mask)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=40,
                               param1=120, param2=40, minRadius=16, maxRadius=120)

    h, w = img_bgr.shape[:2]
    if circles is not None:
        circles = np.uint16(np.around(circles[0]))
        # biggest circle (badge) wins
        x, y, r = max(circles, key=lambda c: c[2])
        pad = int(0.25*r)
        x1, y1 = max(0, x-r-pad), max(0, y-r-pad)
        x2, y2 = min(w, x+r+pad), min(h, y+r+pad)
        roi = img_bgr[y1:y2, x1:x2]
        return roi if roi.size else None

    # fallback: your previous contour logic
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: 
        return None
    x,y,ww,hh = max((cv2.boundingRect(c) for c in cnts), key=lambda b:b[2]*b[3])
    pad = int(max(8, 0.2*max(ww,hh)))
    return img_bgr[max(0,y-pad):min(h,y+hh+pad), max(0,x-pad):min(w,x+ww+pad)]


def _ocr_digits_only(img_bgr):
    if img_bgr is None or img_bgr.size == 0:
        return None

    def _prep(img, invert=False):
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        g = cv2.bilateralFilter(g, 7, 50, 50)
        thr = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV if not invert else cv2.THRESH_BINARY,
                                    31, 15)
        thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)
        return cv2.resize(thr, None, fx=2.2, fy=2.2, interpolation=cv2.INTER_CUBIC)

    candidates = []
    for invert in (False, True):
        thr = _prep(img_bgr, invert=invert)
        try:
            import pytesseract
            # try “single char” and “single line”
            for psm in (10, 7, 8, 13):
                cfg = f"--oem 3 --psm {psm} -c tessedit_char_whitelist=0123456789.,%"
                txt = pytesseract.image_to_string(thr, config=cfg).strip()
                if txt:
                    candidates.append(txt)
        except Exception:
            pass
        # EasyOCR fallback
        try:
            if _HAS_EASYOCR and _EASYOCR_READER is not None:
                res = _EASYOCR_READER.readtext(thr, detail=0, paragraph=False)
                candidates += res or []
        except Exception:
            pass

    # normalize and vote
    nums = []
    for t in candidates:
        m = re.search(r"(\d+(?:[.,]\d+)?)", t)
        if not m: 
            continue
        v = float(m.group(1).replace(",", "."))
        if 0.0 <= v <= 10.0:
            nums.append(round(v, 2))

    if not nums:
        return None

    # prefer exact “1.0/1/3.15” if present, else median
    # (avoids 1 → 3 outliers)
    if 1.0 in nums or 1 in nums: 
        return 1.0
    if 3.15 in nums: 
        return 3.15
    return float(np.median(nums))

def read_badge_percent(img_bgr):
    """High-confidence % from the colored badge, if present."""
    roi = _find_badge_roi(img_bgr)
    if roi is None: 
        return None
    return _ocr_digits_only(roi)

def _badge_color_name(roi_bgr):
    """Return human color name for the badge ROI: 'red'/'blue'/'green'/'yellow'/'other'/None."""
    if roi_bgr is None or roi_bgr.size == 0:
        return None
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    h = hsv[..., 0]
    s = hsv[..., 1]
    v = hsv[..., 2]
    # basic saturation/brightness check
    if float(np.mean(s)) < 60 or float(np.mean(v)) < 60:
        return None
    # pick dominant hue from saturated pixels
    mask = s > 60
    hue = float(np.median(h[mask])) if np.any(mask) else float(np.median(h))
    if hue < 10 or hue > 160:
        return "red"
    if 95 < hue < 135:
        return "blue"
    if 25 < hue < 40:
        return "yellow"
    if 40 <= hue <= 85:
        return "green"
    return "other"


def _badge_color_from_image_path(feature_path):
    """
    Use the saved _image.jpg next to the feature to estimate badge color.
    """
    img_path = feature_path.replace("_feature.joblib", "_image.jpg")
    if not os.path.exists(img_path):
        return None
    img = cv2.imread(img_path)
    roi = _find_badge_roi(img)
    return _badge_color_name(roi)


# ---------------- Product Scanner (generic pipeline) ----------------
class ProductScannerSQL:
    def __init__(self, db_url, features_dir='features'):
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

        # Index / banks
        self.X = None
        self.y = []
        self.file_paths = []
        self.fine_bank = None
        self.fine_bank_norm = None
        self.tfidf_bank = None
        self.sku_stats = {}

        self.faiss_index = None
        self.knn = None
        self.is_trained = False

        self.sim_threshold_global = 0.45
        self.prod_index = None

    def audit_numeric_payloads(self):
        bad = []
        for fp, name in zip(self.file_paths or [], self.y or []):
            try:
                p = joblib.load(fp)
                rn = p.get("numeric")
                ok = isinstance(rn, (list, tuple, np.ndarray)) and len(rn) >= 3
                pct = rn[0]*10.0 if ok and rn[0] > 0 else None
                ml  = rn[2]*1000.0 if ok and rn[2] > 0 else None
                if pct is None or pct > 10 or pct < 0:
                    bad.append((name, fp, "percent", pct))
                if ml is not None and (ml < 20 or ml > 5000):
                    bad.append((name, fp, "ml", ml))
            except Exception as e:
                bad.append((name, fp, "load_error", str(e)))
        return bad


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
            raise ValueError("❌ preprocess_frame: empty image")
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

    def _extract_payload(self, img_array):
        cnn_global = self._cnn_raw(img_array)
        # NEW: multi-crop average to emphasize local print
        crops = _multicrops(img_array)
        if crops:
            cnn_locals = [self._cnn_raw(c) for c in crops]
            cnn_local = np.mean(np.stack(cnn_locals, axis=0), axis=0).astype(np.float32)
        else:
            cnn_local = cnn_global.copy()

        ocr_text = extract_ocr_text(img_array)
        numeric = extract_numeric_cues(ocr_text)
        global_hist = global_hsv_hist(img_array)
        return {"cnn": cnn_global, "cnn_local": cnn_local,
                "ocr_text": ocr_text, "numeric": numeric, "global_hist": global_hist}

    def _build_vectors(self, payload):
        cnn_global = payload["cnn"]
        cnn_local  = payload["cnn_local"]
        ocr_text = payload["ocr_text"]
        numeric = payload["numeric"]
        global_hist = payload["global_hist"]

        # Coarse uses global CNN only
        if self.pca is not None and self.use_pca:
            coarse = self.pca.transform([cnn_global]).astype(np.float32)[0]
            local_reduced = self.pca.transform([cnn_local]).astype(np.float32)[0]
        else:
            coarse = cnn_global[:self.coarse_dim].astype(np.float32)
            local_reduced = cnn_local[:self.coarse_dim].astype(np.float32)

        # TF-IDF
        if self.vectorizer is not None:
            tfidf = self.vectorizer.transform([ocr_text]).astype(np.float32).toarray()[0]
        else:
            tfidf = np.zeros((256,), dtype=np.float32)

        # Fine vector (multi-modal + local)
        fine = np.concatenate([
            coarse,                 # global
            0.8 * local_reduced,    # local emphasis
            3.0 * tfidf,            # OCR weighted
            global_hist,
            10.0 * numeric          # numeric weighted
        ], axis=0).astype(np.float32)

        return coarse, fine
    
    def _resolve_feature_path(self, file_path):
        """
        Return a valid absolute path to the feature file if possible.
        Tries original, absolute, features_dir+basename.
        Updates DB to the corrected path when fixed.
        """
        original = file_path
        candidates = []
        # 1) as-is
        candidates.append(file_path)
        # 2) absolute of as-is
        if not os.path.isabs(file_path):
            candidates.append(os.path.abspath(file_path))
        # 3) features_dir + basename
        candidates.append(os.path.abspath(os.path.join(self.features_dir, os.path.basename(file_path))))

        for p in candidates:
            if os.path.exists(p):
                if p != original:
                    try:
                        self.cursor.execute(
                            "UPDATE product_vectors SET file_path=%s WHERE file_path=%s",
                            (p, original)
                        )
                        self.conn.commit()
                        print(f"🛠️  Fixed stale path in DB:\n    {original}\n -> {p}")
                    except Exception as e:
                        print(f"⚠️ Could not update DB path ({e})")
                return p
        return None  # not found

    def _self_heal_feature(self, feature_path):
        """
        If the .joblib is missing but the paired _image.jpg exists,
        re-extract payload and save a new .joblib. Returns True if healed.
        """
        img_path = feature_path.replace("_feature.joblib", "_image.jpg")
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                try:
                    payload = self._extract_payload(img)
                    os.makedirs(os.path.dirname(feature_path), exist_ok=True)
                    joblib.dump(payload, feature_path)
                    print(f"🩹 Rebuilt missing feature from image:\n    {feature_path}")
                    return True
                except Exception as e:
                    print(f"❌ Failed to rebuild feature ({e})")
        return False
    
    def _tokenize(self, text: str):
        text = (text or "").lower()
        text = re.sub(r"[^a-z0-9%\. ]+", " ", text)
        return set(t for t in text.split() if t)

    def _majority(self, counter: Counter):
        return counter.most_common(1)[0][0] if counter else None
    
    def _ref_pct_from_payload_idx(self, i):
        try:
            rn = joblib.load(self.file_paths[i])["numeric"]
            return rn[0]*10.0 if rn[0] > 0 else None
        except Exception:
            return None

    def _pct_from_name(self, name):
        m = re.search(r"(\d+(?:\.\d+)?)\s*%", str(name))
        return float(m.group(1)) if m else None

    def _build_product_index(self):
        """
        Aggregate all saved views per product into a canonical profile
        (majority % / ml / badge color, union of tokens, file_paths list).
        """
        self.cursor.execute("""
            SELECT pv.file_path, p.name
            FROM product_vectors pv
            JOIN products p ON p.id = pv.product_id
        """)
        rows = self.cursor.fetchall()

        by_product = defaultdict(lambda: {
            "percents": Counter(),
            "volumes": Counter(),
            "colors":  Counter(),
            "tokens":  set(),
            "file_paths": []
        })

        for fp, name in rows:
            fpath = self._resolve_feature_path(fp) or fp
            if not os.path.exists(fpath):
                # try auto-heal into canonical location
                alt = os.path.abspath(os.path.join(self.features_dir, os.path.basename(fp)))
                if not os.path.exists(alt):
                    continue
                fpath = alt

            try:
                payload = joblib.load(fpath)
            except Exception:
                continue

            by_product[name]["file_paths"].append(fpath)

            # percent/ml from stored numeric (scaled) if available
            rn = payload.get("numeric")
            if isinstance(rn, (list, tuple, np.ndarray)) and len(rn) >= 3:
                try:
                    p_pct = rn[0] * 10.0 if rn[0] > 0 else None
                    p_ml  = rn[2] * 1000.0 if rn[2] > 0 else None
                except Exception:
                    p_pct = None; p_ml = None
                if isinstance(p_pct, (int,float)):
                    by_product[name]["percents"][round(float(p_pct), 3)] += 1
                if isinstance(p_ml, (int,float)):
                    by_product[name]["volumes"][round(float(p_ml), 1)] += 1

            # tokens from OCR text (if present)
            toks = self._tokenize(payload.get("ocr_text", ""))
            by_product[name]["tokens"].update(toks)

            # majority badge color from the saved image (optional, cheap)
            col = _badge_color_from_image_path(fpath)
            if col:
                by_product[name]["colors"][col] += 1

        self.prod_index = by_product
 
    
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

            # Force percent from badge if readable (keeps references correct)
            try:
                bp = read_badge_percent(self._label_roi(img_array))
                if bp is not None and 0.0 <= bp <= 10.0:
                    payload["numeric"][0] = float(bp) / 10.0  # 1% -> 0.10
            except Exception:
                pass

            vector_id = str(uuid.uuid4())
            feature_path = os.path.abspath(os.path.join(self.features_dir, f"{vector_id}_feature.joblib"))
            image_path   = os.path.abspath(os.path.join(self.features_dir, f"{vector_id}_image.jpg"))

            joblib.dump(payload, feature_path)
            cv2.imwrite(image_path, img_array)

            self.cursor.execute(
                "INSERT INTO product_vectors (id, product_id, file_path, created_at) VALUES (%s, %s, %s, %s)",
                (vector_id, product_id, feature_path, datetime.now())
            )
            self.conn.commit()
            self.is_trained = False
            self.prod_index = None
            print(f"✅ Added vector to product: {product_name} (SKU: {product_sku})")
            return True
        except Exception as e:
            self.conn.rollback()
            print(f"❌ Error adding product: {str(e)}")
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
        self.cursor.execute("""
            SELECT pv.file_path, p.name
            FROM product_vectors pv
            JOIN products p ON p.id = pv.product_id
        """)
        rows = self.cursor.fetchall()
        if not rows:
            print("⚠️ No vectors available for training.")
            return False

        payloads, labels, paths = [], [], []
        for file_path, name in rows:
            try:
                fpath = self._resolve_feature_path(file_path)
                if fpath is None:
                    # try auto-heal in canonical location
                    alt = os.path.abspath(os.path.join(self.features_dir, os.path.basename(file_path)))
                    if self._self_heal_feature(alt):
                        fpath = alt
                        try:
                            self.cursor.execute(
                                "UPDATE product_vectors SET file_path=%s WHERE file_path=%s",
                                (fpath, file_path)
                            )
                            self.conn.commit()
                        except Exception as e:
                            print(f"⚠️ Could not update DB after heal ({e})")
                    else:
                        print(f"⚠️ Missing feature file: {file_path}")
                        continue

                payloads.append(joblib.load(fpath))
                labels.append(name)
                paths.append(fpath)
            except Exception as e:
                print(f"⚠️ Could not load {file_path}: {e}")

        # If nothing loaded, abort gracefully (prevents np.stack crash)
        if not payloads:
            print("⚠️ No usable feature payloads found. "
                "Check that your 'features/' files exist and DB file paths are valid.")
            return False

        cnn_bank = np.stack([p["cnn"] for p in payloads]).astype(np.float32)
        try:
            n_comp = min(self.coarse_dim, len(cnn_bank))
            self.pca = PCA(n_components=n_comp, random_state=42)
            coarse_bank = self.pca.fit_transform(cnn_bank).astype(np.float32)
            self.use_pca = True
            self.coarse_dim = n_comp
            print(f"✅ PCA({n_comp}) fitted.")
        except Exception as e:
            print(f"⚠️ PCA failed ({e}), fallback raw.")
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
        _ = self.vectorizer.fit([p["ocr_text"] for p in payloads])

        # Fine bank (+ caches)
        fine_bank = []
        tfidf_bank = []
        for p in payloads:
            _, f = self._build_vectors(p)
            fine_bank.append(f)
            tfidf_bank.append(self.vectorizer.transform([p["ocr_text"]]).astype(np.float32).toarray()[0])

        fine_bank = np.stack(fine_bank).astype(np.float32)
        fine_bank_norm = fine_bank / (np.linalg.norm(fine_bank, axis=1, keepdims=True) + 1e-8)
        tfidf_bank = np.stack(tfidf_bank).astype(np.float32)

        # Store
        self.X = coarse_bank
        self.y = labels
        self.file_paths = paths
        self.fine_bank = fine_bank
        self.fine_bank_norm = fine_bank_norm
        self.tfidf_bank = tfidf_bank
        self.is_trained = True

        # Indexes
        if _HAS_FAISS:
            self._build_faiss()
            print(f"🔁 FAISS index built on {len(self.X)} vectors.")
        else:
            self.knn = NearestNeighbors(n_neighbors=max(1, min(50, len(self.X))), metric='cosine')
            self.knn.fit(self.X)
            print(f"🔁 KNN (cosine) fitted on {len(self.X)} vectors. (Install faiss-cpu for speed)")

        # Per-SKU centroid stats
        self.sku_stats = {}
        groups = defaultdict(list)
        for vec, label in zip(self.fine_bank_norm, self.y):
            groups[label].append(vec)
        for label, vecs in groups.items():
            A = np.stack(vecs).astype(np.float32)
            c = np.mean(A, axis=0)
            dists = 1 - (A @ c)
            self.sku_stats[label] = (c, float(np.mean(dists)), float(np.std(dists) + 1e-6))

        print(f"📦 Trained {len(self.X)} vectors across {len(set(self.y))} products.")
        return True

    # ---------- Helpers ----------
    def _crop_for_ds(self, img_bgr):
        h, w = img_bgr.shape[:2]
        y1, y2 = int(0.40*h), int(0.88*h)
        x1, x2 = int(0.45*w), int(0.97*w)
        crop = img_bgr[y1:y2, x1:x2]
        return crop if crop.size else img_bgr

    def _label_roi(self, img):
        """Generic label ROI: central area; avoids background/edges for ORB."""
        h, w = img.shape[:2]
        x1, y1 = int(0.12*w), int(0.18*h)
        x2, y2 = int(0.88*w), int(0.90*h)
        roi = img[y1:y2, x1:x2]
        return roi if roi.size else img

    def _load_ref_image(self, file_path):
        p = self._resolve_feature_path(file_path) or file_path
        img_path = (p if p else file_path).replace("_feature.joblib", "_image.jpg")
        if os.path.exists(img_path):
            return cv2.imread(img_path)
        return None

    def _geo_score(self, img_q, img_r):
        """Generic geometric consistency via ORB; returns [0,1]."""
        try:
            if img_q is None or img_r is None:
                return 0.0
            img_q = self._label_roi(img_q)
            img_r = self._label_roi(img_r)
            orb = cv2.ORB_create(nfeatures=1200)
            kq, dq = orb.detectAndCompute(cv2.cvtColor(img_q, cv2.COLOR_BGR2GRAY), None)
            kr, dr = orb.detectAndCompute(cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY), None)
            if dq is None or dr is None or len(kq) < 10 or len(kr) < 10:
                return 0.0
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            matches = bf.knnMatch(dq, dr, k=2)
            good = [m for m, n in matches if m.distance < 0.75 * n.distance] if matches else []
            denom = float(max(10, min(len(kq), len(kr))))
            return max(0.0, min(1.0, len(good) / denom))
        except Exception:
            return 0.0
        

    # ---------- Recognition ----------ss
    def recognize(self, img_array):
        if not self.is_trained and not self.train():
            return None

        # ---- Build query vectors ----
        q_payload = self._extract_payload(img_array)
        q_coarse, q_fine = self._build_vectors(q_payload)
        q_coarse_n = q_coarse.astype(np.float32)
        q_coarse_n /= (np.linalg.norm(q_coarse_n) + 1e-8)
        qf = q_fine / (np.linalg.norm(q_fine) + 1e-8)
        q_tfidf = self.vectorizer.transform([q_payload["ocr_text"]]).astype(np.float32).toarray()[0]
        qn = q_payload["numeric"]

        # ---- Numeric cues ----
        try:
            badge_percent = read_badge_percent(self._label_roi(img_array))
        except Exception:
            badge_percent = None

        ds_percent = None; ds_volume = None
        try:
            ds = deepseek_read_label(self._crop_for_ds(img_array))
            ds_percent = ds.get("percent"); ds_volume = ds.get("volume_ml")
        except Exception:
            pass

        ocr_percent = qn[0] * 10.0 if qn[0] > 0 else None
        ocr_volume  = qn[2] * 1000.0 if qn[2] > 0 else None

        try:
            roi = _find_badge_roi(img_array)
            badge_color = _badge_color_name(roi)
        except Exception:
            badge_color = None

        q_percent = None; q_percent_conf = 0.0
        if badge_percent is not None:
            q_percent, q_percent_conf = float(badge_percent), 0.95
        elif ds_percent is not None:
            q_percent, q_percent_conf = float(ds_percent), 0.80
        elif ocr_percent is not None:
            q_percent, q_percent_conf = float(ocr_percent), 0.60

        q_volume = None; q_volume_conf = 0.0
        if ds_volume is not None:
            q_volume, q_volume_conf = float(ds_volume), 0.75
        elif ocr_volume is not None:
            q_volume, q_volume_conf = float(ocr_volume), 0.55

        if badge_color == "blue" and q_percent is not None and 0.7 <= q_percent <= 1.4:
            q_percent, q_percent_conf = 1.0, 0.99

        # ---- Stage A: coarse retrieval (once) ----
        K = min(200, len(self.X))
        if _HAS_FAISS and self.faiss_index is not None:
            sims, idx = self.faiss_index.search(q_coarse_n[None, :], K)
            cand_idx = idx[0].tolist()
            base_sims = sims[0].tolist()
        else:
            dist, idx = self.knn.kneighbors([q_coarse], n_neighbors=K)
            cand_idx = idx[0].tolist()
            base_sims = [1.0 - float(d) for d in dist[0].tolist()]

        # ---- DeepSeek hard gate on % (authoritative) ----
        FORCE_DS_HARD = True
        if FORCE_DS_HARD and ds_percent is not None:
            target_pct = float(ds_percent)
            pct_gate = 0.30
            kept = []
            for i, b in zip(cand_idx, base_sims):
                rp = self._ref_pct_from_payload_idx(i)
                if rp is None:
                    rp = self._pct_from_name(self.y[i])
                if rp is not None and abs(rp - target_pct) <= pct_gate:
                    kept.append((i, b))
            if not kept:
                print(f"❌ DeepSeek says {target_pct}% and no reference matches → reject.")
                return None
            cand_idx, base_sims = map(list, zip(*kept))
            print(f"🧱 DeepSeek % gate kept {len(cand_idx)}/{K} (±{pct_gate:.2f}%)")

        if not cand_idx:
            return None

        # ---- Group by family (name) and keep top families ----
        fam_scores = {}
        for i, s in zip(cand_idx, base_sims):
            fam = self.y[i]
            fam_scores[fam] = max(fam_scores.get(fam, -1.0), s)
        top_families = sorted(fam_scores.items(), key=lambda x: x[1], reverse=True)[:10]
        fam_names = [f for f, _ in top_families]

        if self.prod_index is None:
            self._build_product_index()

        # ---- Mandatory gates within families ----
        fam_member_idx = [i for i in cand_idx if self.y[i] in fam_names]
        fam_member_base = [b for i, b in zip(cand_idx, base_sims) if self.y[i] in fam_names]
        if not fam_member_idx:
            return None

        def _ref_pct(i):
            try:
                rn = joblib.load(self.file_paths[i])["numeric"]
                return rn[0]*10.0 if rn[0] > 0 else None
            except Exception:
                return None

        def _ref_ml(i):
            try:
                rn = joblib.load(self.file_paths[i])["numeric"]
                return rn[2]*1000.0 if rn[2] > 0 else None
            except Exception:
                return None

        pct_gate = 0.30 if (q_percent is not None and q_percent_conf >= 0.90) else 0.60
        ml_gate  = 80.0 if (q_volume  is not None and q_volume_conf  >= 0.70) else 160.0

        gated_idx, gated_base = fam_member_idx, fam_member_base
        if q_percent is not None:
            ki, kb = [], []
            for i, b in zip(gated_idx, gated_base):
                rp = _ref_pct(i)
                if rp is None or abs(rp - q_percent) <= pct_gate:
                    ki.append(i); kb.append(b)
            if not ki and q_percent_conf >= 0.80:
                print("❌ hard % gate reject")
                return None
            if ki:
                gated_idx, gated_base = ki, kb
                print(f"🧱 % gate kept {len(gated_idx)}/{len(fam_member_idx)} (±{pct_gate:.2f}%)")

        if q_volume is not None and gated_idx:
            ki, kb = [], []
            for i, b in zip(gated_idx, gated_base):
                rm = _ref_ml(i)
                if rm is None or abs(rm - q_volume) <= ml_gate:
                    ki.append(i); kb.append(b)
            if not ki and q_volume_conf >= 0.70:
                print("❌ hard ml gate reject")
                return None
            if ki:
                gated_idx, gated_base = ki, kb
                print(f"🧱 ml gate kept {len(gated_idx)}/{len(fam_member_idx)} (±{ml_gate:.0f}ml)")

        if not gated_idx:
            return None

        # ---- Re-rank remaining ----
        PCT_TOL_SOFT = 0.60
        PCT_TOL_HARD = 0.10
        ML_TOL       = 120.0

        agree_sources = 0
        src_vals = [v for v in (badge_percent, ds_percent, ocr_percent) if v is not None]
        if len(src_vals) >= 2:
            for a in src_vals:
                for b in src_vals:
                    if a is b:
                        continue
                    if abs(float(a) - float(b)) <= 0.30:
                        agree_sources = 2; break
                if agree_sources >= 2:
                    break

        def score_candidate(i, base_sim):
            rf = self.fine_bank_norm[i]
            sim_visual = float(np.dot(qf, rf))

            rt = self.tfidf_bank[i]
            sim_ocr = 0.0
            if np.linalg.norm(q_tfidf) > 0 and np.linalg.norm(rt) > 0:
                sim_ocr = float(np.dot(q_tfidf, rt) /
                                ((np.linalg.norm(q_tfidf)+1e-8)*(np.linalg.norm(rt)+1e-8)))

            rn = joblib.load(self.file_paths[i])["numeric"]
            sim_num = 0.0
            if np.linalg.norm(qn) > 0 and np.linalg.norm(rn) > 0:
                sim_num = float(np.dot(qn, rn) /
                                ((np.linalg.norm(qn)+1e-8)*(np.linalg.norm(rn)+1e-8)))

            r_percent = rn[0]*10.0 if rn[0] > 0 else None
            r_volume  = rn[2]*1000.0 if rn[2] > 0 else None

            penalty = 0.0
            if q_percent is not None and r_percent is not None:
                diff = abs(q_percent - r_percent)
                if agree_sources >= 2 and diff > PCT_TOL_HARD:
                    penalty += 0.60
                elif diff > PCT_TOL_SOFT:
                    penalty += 0.25
            if q_volume is not None and r_volume is not None:
                if abs(q_volume - r_volume) > ML_TOL:
                    penalty += 0.15 if q_volume_conf >= 0.70 else 0.05

            return (0.45*base_sim + 0.33*sim_visual + 0.17*sim_ocr + 0.05*sim_num) - penalty

        prelim = [(i, score_candidate(i, b)) for b, i in zip(gated_base, gated_idx)]
        prelim.sort(key=lambda x: x[1], reverse=True)
        prelim = prelim[:20]

        # ---- ORB geometric verification ----
        finals = []
        for i, s in prelim:
            r_img = self._load_ref_image(self.file_paths[i])
            geo = self._geo_score(img_array, r_img)
            finals.append((i, s + 0.20*geo))
        finals.sort(key=lambda x: x[1], reverse=True)

        best_idx = finals[0][0]
        best_score = finals[0][1]
        best_label = self.y[best_idx]

        # ---- Centroid acceptance ----
        c, mu, sigma = self.sku_stats.get(best_label, (None, 0.3, 0.1))
        if c is not None:
            dist_to_centroid = 1.0 - float(np.dot(qf, c))
            if dist_to_centroid > (mu + 2.0*sigma):
                return None

        confidence = max(0.0, min(1.0, best_score))
        if confidence < self.sim_threshold_global:
            return None
        return {"label": best_label, "confidence": confidence}




    
# ---------------- CLI / Camera flow (kept) ----------------
def main():
    # Check if running in headless/server environment
    try:
        is_server = os.environ.get('DISPLAY') is None and os.name != 'nt'
        is_container = os.path.exists('/.dockerenv')
        if is_server or is_container:
            print("⚠️ Running in headless/container environment. GUI features disabled.")
            print("🚀 Use the FastAPI endpoints instead:")
            print("   - POST /scan - Scan a product")
            print("   - POST /add - Add a new product")
            return
    except:
        pass

    scanner = ProductScannerSQL(db_url=db_url)
    scanner.train()
    issues = scanner.audit_numeric_payloads()
    if issues:
        print(f"[AUDIT] Found {len(issues)} payload issues")
        for name, fp, what, val in issues[:25]:
            print(f"  - {name} | {what}={val} | {fp}")
    else:
        print("[AUDIT] No payload issues found")

    # Try multiple video devices
    video_devices = [0, 2, 1]
    cap = None
    for device in video_devices:
        try:
            cap = cv2.VideoCapture(device)
            if cap.isOpened():
                print(f"✅ Successfully opened camera device {device}")
                break
            cap.release()
        except:
            continue
    if cap is None or not cap.isOpened():
        print("❌ Error: Could not access any webcam.")
        print("🚀 Please use the FastAPI endpoints instead:")
        print("   - POST /scan - Scan a product")
        print("   - POST /add - Add a new product")
        return

    print("""
    📷 Product Scanner - Controls:
    [s] - Scan current frame
    [a] - Add new product
    [q] - Quit
    """)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Camera error.")
            break

        filtered_frame = mask_humans(frame)

        try:
            cv2.imshow('Product Scanner', filtered_frame)
        except cv2.error:
            print("⚠️ Cannot display video (headless environment). Use FastAPI endpoints instead.")
            break

        key = cv2.waitKey(1) & 0xFF

        if key == ord('s') or key == 19:  # Ctrl+S
            result = scanner.recognize(frame)
            print(result)
            if result:
                print(f"\n✅ Product recognized: {result['label']} | Confidence: {result['confidence']:.2f}\n")
            else:
                print("\n❌ Product not recognized.")
                choice = input("Do you want to store this product? (y/n): ").strip().lower()
                if choice == 'y':
                    product_name = input("Enter product name (e.g., 'Ivermectina 500ml 3.15'): ").strip()
                    product_sku = input("Enter product SKU: ").strip()
                    if product_name and product_sku:
                        print(f"📸 Capture at least 5 views of '{product_name}' (SKU: {product_sku}).")
                        print("    Press [SPACE] to capture each photo. Press [ESC] to cancel.")
                        captured = 0
                        while captured < 5:
                            ret, frame = cap.read()
                            if not ret:
                                print("❌ Camera error.")
                                break
                            filtered_preview = mask_humans(frame)
                            display = filtered_preview.copy()
                            cv2.putText(display, f"Capture {captured + 1}/5 - [SPACE]=save  [ESC]=cancel",
                                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                            cv2.imshow('Capture Product Views', display)
                            key_inner = cv2.waitKey(1)
                            if key_inner == 32:  # SPACE
                                if scanner.add_product(product_name, product_sku, frame):
                                    print(f"✅ View {captured + 1} saved.")
                                    captured += 1
                            elif key_inner == 27:  # ESC
                                print("❌ Capture cancelled by user.")
                                captured = 0
                                break
                        cv2.destroyWindow('Capture Product Views')
                        if captured == 5:
                            print(f"🎉 Product '{product_name}' stored with 5 views.")
                            print("🔄 Ready for next product...")
                            time.sleep(1)
                    else:
                        print("❌ No product name/SKU provided. Skipping storage.")
                else:
                    print("ℹ️ Skipped storing.")

        elif key == ord('a'):
            product_name = input("Enter product name: ").strip()
            product_sku = input("Enter product SKU: ").strip()
            if product_name and product_sku:
                if scanner.add_product(product_name, product_sku, frame):
                    print(f"✅ Image added to product '{product_name}' (SKU: {product_sku})")
                else:
                    print("❌ Failed to add product.")
            else:
                print("❌ Both product name and SKU are required.")

        elif key == ord('q'):
            break

    cap.release()
    mp_face_detection.close()
    cv2.destroyAllWindows()
    print("👋 Scanner closed.")

def is_running_in_container():
    return os.path.exists('/.dockerenv')

if __name__ == "__main__":
    if is_running_in_container():
        print("🐳 Running in Docker container")
        print("ℹ️ GUI features are disabled in container environment")
        print("🚀 Use the FastAPI endpoints:")
        print("   - POST /scan - Scan a product")
        print("   - POST /add - Add a new product")
    else:
        main()
