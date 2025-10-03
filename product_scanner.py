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
    import faiss
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
        "You are a label reader. Extract ONLY what is printed on the product label.\n"
        "Return strict JSON with keys: percent (number or null), volume_ml (number or null), badge_color (string or null)."
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

        self.sim_threshold_global = 0.48

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
            SELECT pv.file_path, p.name, p.sku
            FROM product_vectors pv
            JOIN products p ON p.id = pv.product_id
        """)
        rows = self.cursor.fetchall()
        if not rows:
            print("⚠️ No vectors available for training.")
            self.is_trained = False
            return False

        payloads, labels, paths = [], [], []
        for file_path, name, sku in rows:
            try:
                payloads.append(joblib.load(file_path))
                labels.append(sku)
                paths.append(file_path)
            except Exception as e:
                print(f"⚠️ Could not load {file_path}: {e}")

        # Check if we have any payloads to train on
        if not payloads:
            print("⚠️ No product vectors found for training. Add some products first.")
            self.is_trained = False
            return False

        # PCA on CNN (fit on global vectors only)
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
        # Check if we have valid OCR text in any payload
        ocr_texts = [p["ocr_text"] for p in payloads]
        if any(text.strip() for text in ocr_texts):
            try:
                _ = self.vectorizer.fit(ocr_texts)
            except ValueError as e:
                print(f"⚠️ TF-IDF fitting failed: {e}. Using empty vectorizer.")
                self.vectorizer = TfidfVectorizer(max_features=1)
                _ = self.vectorizer.fit(["placeholder"])
        else:
            print("⚠️ No valid OCR text found. Using empty vectorizer.")
            self.vectorizer = TfidfVectorizer(max_features=1)
            _ = self.vectorizer.fit(["placeholder"])

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

    def _load_ref_image(self, feature_path):
        img_path = feature_path.replace("_feature.joblib", "_image.jpg")
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

    # ---------- Recognition ----------
    def recognize(self, img_array):
        if not self.is_trained and not self.train():
            return None

        # Build query vectors
        q_payload = self._extract_payload(img_array)
        q_coarse, q_fine = self._build_vectors(q_payload)
        q_coarse_n = q_coarse.astype(np.float32)
        q_coarse_n /= (np.linalg.norm(q_coarse_n) + 1e-8)
        qf = q_fine / (np.linalg.norm(q_fine) + 1e-8)
        q_tfidf = self.vectorizer.transform([q_payload["ocr_text"]]).astype(np.float32).toarray()[0]

        # ---- NEW: read numeric with DeepSeek on label ROI (fallback to OCR) ----
        ds_percent = None
        ds_volume  = None
        try:
            ds = deepseek_read_label(self._crop_for_ds(img_array))
            ds_percent = ds.get("percent")
            ds_volume  = ds.get("volume_ml")
        except Exception:
            pass

        qn = q_payload["numeric"]
        q_percent_ocr = qn[0] * 10.0 if qn[0] > 0 else None
        q_volume_ocr  = qn[2] * 1000.0 if qn[2] > 0 else None

        # Prefer DS when available; otherwise OCR
        q_percent = ds_percent if ds_percent is not None else q_percent_ocr
        q_volume  = ds_volume  if ds_volume  is not None else q_volume_ocr

        # (Optional) debug
        if q_percent is not None:
            print(f"🔎 Detected %: {q_percent}")
        if q_volume is not None:
            print(f"🔎 Detected ml: {q_volume}")

        # ANN retrieval (top-K)
        K = min(100, len(self.X))
        if _HAS_FAISS and self.faiss_index is not None:
            sims, idx = self.faiss_index.search(q_coarse_n[None, :], K)  # cosine via inner product
            cand_idx = idx[0].tolist()
            base_sims = sims[0].tolist()
        else:
            dist, idx = self.knn.kneighbors([q_coarse], n_neighbors=K)
            cand_idx = idx[0].tolist()
            base_sims = [1.0 - float(d) for d in dist[0].tolist()]

        # --- Generic multi-modal re-rank + numeric contradiction penalty ---
        PCT_TOL = 0.08    # 0.08% tolerance; tune 0.05–0.15
        ML_TOL  = 100.0   # 100 ml tolerance

        def score_candidate(i, base_sim):
            rf = self.fine_bank_norm[i]
            sim_visual = float(np.dot(qf, rf))

            rt = self.tfidf_bank[i]
            if np.linalg.norm(q_tfidf) > 0 and np.linalg.norm(rt) > 0:
                sim_ocr = float(np.dot(q_tfidf, rt) /
                                ((np.linalg.norm(q_tfidf) + 1e-8) * (np.linalg.norm(rt) + 1e-8)))
            else:
                sim_ocr = 0.0

            rn = joblib.load(self.file_paths[i])["numeric"]
            if np.linalg.norm(qn) > 0 and np.linalg.norm(rn) > 0:
                sim_num = float(np.dot(qn, rn) /
                                ((np.linalg.norm(qn) + 1e-8) * (np.linalg.norm(rn) + 1e-8)))
            else:
                sim_num = 0.0

            # ref numeric in human units
            r_percent = rn[0] * 10.0 if rn[0] > 0 else None
            r_volume  = rn[2] * 1000.0 if rn[2] > 0 else None

            # Strong, generic penalty if DS/OCR contradict ref
            penalty = 0.0
            if q_percent is not None and r_percent is not None and abs(q_percent - r_percent) > PCT_TOL:
                penalty += 0.60
            if q_volume  is not None and r_volume  is not None and abs(q_volume - r_volume) > ML_TOL:
                penalty += 0.15

            score = (0.50*base_sim + 0.30*sim_visual + 0.15*sim_ocr + 0.05*sim_num) - penalty
            return score

        prelim = []
        for base_sim, i in zip(base_sims, cand_idx):
            prelim.append((i, score_candidate(i, base_sim)))
        prelim.sort(key=lambda x: x[1], reverse=True)
        prelim = prelim[:20]

        # Geometric verification on label region (ORB)
        finals = []
        for i, s in prelim:
            r_img = self._load_ref_image(self.file_paths[i])
            geo = self._geo_score(img_array, r_img)
            finals.append((i, s + 0.20*geo))
        finals.sort(key=lambda x: x[1], reverse=True)

        best_idx = finals[0][0]
        best_score = finals[0][1]
        best_label = self.y[best_idx]

        # --- Final hard guard: if DS/OCR % contradicts chosen ref → reject ---
        if q_percent is not None:
            ref = joblib.load(self.file_paths[best_idx])
            rn  = ref["numeric"]
            r_percent = rn[0] * 10.0 if rn[0] > 0 else None
            if r_percent is not None and abs(q_percent - r_percent) > PCT_TOL:
                print(f"❌ Rejected by % mismatch: query {q_percent} vs ref {r_percent}")
                return None

        # Centroid-based acceptance (class-agnostic)
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

        filtered_frame = frame

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
                            filtered_preview = frame
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
