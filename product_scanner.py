import os
import uuid
import cv2
import joblib
import psycopg2
import numpy as np
import time
from datetime import datetime
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.neighbors import NearestNeighbors
from dotenv import load_dotenv
from collections import Counter
from sklearn.decomposition import PCA
import mediapipe as mp

mp_face = mp.solutions.face_detection
mp_hands = mp.solutions.hands
# Initialize hands detection with specific dimensions
mp_hands_detection = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

load_dotenv()

USER = os.getenv('DB_USER')
PASSWORD = os.getenv('DB_PASSWORD')
DATABASE = os.getenv('DB_DATABASE')
HOST = os.getenv('DB_HOST')
PORT = os.getenv('DB_PORT')

db_url = f"postgresql://{USER}:{PASSWORD}@{HOST}:{PORT}/{DATABASE}"

def mask_humans(img):
    if img is None or img.size == 0:
        raise ValueError("❌ mask_humans: Received empty image.")

    h, w, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    masked_img = img.copy()
    
    # Normalize image for mediapipe processing
    mp_img = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=img_rgb
    )

    # Detect and blur face
    results = mp_face_detection.process(img_rgb)
    if results.detections:
        for det in results.detections:
            bbox = det.location_data.relative_bounding_box
            x1 = int(bbox.xmin * w)
            y1 = int(bbox.ymin * h)
            x2 = x1 + int(bbox.width * w)
            y2 = y1 + int(bbox.height * h)
            masked_img[y1:y2, x1:x2] = cv2.GaussianBlur(masked_img[y1:y2, x1:x2], (55, 55), 30)

    # Detect and blur hands with image dimensions context
    hand_results = mp_hands_detection.process(img_rgb)
    if hand_results and hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            # Convert normalized coordinates to pixel coordinates
            x_coords = [int(lm.x * w) for lm in hand_landmarks.landmark]
            y_coords = [int(lm.y * h) for lm in hand_landmarks.landmark]
            x1 = max(int(min(x_coords) * w) - 10, 0)
            y1 = max(int(min(y_coords) * h) - 10, 0)
            x2 = min(int(max(x_coords) * w) + 10, w)
            y2 = min(int(max(y_coords) * h) + 10, h)
            masked_img[y1:y2, x1:x2] = cv2.GaussianBlur(masked_img[y1:y2, x1:x2], (55, 55), 30)

    return masked_img



# Global mediapipe objects (initialized once)
mp_face_detection = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)


class ProductScannerSQL:
    def __init__(self, db_url, features_dir='features'):
        self.conn = psycopg2.connect(db_url)
        self.cursor = self.conn.cursor()
        self.feature_extractor = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
        self.knn = NearestNeighbors(n_neighbors=1, metric='cosine')
        self.X = []
        self.y = []
        self.is_trained = False
        self.features_dir = features_dir
        os.makedirs(features_dir, exist_ok=True)
        self.ensure_tables()

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

    def preprocess_frame(self, img_array):
        if img_array is None or img_array.size == 0:
            raise ValueError("❌ preprocess_frame: Received empty image.")

        if len(img_array.shape) == 2 or img_array.shape[2] == 1:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
        elif img_array.shape[2] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)

        masked = mask_humans(img_array)
        resized = cv2.resize(masked, (224, 224))
        arr = image.img_to_array(resized)
        arr = np.expand_dims(arr, axis=0)
        return preprocess_input(arr)

    def process_image(self, img):
        img = cv2.resize(img, (224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        return preprocess_input(img_array)

    def add_product(self, product_name, product_sku, img_array):
        try:
            # Create product or update if exists
            self.cursor.execute(
                "INSERT INTO products (name, sku) VALUES (%s, %s) ON CONFLICT (name) DO UPDATE SET name=EXCLUDED.name, sku=EXCLUDED.sku RETURNING id",
                (product_name, product_sku))
            product_id = self.cursor.fetchone()[0]

            # Process image and extract features
            processed = self.preprocess_frame(img_array)
            features = self.feature_extractor.predict(processed)[0]

            # Generate unique identifier
            vector_id = str(uuid.uuid4())
            
            # Save both feature vector and original image
            feature_path = os.path.join(self.features_dir, f"{vector_id}_feature.joblib")
            image_path = os.path.join(self.features_dir, f"{vector_id}_image.jpg")
            
            # Save files
            joblib.dump(features, feature_path)
            cv2.imwrite(image_path, img_array)

            # Store in database
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

    def train(self):
        self.cursor.execute("""
            SELECT pv.file_path, p.name
            FROM product_vectors pv
            JOIN products p ON p.id = pv.product_id
        """)
        rows = self.cursor.fetchall()
        self.X = []
        self.y = []
        class_counts = Counter()

        for file_path, product_name in rows:
            try:
                vec = joblib.load(file_path)

                # Skip uninformative vectors
                if np.std(vec) < 1e-3:
                    print(f"⚠️ Skipped low-variance vector from {file_path}")
                    continue

                # Optional: downsample overrepresented classes
                if class_counts[product_name] >= 100:  # You can adjust the cap
                    continue

                self.X.append(vec)
                self.y.append(product_name)
                class_counts[product_name] += 1

            except Exception as e:
                print(f"⚠️ Could not load vector from {file_path}: {e}")

        if self.X:
            # ✅ Optional: dimensionality reduction
            try:
                self.pca = PCA(n_components=128)  # reduce MobileNetV2 1280 → 128
                self.X = self.pca.fit_transform(self.X)
                self.knn.fit(self.X, self.y)
                self.use_pca = True
                print("✅ PCA applied.")
            except Exception as e:
                print(f"⚠️ PCA failed: {e}")
                self.knn.fit(self.X, self.y)
                self.use_pca = False
        else:
            print("⚠️ No vectors available for training.")
            return False

        self.is_trained = True
        print(f"🔁 Trained with {len(self.X)} vectors across {len(set(self.y))} products.")
        print(f"📊 Sample distribution: {class_counts}")
        return True

    def recognize(self, img_array):
        if not self.is_trained:
            print("🔁 Training required...")
            if not self.train():
                return None

        processed = self.preprocess_frame(img_array)
        features = self.feature_extractor.predict(processed)[0]

        if hasattr(self, 'use_pca') and self.use_pca:
            features = self.pca.transform([features])[0]

        distances, indices = self.knn.kneighbors([features])
        if distances[0][0] > 0.3:  # similarity threshold
            return None

        label = self.y[indices[0][0]]
        confidence = float(1 - distances[0][0])  # Convert numpy.float32 to Python float
        return {"label": label, "confidence": confidence}  # Return a dict instead of a tuple



def main():
    # Check if running in headless environment
    try:
        import os
        if os.environ.get('DISPLAY') is None and os.name != 'nt':
            print("⚠️ Running in headless environment. GUI features disabled.")
            print("🚀 Use the FastAPI endpoints instead:")
            print("   - POST /scan - Scan a product")
            print("   - POST /add - Add a new product")
            return
    except:
        pass

    scanner = ProductScannerSQL(db_url=db_url)  # 👈 Use the updated class

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not access the webcam.")
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
        
        # Try to show image, but handle headless environments gracefully
        try:
            cv2.imshow('Product Scanner', filtered_frame)
        except cv2.error as e:
            print("⚠️ Cannot display video (headless environment). Use FastAPI endpoints instead.")
            break


        key = cv2.waitKey(1) & 0xFF  # Mask high bits to ensure proper ASCII

        if key == ord('s') or key == 19:  # 19 is Ctrl+S
            result = scanner.recognize(frame)
            print(result)
            if result:
                print(f"\n✅ Product recognized: {result['label']} | Confidence: {result['confidence']:.2f}\n")
            else:
                print("\n❌ Product not recognized.")
                choice = input("Do you want to store this product? (y/n): ").strip().lower()
                if choice == 'y':
                    product_name = input("Enter product name (e.g., 'Victoria Secret 250ml'): ").strip()
                    product_sku = input("Enter product SKU: ").strip()
                    if product_name and product_sku:
                        print(f"📸 Please capture at least 5 different views of '{product_name}' (SKU: {product_sku}). Press [SPACE] to capture each photo. Press [ESC] to cancel.")
                        captured = 0

                        while captured < 5:
                            ret, frame = cap.read()
                            if not ret:
                                print("❌ Camera error.")
                                break

                            filtered_preview = mask_humans(frame)
                            display = filtered_preview.copy()
                            cv2.putText(display, f"Capture {captured + 1}/5 - Press [SPACE] to save, [ESC] to cancel",
                                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                            cv2.imshow('Capture Product Views', display)

                            key_inner = cv2.waitKey(1)
                            if key_inner == 32:  # SPACE key
                                if scanner.add_product(product_name, product_sku, frame):
                                    print(f"✅ View {captured + 1} saved.")
                                    captured += 1
                            elif key_inner == 27:  # ESC key
                                print("❌ Capture cancelled by user.")
                                captured = 0
                                break

                        cv2.destroyWindow('Capture Product Views')
                        if captured == 5:
                            print(f"🎉 Product '{product_name}' successfully stored with 5 views.")
                            print("🔄 Ready for next product...")
                            time.sleep(1)
                else:
                    print("❌ No product name provided. Skipping storage.") 
                    cap.release()            

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

    # Close mediapipe solutions for face and hands detection cleanup
    mp_face_detection.close()


    cv2.destroyAllWindows()
    print("👋 Scanner closed.")



if __name__ == "__main__":
    main()

