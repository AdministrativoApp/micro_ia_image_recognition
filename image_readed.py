import cv2
import numpy as np
import os
import json
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.neighbors import NearestNeighbors

class ProductScanner:
    def __init__(self, db_file='product_db.json'):
        self.db_file = db_file
        self.feature_extractor = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
        self.database = {}  # Format: {'Product A': [feature_vector1, feature_vector2, ...]}
        self.knn = NearestNeighbors(n_neighbors=1, metric='cosine')
        self.X = []
        self.y = []
        self.is_trained = False
        self.initialize_database()

    def initialize_database(self):
        if not os.path.exists(self.db_file):
            print(f"No database found at {self.db_file}, creating new one")
            self.save_database()
        else:
            self.load_database()

    def process_image(self, img):
        img = cv2.resize(img, (224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        return preprocess_input(img_array)

    def add_product(self, product_name, img_path=None, img_array=None):
        if img_path:
            img = cv2.imread(img_path)
            if img is None:
                print(f"Error reading image from {img_path}")
                return False
        elif img_array is not None:
            img = img_array
        else:
            print("No image provided!")
            return False

        try:
            processed = self.process_image(img)
            features = self.feature_extractor.predict(processed)[0].tolist()  # convert to list for JSON

            if product_name not in self.database:
                self.database[product_name] = []
            self.database[product_name].append(features)
            self.is_trained = False
            self.save_database()
            print(f"✅ Added image to product: {product_name}")
            return True
        except Exception as e:
            print(f"Error processing image: {e}")
            return False

    def train(self):
        if not self.database:
            print("⚠️ No products in database.")
            return False

        try:
            self.X = []
            self.y = []
            for name, features_list in self.database.items():
                for features in features_list:
                    self.X.append(features)
                    self.y.append(name)

            self.knn.fit(self.X, self.y)
            self.is_trained = True
            print(f"🔁 Trained with {len(self.X)} images across {len(self.database)} products.")
            return True
        except Exception as e:
            print(f"Training failed: {e}")
            return False

    def recognize(self, img):
        if not self.is_trained:
            print("🔁 Training required...")
            if not self.train():
                return None

        try:
            processed = self.process_image(img)
            features = self.feature_extractor.predict(processed)[0]

            distances, indices = self.knn.kneighbors([features])

            if distances[0][0] > 0.3:  # similarity threshold
                return None

            predicted_label = self.y[indices[0][0]]
            confidence = 1 - distances[0][0]
            return predicted_label, confidence
        except Exception as e:
            print(f"Recognition error: {e}")
            return None

    def save_database(self):
        try:
            with open(self.db_file, 'w') as f:
                json.dump(self.database, f)
            return True
        except Exception as e:
            print(f"Error saving database: {e}")
            return False

    def load_database(self):
        try:
            with open(self.db_file, 'r') as f:
                self.database = json.load(f)
            print(f"📦 Loaded {sum(len(v) for v in self.database.values())} images across {len(self.database)} products.")
            self.train()
            return True
        except Exception as e:
            print(f"Error loading database: {e}")
            self.database = {}
            return False

def main():
    scanner = ProductScanner()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam.")
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
            print("Camera error")
            break

        cv2.imshow('Product Scanner', frame)
        key = cv2.waitKey(1)

        if key == ord('s'):
            result = scanner.recognize(frame)
            if result:
                name, confidence = result
                print(f"\n✅ Producto reconocido: {name} Confidencia: {confidence:.2f})\n")
            else:
                print("\n❌ Producto no reconocido.")
                choice = input("¿Quieres almacenar este producto? (y/n): ").strip().lower()
                if choice == 'y':
                    product_name = input("Enter product name (e.g., 'Victoria Secret 250ml'): ").strip()
                    if scanner.add_product(product_name, img_array=frame):
                        print(f"✅ Added image to '{product_name}'")
                    else:
                        print("❌ Failed to store image.")

        elif key == ord('a'):
            product_name = input("Enter product name: ").strip()
            if product_name:
                if scanner.add_product(product_name, img_array=frame):
                    print(f"✅ Added image to '{product_name}'")
                else:
                    print("❌ Failed to add product.")

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("👋 Scanner closed.")

if __name__ == "__main__":
    main()
