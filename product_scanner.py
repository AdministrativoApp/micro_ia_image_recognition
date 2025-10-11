import os
import json
import hashlib
import time
import requests
import cv2
from pathlib import Path
from dotenv import load_dotenv
import easyocr
import warnings

# Setup
warnings.filterwarnings("ignore")
load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
reader = easyocr.Reader(['es'])

# Cache setup
CACHE_DIR = "./scan_cache"
Path(CACHE_DIR).mkdir(exist_ok=True)

def get_image_signature(image_path_or_array):
    """Generate unique signature for image"""
    if isinstance(image_path_or_array, str):
        # For file paths, use file metadata
        stat = os.stat(image_path_or_array)
        return hashlib.md5(f"{image_path_or_array}_{stat.st_mtime}_{stat.st_size}".encode()).hexdigest()
    else:
        # For image arrays, use pixel data
        return hashlib.md5(image_path_or_array.tobytes()).hexdigest()

def extract_product_info(image_path_or_array):
    """
    Extract product info from image using OCR + DeepSeek WITH CACHING
    """
    cache_key = get_image_signature(image_path_or_array)
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")
    
    # Check cache first
    if os.path.exists(cache_file):
        if time.time() - os.path.getmtime(cache_file) < 3600:  # 1 hour cache
            print("📦 Loading from cache...")
            with open(cache_file, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
                cached_data['cached'] = True
                return cached_data
        else:
            # Cache expired
            os.remove(cache_file)
    
    # Not in cache - process normally
    print("🔄 Processing fresh scan...")
    
    # Prepare image
    if isinstance(image_path_or_array, str):
        img = cv2.imread(image_path_or_array)
    else:
        img = image_path_or_array

    if img is None:
        return {"error": "Could not load image"}

    # Step 1: Extract text using OCR
    try:
        results = reader.readtext(img)
        extracted_text = " ".join([text for (_, text, _) in results])
    except Exception as e:
        print(f"OCR failed: {e}")
        return {"error": f"OCR failed: {str(e)}"}

    if not extracted_text.strip():
        return {"error": "No text found in image"}

    # Step 2: Send extracted text to DeepSeek for parsing
    short_prompt = f"""Extract veterinary product info as JSON from: {extracted_text}
    Return only: product_name, active_ingredient, concentration, formulation, volume, manufacturer
    Use null for missing fields. JSON only, no explanations."""
    
    headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}"}
    payload = {
        "model": "deepseek-chat", 
        "messages": [{"role": "user", "content": short_prompt}],
        "temperature": 0.1,
        "max_tokens": 400
    }

    try:
        response = requests.post(
            "https://api.deepseek.com/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            content = result["choices"][0]["message"]["content"]

            # Clean the response
            content = content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            # Parse JSON
            parsed_data = json.loads(content)
            
            # Cache the result if successful
            if 'error' not in parsed_data:
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(parsed_data, f, ensure_ascii=False)
                parsed_data['cached'] = False
            
            return parsed_data
        else:
            return {"error": f"API returned status {response.status_code}"}

    except Exception as e:
        return {"error": str(e)}

# # Test function
# def test_extraction():
#     """Test the extraction with your image"""
#     result = extract_product_info("ivermectine2.jpeg")
#     print("Extraction result:")
#     print(json.dumps(result, indent=2, ensure_ascii=False))

# if __name__ == "__main__":
#     test_extraction()