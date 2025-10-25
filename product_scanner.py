import os
import json
import hashlib
import time
import requests
import cv2
import base64
from pathlib import Path
from dotenv import load_dotenv
import warnings

# Setup
warnings.filterwarnings("ignore")
load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

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

def image_to_base64(image_path_or_array):
    """Convert image to base64 string for API"""
    if isinstance(image_path_or_array, str):
        # Read from file path
        with open(image_path_or_array, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    else:
        # Convert numpy array to base64
        success, encoded_image = cv2.imencode('.jpg', image_path_or_array)
        if success:
            return base64.b64encode(encoded_image).decode('utf-8')
        else:
            raise ValueError("Could not encode image to JPEG")

def extract_product_info(image_path_or_array):
    """
    Extract product info from image using DeepSeek API only - no local OCR
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
    
    # Not in cache - process with DeepSeek
    print("🔄 Processing with DeepSeek...")
    
    try:
        # Convert image to base64
        image_base64 = image_to_base64(image_path_or_array)
    except Exception as e:
        return {"error": f"Could not process image: {str(e)}"}

    # Enhanced prompt for comprehensive veterinary product extraction
    prompt = """Analyze this veterinary product image and extract ALL visible text and information as structured JSON:

REQUIRED FIELDS:
- product_name: The complete product name
- active_ingredient: Main active ingredient(s)
- concentration: Concentration percentage or amount
- formulation: Formulation type (injection, tablet, solution, etc.)
- volume: Volume in ml or quantity (tablets, etc.)
- manufacturer: Manufacturer or brand name
- presentation_details: Packaging details (bottle, box, blister, etc.)
- usage_instructions: Dosage and administration instructions
- warnings: Precautions and warnings

EXTRACTION RULES:
1. Extract ALL text you can read from the image
2. If a field cannot be determined, use null
3. Return ONLY valid JSON, no explanations
4. Preserve the exact text as it appears
5. Include all numbers, percentages, and measurements"""

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Try multiple API endpoints and formats
    endpoints = [
        {
            "url": "https://api.deepseek.com/chat/completions",
            "payload": {
                "model": "deepseek-chat",
                "messages": [
                    {
                        "role": "user", 
                        "content": f"{prompt}\n\nImage data: {image_base64}"
                    }
                ],
                "temperature": 0.1,
                "max_tokens": 2000
            }
        },
        {
            "url": "https://api.deepseek.com/chat/completions", 
            "payload": {
                "model": "deepseek-chat",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                        ]
                    }
                ],
                "temperature": 0.1,
                "max_tokens": 2000
            }
        }
    ]

    for endpoint in endpoints:
        try:
            response = requests.post(
                endpoint["url"],
                headers=headers,
                json=endpoint["payload"],
                timeout=60
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
                try:
                    parsed_data = json.loads(content)
                    
                    # Cache the result if successful
                    if 'error' not in parsed_data:
                        with open(cache_file, 'w', encoding='utf-8') as f:
                            json.dump(parsed_data, f, ensure_ascii=False)
                        parsed_data['cached'] = False
                    
                    return parsed_data
                    
                except json.JSONDecodeError as e:
                    print(f"JSON parse failed: {e}")
                    # If JSON parsing fails, try to extract JSON from text
                    json_match = re.search(r'\{.*\}', content, re.DOTALL)
                    if json_match:
                        try:
                            parsed_data = json.loads(json_match.group())
                            if 'error' not in parsed_data:
                                with open(cache_file, 'w', encoding='utf-8') as f:
                                    json.dump(parsed_data, f, ensure_ascii=False)
                                parsed_data['cached'] = False
                            return parsed_data
                        except:
                            continue
                    
            else:
                print(f"API error {response.status_code}: {response.text}")
                continue
                
        except requests.exceptions.Timeout:
            print("Request timeout, trying next endpoint...")
            continue
        except Exception as e:
            print(f"Request failed: {e}")
            continue

    return {"error": "All DeepSeek API attempts failed"}

# # Simple test function
# def test_extraction():
#     """Test the extraction with your image"""
#     result = extract_product_info("ivermectine2.jpeg")
#     print("Extraction result:")
#     print(json.dumps(result, indent=2, ensure_ascii=False))

# if __name__ == "__main__":
#     test_extraction()