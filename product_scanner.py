import os
import json
import base64
import requests
import cv2
import numpy as np
from dotenv import load_dotenv
import easyocr
import warnings

# Suppress PyTorch warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.utils.data.dataloader")
warnings.filterwarnings("ignore", message="Neither CUDA nor MPS are available*")
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Disable GPU usage to avoid warnings

# Load environment variables
load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# Initialize OCR reader
reader = easyocr.Reader(['en', 'es'])  # English and Spanish support


def extract_product_info(image_path_or_array):
    """
    Extract product info from image using OCR + DeepSeek
    """

    # Prepare image
    if isinstance(image_path_or_array, str):
        # Read from file path
        img = cv2.imread(image_path_or_array)
    else:
        # Assume it's already a numpy array
        img = image_path_or_array

    if img is None:
        return {"error": "Could not load image"}

    # Step 1: Extract text using OCR
    print("Running OCR...")
    try:
        results = reader.readtext(img)
        extracted_text = " ".join([text for (_, text, _) in results])
        # print(f"OCR Text: {extracted_text[:200]}...")  # Show first 200 chars
    except Exception as e:
        print(f"OCR failed: {e}")
        return {"error": f"OCR failed: {str(e)}"}

    if not extracted_text.strip():
        return {"error": "No text found in image"}

    # Step 2: Send extracted text to DeepSeek for parsing
    prompt = f"""
    Analyze this veterinary product text extracted from OCR and extract ALL available information as a JSON object:

    Text: "{extracted_text}"

    Return a JSON object with ALL information you can extract from the text:

    CRITICAL Rules:
    - Return ONLY a valid JSON object, no other text, no explanations, no markdown
    - Fix OCR errors: "mu" = "ml", "lugectalsle" = "inyectable", "3,15" = 3.15, etc.
    - Extract EVERY piece of information you can identify
    - Use null for missing information, not empty strings
    - Be thorough - veterinary product labels contain critical information
    - If uncertain about a field, use null rather than guessing
    - Numbers should use dots as decimals (3.15 not 3,15)
    - If the value is null, do not include it in the JSON
    - Recognize details like dots, colors, shapes, etc, small details that differ from other products
    - JSON format example:
    {{
        "product_name": "Ivermectina 1%",
        "active_ingredient": "Ivermectina",
        "concentration": "1%",
        "formulation": "Oral solution",
        "volume": "100 ml",
        "indications": "For the treatment of parasitic infections in cattle",
        "dosage": "0.2 mg/kg body weight",
        "administration_route": "Oral",
        "warnings": "Do not use in animals intended for human consumption",
        "storage_conditions": "Store below 30°C",
        "manufacturer": "VetPharma Inc.",
        "batch_number": "B12345",
        "expiration_date": "2024-12-31",
        "lot_number": "L67890",
        "barcode": "0123456789012",
        "color": "Yellow",
        "shape": "Bottle with dropper",
        "other_details": "Shake well before use"
    }}
    """

    # API request
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "deepseek-chat",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.1,
        "max_tokens": 1000
    }

    try:
        response = requests.post(
            "https://api.deepseek.com/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )

        print(f"API Status: {response.status_code}")

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
            return parsed_data
        else:
            print(f"API Error: {response.text}")
            return {"error": f"API returned status {response.status_code}"}

    except Exception as e:
        print(f"Exception: {str(e)}")
        return {"error": str(e)}

# Test function
def test_extraction():
    """Test the extraction with your image"""
    # Test with your image file
    result = extract_product_info("ivermectine2.jpeg")
    print("Extraction result:")
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    test_extraction()