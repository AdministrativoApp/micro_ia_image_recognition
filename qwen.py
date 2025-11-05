import os
import base64
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Reuse the same working client config
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
)

def image_to_base64_data_url(image_path):
    """Convert local image to data URL for OpenAI-compatible API."""
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:image/jpeg;base64,{b64}"

def analyze_with_qwen_vl_modelstudio(image_path):
    """
    Strict veterinary label reader using qwen3-vl-plus via OpenAI-compatible API.
    Uses the same working credentials as your test script.
    """
    image_url = image_to_base64_data_url(image_path)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": image_url}
                },
                {
                    "type": "text",
                    "text": (
                        "You are a strict veterinary label reader. Extract ONLY visible text. "
                        "Return valid JSON with: product_name, active_ingredient, concentration, manufacturer, all_visible_text. "
                        "Rules: DO NOT HALLUCINATE. DO NOT infer. If not visible, use null. "
                        "Return concentration exactly as shown (e.g., '1%'). "
                        "Example: {\"concentration\": \"1%\", \"all_visible_text\": \"INDHELMIN IVERMECTINA 1% INDICUS\", \"Other important fields\": \"...\"}."
                    )
                }
            ]
        }
    ]

    try:
        print("🔍 Analyzing image with qwen3-vl-plus ...")
        completion = client.chat.completions.create(
            model="qwen3-vl-plus",
            messages=messages,
            temperature=0.01,
            max_tokens=300,
            stream=False
        )

        raw_content = completion.choices[0].message.content.strip()
        # print(f"📄 Raw Response: {raw_content}")

        # Attempt to parse JSON
        try:
            parsed = json.loads(raw_content)
            return parsed
        except json.JSONDecodeError:
            return {
                "error": "Failed to parse JSON",
                "raw_response": raw_content
            }

    except Exception as e:
        return {
            "error": f"API call failed: {str(e)}"
        }

# Test
if __name__ == "__main__":
    result = analyze_with_qwen_vl_modelstudio("Estrepen100.jpeg")
    print("\n📊 Final Result:")
    print(json.dumps(result, indent=2, ensure_ascii=False))