import os
import base64
import mimetypes
from groq import Groq

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

def encode_image(image_path: str):
    """Return (base64_string, mime_type) for the given image_path."""
    if not image_path or not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        # Safe fallback
        mime_type = "image/jpeg"
    return encoded, mime_type

def analyze_image_with_query(query: str, model: str, image_path: str, api_key: str | None = None) -> str:
    """Send a text+image prompt to Groq vision model and return the text response."""
    encoded_image, mime_type = encode_image(image_path)
    client = Groq(api_key=api_key or GROQ_API_KEY)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": query},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{encoded_image}"},
                },
            ],
        }
    ]
    chat_completion = client.chat.completions.create(messages=messages, model=model)
    return chat_completion.choices[0].message.content or ""
