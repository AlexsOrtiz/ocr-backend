import base64
import json
import os
from typing import Any

import cv2
import numpy as np

from app.models.schemas import OCRResult

VISION_PROMPT = """This image shows a power tool or product label. Extract and return ONLY a JSON object with: sku, brand, model, serial (use null if not visible). Nothing else."""


def _image_to_base64(image: np.ndarray) -> str:
    _, buf = cv2.imencode(".png", image)
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def extract_metadata_with_vision(
    image: np.ndarray,
    filename: str,
    *,
    api_key: str | None = None,
) -> OCRResult | None:
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key or not key.strip():
        return None

    try:
        import openai
    except ImportError:
        return None

    b64 = _image_to_base64(image)
    client = openai.OpenAI(api_key=key)

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": VISION_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"},
                        },
                    ],
                }
            ],
            max_tokens=300,
        )
    except Exception:
        return None

    text = (response.choices[0].message.content or "").strip()
    if not text:
        return None

    if "```" in text:
        start = text.find("```")
        if start >= 0:
            start = text.find("\n", start) + 1
            end = text.find("```", start)
            if end > start:
                text = text[start:end]
    try:
        data: dict[str, Any] = json.loads(text)
    except json.JSONDecodeError:
        return None

    sku = data.get("sku")
    brand = data.get("brand")
    model = data.get("model")
    serial = data.get("serial")
    if sku is not None:
        sku = str(sku).strip() or None
    if brand is not None:
        brand = str(brand).strip() or None
    if model is not None:
        model = str(model).strip() or None
    if serial is not None:
        serial = str(serial).strip() or None

    return OCRResult(
        filename=filename,
        angle=0.0,
        raw_text=[],
        sku=sku,
        brand=brand,
        model=model,
        serial=serial,
        tool_type=None,
        sbd_brand=False,
        confidence=0.9,
    )
