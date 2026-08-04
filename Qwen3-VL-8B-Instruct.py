"""
test_qwen3_vl_8b_featherless.py

Standalone test: Qwen3-VL-8B-Instruct via Featherless AI
(confirmed "Preferred" provider for this model on your HF account).

Usage:
    python test_qwen3_vl_8b_featherless.py --image input/text_extraction.png
    python test_qwen3_vl_8b_featherless.py            # auto-picks first image in input/

Requirements:
    pip install huggingface_hub pillow python-dotenv
"""

import argparse
import base64
import io
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from PIL import Image

load_dotenv()

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
PROVIDER = "featherless-ai"
MAX_DIMENSION = 1024


def find_default_image(input_dir: str = "input") -> Path | None:
    folder = Path(input_dir)
    if not folder.exists():
        return None
    for p in sorted(folder.iterdir()):
        if p.suffix.lower() in (".png", ".jpg", ".jpeg"):
            return p
    return None


def image_to_data_url(image_path: Path, max_dimension: int = MAX_DIMENSION) -> str:
    img = Image.open(image_path).convert("RGB")
    if max(img.size) > max_dimension:
        img.thumbnail((max_dimension, max_dimension), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def call_with_retry(client, max_retries: int = 3, backoff_seconds: int = 5, **kwargs):
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as e:
            last_error = e
            err_str = str(e)
            transient = any(code in err_str for code in ("502", "503", "504", "Timeout", "timeout"))
            if transient and attempt < max_retries:
                print(f"    attempt {attempt} failed ({e}); retrying in {backoff_seconds}s ...")
                time.sleep(backoff_seconds)
                continue
            raise
    raise last_error


def main():
    parser = argparse.ArgumentParser(description="Test Qwen3-VL-8B-Instruct via Featherless AI.")
    parser.add_argument("--image", default=None, help="Image path. Defaults to first image in input/.")
    parser.add_argument("--skip_text_test", action="store_true", help="Skip the text-only sanity check.")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit(
            "No HF token found. Add HF_TOKEN=hf_xxx to your .env file. "
            "Get one at https://huggingface.co/settings/tokens"
        )

    client = InferenceClient(api_key=token, provider=PROVIDER)

    # --- Text-only sanity check ---------------------------------------------
    if not args.skip_text_test:
        completion = call_with_retry(
            client,
            model=MODEL_ID,
            messages=[{"role": "user", "content": "What is machine learning?"}],
            max_tokens=100,
        )
        print("Text-only response:")
        print(completion.choices[0].message.content)
        print()

    # --- Vision test ---------------------------------------------------------
    image_path = Path(args.image) if args.image else find_default_image()
    if image_path is None or not image_path.exists():
        raise SystemExit(f"Image not found: {image_path}. Pass --image <path>.")

    print(f"Using image: {image_path}")
    data_url = image_to_data_url(image_path)

    completion = call_with_retry(
        client,
        model=MODEL_ID,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image in one sentence."},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        max_tokens=200,
    )
    print("Vision response:")
    print(completion.choices[0].message.content)


if __name__ == "__main__":
    main()