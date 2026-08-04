"""
test_featherless_7b.py

Minimal standalone test: Qwen2.5-VL-7B-Instruct via Featherless AI,
using huggingface_hub's InferenceClient (matches the "huggingface_hub"
tab in the model's Inference Providers panel).

Usage:
    python test_featherless_7b.py --image input/text_extraction.png
    python test_featherless_7b.py --image path/to/any_image.jpg
    python test_featherless_7b.py            # auto-picks the first image in input/
"""

import argparse
import base64
import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()


def find_default_image(input_dir: str = "input") -> Path | None:
    """Fallback: grab the first image found in the input/ folder."""
    folder = Path(input_dir)
    if not folder.exists():
        return None
    for p in sorted(folder.iterdir()):
        if p.suffix.lower() in (".png", ".jpg", ".jpeg"):
            return p
    return None


def main():
    parser = argparse.ArgumentParser(description="Test Qwen2.5-VL-7B-Instruct via Featherless AI on a single image.")
    parser.add_argument(
        "--image", default=None,
        help="Path to the image to test. If omitted, uses the first image found in input/."
    )
    parser.add_argument(
        "--skip_text_test", action="store_true",
        help="Skip the plain text-only sanity check and only run the vision test."
    )
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit(
            "No HF token found. Add HF_TOKEN=hf_xxx to your .env file or export it. "
            "Get one at https://huggingface.co/settings/tokens"
        )

    client = InferenceClient(
        provider="featherless-ai",
        api_key=token,
    )

    # --- Text-only sanity check (matches the panel's default snippet) ------
    if not args.skip_text_test:
        completion = client.chat.completions.create(
            model="Qwen/Qwen2.5-VL-7B-Instruct",
            messages=[{"role": "user", "content": "What is machine learning?"}],
            max_tokens=100,
        )
        print("Text-only response:")
        print(completion.choices[0].message.content)
        print()

    # --- Vision test: describe the given image ------------------------------
    image_path = Path(args.image) if args.image else find_default_image()

    if image_path is None:
        print("No image provided and none found in input/. Pass --image <path>.")
        return

    if not image_path.exists():
        print(f"Image not found: {image_path}")
        return

    ext = image_path.suffix.lower().lstrip(".")
    mime = "jpeg" if ext in ("jpg", "jpeg") else ext
    b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    data_url = f"data:image/{mime};base64,{b64}"

    print(f"Using image: {image_path}")
    completion = client.chat.completions.create(
        model="Qwen/Qwen2.5-VL-7B-Instruct",
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