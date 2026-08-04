"""
extract_card.py

Extracts structured text + table data from student record card images
using Qwen3-VL-30B-A3B-Instruct via the HuggingFace Inference API
(no local model download required).

Usage:
    # Put HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx in a .env file in this directory
    python extract_card.py --input_dir input --output_file output.json

Requirements:
    pip install huggingface_hub pillow python-dotenv
"""

import argparse
import base64
import io
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from PIL import Image

load_dotenv()  # reads .env in the current working directory into os.environ

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
PROVIDER = "featherless-ai"
MAX_DIMENSION = 1024  # resize images before sending -- smaller payloads
                       # process faster and time out less often

# ---- Schema-driven prompt -------------------------------------------------
EXTRACTION_PROMPT = """You are an information extraction engine. Look at this image of a student record card.

Extract the data into ONLY valid JSON (no markdown fences, no explanation, no preamble) matching exactly this schema:

{
  "name": "string",
  "roll_no": "number or null",
  "address": "string or null",
  "backlogs": "number or null",
  "academic_record": [
    {"year": "number", "cgpa": "number or null", "percentage": "number or null"}
  ]
}

Rules:
- The top of the card has a bordered TABLE with columns like Year / CGPA / Percentage. Each row of that table is ONE object in the "academic_record" array. If there is only one row, the array still has exactly one object.
- Percentage should be a plain number (e.g. 90 for "90%"), not a string with a % sign.
- The rest of the card has label:value pairs (Name, Roll No, Address, Backlogs) — map these directly to the matching schema fields.
- If a field is missing or unreadable, use null. Do not guess.
- Output ONLY the JSON object, nothing else.
"""


def image_to_data_url(image_path: Path) -> str:
    ext = image_path.suffix.lower().lstrip(".")
    mime = "jpeg" if ext in ("jpg", "jpeg") else ext
    b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:image/{mime};base64,{b64}"


def extract_json_from_text(text: str) -> dict:
    """Strip markdown fences / stray text defensively before parsing."""
    cleaned = re.sub(r"```json|```", "", text).strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in model output:\n{text}")
    return json.loads(match.group(0))


def run_extraction(client: InferenceClient, image_path: Path) -> dict:
    data_url = image_to_data_url(image_path)

    completion = client.chat.completions.create(
        model=MODEL_ID,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": EXTRACTION_PROMPT},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        max_tokens=1024,
    )

    output_text = completion.choices[0].message.content
    return extract_json_from_text(output_text)


def main():
    parser = argparse.ArgumentParser(description="Extract structured data from student card images via HF Inference API.")
    parser.add_argument("--input_dir", default="input", help="Folder containing card images")
    parser.add_argument("--output_file", default="output.json", help="Where to write extracted JSON array")
    parser.add_argument("--hf_token", default=None, help="HF access token (or set HF_TOKEN env var)")
    args = parser.parse_args()

    token = args.hf_token or os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit(
            "No HF token found. Add HF_TOKEN=hf_xxx to your .env file, "
            "set the HF_TOKEN env var, or pass --hf_token. "
            "Get one at https://huggingface.co/settings/tokens"
        )

    input_dir = Path(args.input_dir)
    image_paths = sorted(
        [p for p in input_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")]
    )

    if not image_paths:
        print(f"No images found in {input_dir}/")
        return

    client = InferenceClient(api_key=token, provider=PROVIDER)

    results = []
    for img_path in image_paths:
        print(f"Processing {img_path.name} ...")
        try:
            data = run_extraction(client, img_path)
            data["_source_file"] = img_path.name
            data["_model_used"] = MODEL_ID
            data["_provider_used"] = PROVIDER
            results.append(data)
            print(f"  ok: {data.get('name')} (roll_no={data.get('roll_no')})")
        except Exception as e:
            print(f"  FAILED on {img_path.name}: {e}")
            results.append({"_source_file": img_path.name, "_error": str(e)})

    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nWrote {len(results)} records to {args.output_file}")


if __name__ == "__main__":
    main()