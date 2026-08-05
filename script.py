"""
extract_card.py

Extracts structured text + table data from engineering drawing title
block images using Qwen3-VL-4B-Instruct via the HuggingFace Inference
API (no local model download required).

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

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
PROVIDER = "featherless-ai"
MAX_DIMENSION = 1024  # resize images before sending -- smaller payloads
                       # process faster and time out less often

# ---- Schema-driven prompt -------------------------------------------------
EXTRACTION_PROMPT = """You are an information extraction engine. Look at this image of an engineering drawing title block.

Extract the data into ONLY valid JSON (no markdown fences, no explanation, no preamble) matching exactly this schema:

{
  "revision_table": [
    {
      "rev": "string",
      "cr": "string or null",
      "description": "string",
      "drn": "string or null",
      "chk": "string or null",
      "date": "string or null"
    }
  ],
  "reference_box": {
    "note": "string or null",
    "items": ["string"]
  },
  "confidential_info": "string or null",
  "part_name": "string or null",
  "metric_note": "string or null",
  "scale_note": "string or null",
  "part_no": "string or null",
  "customer_part_names": ["string"],
  "dwg_size": "string or null",
  "sheet": "string or null",
  "off": "string or null"
}

Rules:
- The top has a bordered TABLE with a header row (REV / CR / Revision Description / DRN / CHK / Date) and one or more data rows below it. Match each value to its column strictly by HORIZONTAL POSITION (which column it visually sits under), not by proximity to neighboring rows or cells. Do NOT include the header row itself as a data row, and do NOT let header label text (e.g. "DD-MM-YY") leak into a data row's values.
- Each data row of that table is ONE object in "revision_table". If there is only one data row, the array still has exactly one object.
- "reference_box" corresponds to the box labeled "REFERENCE BOX" — capture its header note text in "note" and each bulleted/listed line as a separate string in "items".
- "confidential_info" is the free text under the "CONFIDENTIAL AND OTHER INFORMATION" heading — capture it as a single string, preserving the meaning even if the box shows underlined or unusual text.
- "part_name" is the text under "PART NAME".
- "metric_note" is the text under "METRIC" (e.g. dimensions units note).
- "scale_note" is the text under "DO NOT SCALE DRAWING" (e.g. views/scale instructions).
- "part_no" is the value under "PART No.".
- "customer_part_names" is a list — capture each "CUSTOMER PART NAME" value as a separate string in order (there may be more than one row).
- "dwg_size", "sheet", "off" come from the bottom-right block (DWG SIZE, SHT, OFF).
- If a field is missing, empty, unreadable, or shows placeholder text like "XXX"/"XXXXXXX", capture it exactly as shown, character for character (do not interpret placeholders as null unless the field is truly blank, and do not round the number of repeated characters).
- TRANSCRIBE ALL TEXT EXACTLY AS WRITTEN, including any misspellings, typos, or grammatical errors visible in the image. Do NOT autocorrect spelling or grammar — reproduce the text verbatim even if it looks like an error.
- Output ONLY the JSON object, nothing else.
"""


def image_to_data_url(image_path: Path, max_dimension: int = MAX_DIMENSION) -> str:
    """Resize (if needed) and base64-encode an image as a data URL."""
    img = Image.open(image_path).convert("RGB")
    if max(img.size) > max_dimension:
        img.thumbnail((max_dimension, max_dimension), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


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
    parser = argparse.ArgumentParser(description="Extract structured data from drawing title block images via HF Inference API.")
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
            print(f"  ok: {data.get('part_name')} (part_no={data.get('part_no')})")
        except Exception as e:
            print(f"  FAILED on {img_path.name}: {e}")
            results.append({"_source_file": img_path.name, "_error": str(e)})

    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nWrote {len(results)} records to {args.output_file}")


if __name__ == "__main__":
    main()