import os
import json
import base64
import time
import re
from pathlib import Path
from groq import Groq

# --- Configuration ---
# NOTE: Since this key was shared in a chat, it is highly recommended to rotate it!
API_KEY = "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"

# Vision-capable models for the 2026 Groq environment
MODELS_LIST = ["meta-llama/llama-4-scout-17b-16e-instruct"]

# Set to True to wipe previous dataset_captions.json and start fresh
CLEAN_START = True

# Initialize Client
client = Groq(api_key=API_KEY)


def encode_image(image_path):
    """Encodes image to base64 for API transmission."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def parse_caption_content(raw_text):
    """
    Parses the model's output.
    Handles direct JSON objects (as requested in your prompt)
    and cleans up potential Markdown artifacts.
    """
    # Remove markdown code blocks if the model included them
    clean_text = raw_text.replace("```json", "").replace("```", "").strip()

    # 1. Attempt to parse as JSON
    try:
        data = json.loads(clean_text)
        if isinstance(data, dict):
            return {
                "primary": data.get("primary", "").strip(),
                "extended": data.get("extended", "").strip(),
            }
    except json.JSONDecodeError:
        pass

    # 2. Fallback: Regex if the model used markers but failed JSON formatting
    primary_match = re.search(
        r"primary[:\s]+(.*?)(?=extended:|$)", clean_text, re.IGNORECASE | re.DOTALL
    )
    extended_match = re.search(
        r"extended[:\s]+(.*)", clean_text, re.IGNORECASE | re.DOTALL
    )

    primary = primary_match.group(1).strip() if primary_match else clean_text
    extended = extended_match.group(1).strip() if extended_match else ""

    return {"primary": primary, "extended": extended}


def initialize_json_structure(dataset_path, prompt_text):
    """Initializes the clean metadata header."""
    return {
        "metadata": {
            "dataset_name": Path(dataset_path).name,
            "prompt_used": prompt_text,
            "models_evaluated": MODELS_LIST,
        },
        "images": {},
    }


def process_dataset(dataset_root, prompt_file, output_json):
    # 0. Handle Clean Start
    if CLEAN_START and os.path.exists(output_json):
        print(f"--- CLEAN START: Deleting '{output_json}' ---")
        os.remove(output_json)

    # 1. Load the prompt from markdown
    if not os.path.exists(prompt_file):
        print(f"Error: Prompt file '{prompt_file}' not found.")
        return

    with open(prompt_file, "r", encoding="utf-8") as f:
        system_prompt = f.read().strip()

    # 2. Setup Data Structure
    data = initialize_json_structure(dataset_root, system_prompt)
    results = data["images"]

    # 3. Find Images
    valid_extensions = (".jpg", ".jpeg", ".png", ".webp", ".JPEG")
    image_paths = []
    for root, _, files in os.walk(dataset_root):
        for file in files:
            if file.lower().endswith(valid_extensions):
                image_paths.append(os.path.join(root, file))

    total_images = len(image_paths)
    print(f"Found {total_images} images. Beginning processing...")

    # 4. Processing Loop
    for i, img_path in enumerate(image_paths):
        # Use forward slashes for cross-platform JSON consistency
        rel_path = os.path.relpath(img_path, dataset_root).replace("\\", "/")
        parent_folder = os.path.basename(os.path.dirname(img_path))

        print(f"\n[{i+1}/{total_images}] Processing: {rel_path}")

        if rel_path not in results:
            results[rel_path] = {
                "filename": os.path.basename(img_path),
                "class_label": parent_folder,
                "captions": {},
            }

        try:
            base64_image = encode_image(img_path)
        except Exception as e:
            print(f"   ! Error reading file: {e}")
            continue

        for model_id in MODELS_LIST:
            success = False
            retries = 0
            wait_time = 2

            while not success and retries < 5:
                try:
                    print(f"   > Requesting from {model_id}...")
                    completion = client.chat.completions.create(
                        model=model_id,
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": system_prompt},
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpeg;base64,{base64_image}"
                                        },
                                    },
                                ],
                            }
                        ],
                        temperature=0.1,  # Low temperature for consistent attribute extraction
                    )

                    raw_response = completion.choices[0].message.content
                    # Parse and save
                    results[rel_path]["captions"][model_id] = parse_caption_content(
                        raw_response
                    )

                    success = True
                    time.sleep(0.5)  # Minimum RPM buffer

                except Exception as e:
                    error_msg = str(e).lower()
                    if "429" in error_msg or "rate_limit" in error_msg:
                        print(f"   ! Rate limit hit. Backing off for {wait_time}s...")
                        time.sleep(wait_time)
                        wait_time *= 2
                        retries += 1
                    else:
                        print(f"   ! API error for {model_id}: {e}")
                        break

        # Incremental save to prevent data loss
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    print(f"\nDone! Final output saved to: {output_json}")


if __name__ == "__main__":
    # Ensure these paths are correct for your local machine
    DATASET_PATH = r"C:\Users\Digi Max\Desktop\AmirHossein\University\Shiraz University\Research\Projects\1. Content-Based Image Retrieval (CBIR)\Dataset\Corel-10K"
    PROMPT_MD = "prompt.md"
    OUTPUT_FILE = "dataset_captions.json"

    process_dataset(DATASET_PATH, PROMPT_MD, OUTPUT_FILE)
