import os
import json
import base64
import time
import re
import threading
import tempfile
import shutil
from pathlib import Path
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor, as_completed

import yaml
from groq import Groq

# ---------------------------------------------------------------------------
# Load configuration
# ---------------------------------------------------------------------------

CONFIG_FILE = "config.yaml"

with open(CONFIG_FILE, "r", encoding="utf-8") as _f:
    _cfg = yaml.safe_load(_f)

# Paths
DATASET_PATH = _cfg["paths"]["dataset"]
PROMPT_MD = _cfg["paths"]["prompt_file"]
OUTPUT_DIR = _cfg["paths"]["output_dir"]

# Derive output file path: <output_dir>/<dataset_name>_captions.json
_dataset_name = Path(DATASET_PATH).name
OUTPUT_FILE = os.path.join(OUTPUT_DIR, f"{_dataset_name}_captions.json")

# Models
MODELS_LIST: list[str] = _cfg["models"]

# API keys
API_KEYS: list[str] = _cfg["api_keys"]

# Rate limiting
KEY_COOLDOWN_SECONDS = _cfg["rate_limiting"]["key_cooldown_seconds"]
ACQUIRE_TIMEOUT = _cfg["rate_limiting"]["acquire_timeout_seconds"]
ALL_COOLING_SLEEP = _cfg["rate_limiting"]["all_cooling_sleep_seconds"]
POST_SUCCESS_SLEEP = _cfg["rate_limiting"]["post_success_sleep_seconds"]

# Inference
TEMPERATURE = _cfg["inference"]["temperature"]
MAX_TOKENS = _cfg["inference"]["max_tokens"]

# Images
VALID_EXTENSIONS: tuple[str, ...] = tuple(
    ext.lower() for ext in _cfg["images"]["valid_extensions"]
)


# ---------------------------------------------------------------------------
# KeyPool — thread-safe pool of Groq clients
# ---------------------------------------------------------------------------


class KeyPool:
    """
    Thread-safe pool of Groq clients backed by a Queue.
    - No separate counter: availability is derived directly from the queue size
      and the number of keys currently in cooldown (tracked via a simple set).
    - Keys are NEVER lost: acquire() uses a try/finally-safe pattern and
      release_after_cooldown() guarantees requeue even if the timer thread crashes.
    """

    def __init__(self, api_keys: list, cooldown_seconds: int = KEY_COOLDOWN_SECONDS):
        if not api_keys:
            raise ValueError("No API keys provided.")
        self.cooldown_seconds = cooldown_seconds
        self._total = len(api_keys)
        self._pool = Queue()
        self._cooling_count = 0
        self._lock = threading.Lock()

        for key in api_keys:
            self._pool.put(Groq(api_key=key))

    def acquire(self, timeout: float = ACQUIRE_TIMEOUT):
        try:
            return self._pool.get(timeout=timeout)
        except Empty:
            return None

    def release(self, client):
        self._pool.put(client)

    def release_after_cooldown(self, client, cooldown: float = None):
        wait = cooldown if cooldown is not None else self.cooldown_seconds
        with self._lock:
            self._cooling_count += 1

        def _requeue():
            try:
                time.sleep(wait)
            finally:
                with self._lock:
                    self._cooling_count -= 1
                self._pool.put(client)

        t = threading.Thread(target=_requeue, daemon=True)
        t.start()

    def all_cooling(self) -> bool:
        with self._lock:
            return self._cooling_count >= self._total


# ---------------------------------------------------------------------------
# JSONManager — atomic, thread-safe JSON persistence
# ---------------------------------------------------------------------------


class JSONManager:
    """
    Thread-safe JSON manager with:
    - In-memory cache (one lock, no repeated file reads per write)
    - Atomic disk writes (write to temp file, then os.replace) so a crash
      mid-write never corrupts the output file
    - Double-check before every write to prevent overwriting valid captions
    """

    def __init__(self, output_path: str):
        self.path = output_path
        self._lock = threading.Lock()
        with open(output_path, "r", encoding="utf-8") as f:
            self._data = json.load(f)

    def is_done(self, rel_path: str) -> bool:
        with self._lock:
            return is_image_fully_processed(self._data["images"].get(rel_path, {}))

    def needs_model(self, rel_path: str, model_id: str) -> bool:
        with self._lock:
            caption = (
                self._data["images"]
                .get(rel_path, {})
                .get("captions", {})
                .get(model_id, {})
            )
            return not caption.get("primary", "").strip()

    def write_caption(
        self,
        rel_path: str,
        filename: str,
        class_label: str,
        model_id: str,
        caption: dict,
    ):
        with self._lock:
            existing = (
                self._data["images"]
                .get(rel_path, {})
                .get("captions", {})
                .get(model_id, {})
            )
            if existing.get("primary", "").strip():
                return  # already written by another thread

            images = self._data["images"]
            if rel_path not in images:
                images[rel_path] = {
                    "filename": filename,
                    "class_label": class_label,
                    "captions": {},
                }
            images[rel_path]["captions"][model_id] = caption

            dir_ = os.path.dirname(os.path.abspath(self.path))
            fd, tmp_path = tempfile.mkstemp(dir=dir_, suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(self._data, f, indent=4)
                for attempt in range(6):
                    try:
                        os.replace(tmp_path, self.path)
                        break
                    except PermissionError:
                        if attempt == 5:
                            raise
                        time.sleep(0.5 * (attempt + 1))
            except Exception:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def parse_caption_content(raw_text: str) -> dict:
    clean_text = raw_text.replace("```json", "").replace("```", "").strip()
    try:
        data = json.loads(clean_text)
        if isinstance(data, dict):
            return {
                "primary": data.get("primary", "").strip(),
                "extended": data.get("extended", "").strip(),
            }
    except json.JSONDecodeError:
        pass

    # Fallback regex — logs a warning so malformed outputs can be audited
    print(f"   ! [PARSE WARNING] JSON decode failed; falling back to regex.")
    primary_match = re.search(
        r"primary[:\s]+(.*?)(?=extended:|$)", clean_text, re.IGNORECASE | re.DOTALL
    )
    extended_match = re.search(
        r"extended[:\s]+(.*)", clean_text, re.IGNORECASE | re.DOTALL
    )
    return {
        "primary": primary_match.group(1).strip() if primary_match else clean_text,
        "extended": extended_match.group(1).strip() if extended_match else "",
    }


def is_image_fully_processed(image_entry: dict) -> bool:
    captions = image_entry.get("captions", {})
    for model_id in MODELS_LIST:
        caption = captions.get(model_id)
        if not caption or not caption.get("primary", "").strip():
            return False
    return True


def initialize_json(output_path: str, dataset_path: str, prompt_text: str):
    data = {
        "metadata": {
            "dataset_name": Path(dataset_path).name,
            "prompt_used": prompt_text,
            "models_evaluated": MODELS_LIST,
        },
        "images": {},
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


def process_image(
    img_path: str,
    dataset_root: str,
    system_prompt: str,
    key_pool: KeyPool,
    json_mgr: JSONManager,
    progress: dict,
    progress_lock: threading.Lock,
) -> tuple:
    """
    Returns (rel_path, status) where status is one of:
      "skipped"    — already done at task start
      "ok"         — all models captioned successfully
      "read_error" — could not read/encode the image file
      "api_error"  — non-rate-limit API failure on at least one model
    """
    rel_path = os.path.relpath(img_path, dataset_root).replace("\\", "/")
    filename = os.path.basename(img_path)
    class_label = os.path.basename(os.path.dirname(img_path))

    if json_mgr.is_done(rel_path):
        _tick(progress, progress_lock, rel_path, "skipped")
        return rel_path, "skipped"

    try:
        base64_image = encode_image(img_path)
    except Exception as e:
        print(f"   ! [READ ERROR] {rel_path}: {e}")
        _tick(progress, progress_lock, rel_path, "read_error")
        return rel_path, "read_error"

    overall_status = "ok"

    for model_id in MODELS_LIST:
        if not json_mgr.needs_model(rel_path, model_id):
            continue

        model_success = False

        while not model_success:
            client = None
            while client is None:
                client = key_pool.acquire(timeout=ACQUIRE_TIMEOUT)
                if client is None:
                    if key_pool.all_cooling():
                        print(f"   ~ All keys cooling. Worker waiting... ({rel_path})")
                        time.sleep(ALL_COOLING_SLEEP)

            try:
                completion = client.chat.completions.create(
                    model=model_id,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}"
                                    },
                                },
                            ],
                        },
                    ],
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                )

                caption = parse_caption_content(completion.choices[0].message.content)
                json_mgr.write_caption(
                    rel_path, filename, class_label, model_id, caption
                )
                key_pool.release(client)
                model_success = True
                time.sleep(POST_SUCCESS_SLEEP)

            except PermissionError as e:
                print(f"   ! [WRITE ERROR] {rel_path}: {e} — retrying...")
                key_pool.release(client)
                time.sleep(1.0)

            except Exception as e:
                error_msg = str(e).lower()
                if "429" in error_msg or "rate_limit" in error_msg:
                    retry_after = None
                    resp = getattr(e, "response", None)
                    if resp is not None:
                        try:
                            retry_after = (
                                float(resp.headers.get("retry-after", 0)) or None
                            )
                        except (AttributeError, ValueError):
                            pass
                    wait_msg = (
                        f"{retry_after:.0f}s" if retry_after else "default cooldown"
                    )
                    print(
                        f"   ! Rate limit ({wait_msg}). Rotating key for {rel_path}..."
                    )
                    key_pool.release_after_cooldown(client, cooldown=retry_after)
                else:
                    print(f"   ! [API ERROR] {rel_path} / {model_id}: {e}")
                    key_pool.release(client)
                    overall_status = "api_error"
                    break

    _tick(progress, progress_lock, rel_path, overall_status)
    return rel_path, overall_status


def _tick(progress: dict, lock: threading.Lock, rel_path: str, status: str):
    with lock:
        if status != "skipped":
            progress["done"] += 1
        done = progress["done"]
        total = progress["total"]
        tag = {
            "ok": "✓",
            "skipped": "~",
            "read_error": "✗",
            "api_error": "✗",
        }.get(status, "?")
        print(f"   [{done}/{total}] {tag} {rel_path}  ({status})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def process_dataset(dataset_root: str, prompt_file: str, output_json: str):
    if not os.path.exists(prompt_file):
        print(f"Error: Prompt file '{prompt_file}' not found.")
        return

    with open(prompt_file, "r", encoding="utf-8") as f:
        system_prompt = f.read().strip()

    os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)

    if not os.path.exists(output_json):
        print(f"No existing output found. Starting fresh: '{output_json}'")
        initialize_json(output_json, dataset_root, system_prompt)
    else:
        print(f"Loaded existing output: '{output_json}'")
        with open(output_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Atomic sync of models list to existing file
        tmp_dir = os.path.dirname(os.path.abspath(output_json))
        data.setdefault("metadata", {})["models_evaluated"] = MODELS_LIST
        fd, tmp_path = tempfile.mkstemp(dir=tmp_dir, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        os.replace(tmp_path, output_json)

    json_mgr = JSONManager(output_json)

    all_image_paths = []
    for root, _, files in os.walk(dataset_root):
        for file in files:
            if file.lower().endswith(VALID_EXTENSIONS):
                all_image_paths.append(os.path.join(root, file))

    pending_paths = []
    skipped = 0
    seen = set()
    for img_path in all_image_paths:
        rel_path = os.path.relpath(img_path, dataset_root).replace("\\", "/")
        if rel_path in seen:
            continue
        seen.add(rel_path)
        if json_mgr.is_done(rel_path):
            skipped += 1
        else:
            pending_paths.append(img_path)

    print(f"\nFound {len(all_image_paths)} images total.")
    print(f"  Already processed : {skipped}")
    print(f"  Pending           : {len(pending_paths)}")
    print(f"  API keys / workers: {len(API_KEYS)}\n")

    if not pending_paths:
        print("All images are already captioned. Nothing to do.")
        return

    key_pool = KeyPool(API_KEYS, cooldown_seconds=KEY_COOLDOWN_SECONDS)
    progress = {"done": 0, "total": len(pending_paths)}
    progress_lock = threading.Lock()

    n_workers = len(API_KEYS)
    print(f"Starting parallel processing with {n_workers} workers...\n")

    failed = []

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(
                process_image,
                img_path,
                dataset_root,
                system_prompt,
                key_pool,
                json_mgr,
                progress,
                progress_lock,
            ): img_path
            for img_path in pending_paths
        }

        for future in as_completed(futures):
            try:
                rel_path, status = future.result()
                if status in ("read_error", "api_error"):
                    failed.append((rel_path, status))
            except Exception as e:
                img_path = futures[future]
                print(f"   [FATAL] Unhandled worker exception for {img_path}: {e}")
                failed.append((img_path, "fatal"))

    print(f"\nDone! Output saved to: {output_json}")

    if failed:
        print(f"\n  {len(failed)} image(s) could not be processed:")
        for rel_path, reason in failed:
            print(f"    ✗ {rel_path}  ({reason})")
    else:
        print("  All images processed successfully.")


if __name__ == "__main__":
    process_dataset(DATASET_PATH, PROMPT_MD, OUTPUT_FILE)
