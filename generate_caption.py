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
from groq import Groq

# --- Configuration ---

API_KEYS = [
    "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",
    "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",
    "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",
    "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",
    "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",
    "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",
    "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",
    "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",
    "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",
    "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",
    "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",
    "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",
    "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",
    "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",
    "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",
    "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",
    "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",
    "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",
    "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",
    "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",
]

KEY_COOLDOWN_SECONDS = 60
MODELS_LIST = ["meta-llama/llama-4-scout-17b-16e-instruct"]


# ---------------------------------------------------------------------------
# Fix 1: KeyPool — no _cooling counter; derive availability purely from Queue
# ---------------------------------------------------------------------------


class KeyPool:
    """
    Thread-safe pool of Groq clients backed by a Queue.
    - No separate counter: availability is derived directly from the queue size
      and the number of keys currently in cooldown (tracked via a simple set).
    - Keys are NEVER lost: acquire() uses a try/finally-safe pattern and
      release_after_cooldown() guarantees requeue even if the timer thread crashes.
    """

    def __init__(self, api_keys: list, cooldown_seconds: int = 60):
        if not api_keys:
            raise ValueError("No API keys provided.")
        self.cooldown_seconds = cooldown_seconds
        self._total = len(api_keys)
        self._pool = Queue()
        self._cooling_count = 0  # how many keys are in cooldown
        self._lock = threading.Lock()  # protects _cooling_count only

        for key in api_keys:
            self._pool.put(Groq(api_key=key))

    def acquire(self, timeout: float = 5.0):
        """
        Block up to `timeout` seconds for an available client.
        Returns None on timeout (caller must retry or wait).
        """
        try:
            return self._pool.get(timeout=timeout)
        except Empty:
            return None

    def release(self, client):
        """Return a healthy client immediately — no counter to touch."""
        self._pool.put(client)

    def release_after_cooldown(self, client):
        """
        Mark key as cooling and requeue it after cooldown.
        The timer thread always requeues — even if it itself raises.
        """
        with self._lock:
            self._cooling_count += 1

        def _requeue():
            try:
                time.sleep(self.cooldown_seconds)
            finally:
                # guaranteed to run even if sleep is interrupted
                with self._lock:
                    self._cooling_count -= 1
                self._pool.put(client)

        t = threading.Thread(target=_requeue, daemon=True)
        t.start()

    def all_cooling(self) -> bool:
        """True when every key is in cooldown (queue is empty and will stay so)."""
        with self._lock:
            return self._cooling_count >= self._total


# ---------------------------------------------------------------------------
# Fix 4: Atomic JSON writes via temp-file + rename
# Fix 3: Single in-memory cache so workers don't hit disk on every check
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
        # Load once into memory; all subsequent reads use the cache
        with open(output_path, "r", encoding="utf-8") as f:
            self._data = json.load(f)

    def is_done(self, rel_path: str) -> bool:
        with self._lock:
            return is_image_fully_processed(self._data["images"].get(rel_path, {}))

    def needs_model(self, rel_path: str, model_id: str) -> bool:
        """Check under lock whether a specific model caption is missing."""
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
        """
        Under lock:
          1. Double-check the caption isn't already set (race-condition guard).
          2. Update the in-memory cache.
          3. Atomically flush to disk (temp file + os.replace).
        """
        with self._lock:
            # Double-check inside the lock
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

            # Atomic write: dump to a sibling temp file, then rename.
            # On Windows, antivirus/Explorer can briefly lock the target file
            # right after a write, causing os.replace() to raise WinError 5
            # (Access Denied). We retry a few times with backoff before giving up.
            dir_ = os.path.dirname(os.path.abspath(self.path))
            fd, tmp_path = tempfile.mkstemp(dir=dir_, suffix=".tmp")
            try:
                # Write and explicitly close the fd before attempting replace
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(self._data, f, indent=4)
                # fd is now closed; retry os.replace() on transient Windows locks
                for attempt in range(6):
                    try:
                        os.replace(tmp_path, self.path)
                        break  # success
                    except PermissionError:
                        if attempt == 5:
                            raise  # give up after 6 attempts (~3s total)
                        time.sleep(0.5 * (attempt + 1))  # 0.5s, 1s, 1.5s …
            except Exception:
                # Cache is already updated; only the disk write failed.
                # Clean up the orphaned temp file so it doesn't litter the dir.
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
# Fix 2 + 6: Worker — key never lost; failed images tracked and reported
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

    # Early exit if already fully processed
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
        # Skip models already done (resume-safe per model)
        if not json_mgr.needs_model(rel_path, model_id):
            continue

        model_success = False

        while not model_success:
            # --- Acquire a key (never blocks forever) ---
            client = None
            while client is None:
                client = key_pool.acquire(timeout=5.0)
                if client is None:
                    # acquire() timed out — check if all keys are cooling
                    if key_pool.all_cooling():
                        print(f"   ~ All keys cooling. Worker waiting... ({rel_path})")
                        time.sleep(2.0)
                    # else: a key is in the queue but another thread grabbed it first;
                    # just retry acquire immediately

            # --- Make the API call ---
            # client is now exclusively held by this thread
            try:
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
                    temperature=0.1,
                )

                caption = parse_caption_content(completion.choices[0].message.content)
                # Write first, THEN release key (so another thread can't grab
                # the same image and also try to write)
                json_mgr.write_caption(
                    rel_path, filename, class_label, model_id, caption
                )
                key_pool.release(client)  # healthy — back to pool immediately
                model_success = True
                time.sleep(0.2)

            except PermissionError as e:
                # Windows file-lock on os.replace() — even after retries.
                # The API call succeeded and caption is already in the in-memory
                # cache. Release the key and retry the loop; needs_model() will
                # short-circuit if the cache write did go through.
                print(f"   ! [WRITE ERROR] {rel_path}: {e} — retrying...")
                key_pool.release(client)
                time.sleep(1.0)

            except Exception as e:
                error_msg = str(e).lower()
                if "429" in error_msg or "rate_limit" in error_msg:
                    print(f"   ! Rate limit. Rotating key for {rel_path}...")
                    key_pool.release_after_cooldown(client)
                else:
                    # Non-rate-limit API error: return key, mark image as failed
                    print(f"   ! [API ERROR] {rel_path} / {model_id}: {e}")
                    key_pool.release(client)
                    overall_status = "api_error"
                    break

    _tick(progress, progress_lock, rel_path, overall_status)
    return rel_path, overall_status


def _tick(progress: dict, lock: threading.Lock, rel_path: str, status: str):
    """Update and print progress counter."""
    with lock:
        if status != "skipped":  # Fix 5: skipped images counted separately
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

    # Initialize or sync JSON file
    if not os.path.exists(output_json):
        print(f"No existing output found. Starting fresh: '{output_json}'")
        initialize_json(output_json, dataset_root, system_prompt)
    else:
        print(f"Loaded existing output: '{output_json}'")
        with open(output_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        data.setdefault("metadata", {})["models_evaluated"] = MODELS_LIST
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    json_mgr = JSONManager(output_json)

    # Collect all images
    valid_extensions = (".jpg", ".jpeg", ".png", ".webp")
    all_image_paths = []
    for root, _, files in os.walk(dataset_root):
        for file in files:
            if file.lower().endswith(valid_extensions):
                all_image_paths.append(os.path.join(root, file))

    # Build pending list — once, upfront, no duplicates
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

    # Fix 6: collect failed images and report summary at the end
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
    DATASET_PATH = r"C:\Users\Digi Max\Desktop\AmirHossein\University\Shiraz University\Research\Projects\1. Content-Based Image Retrieval (CBIR)\Dataset\Corel-10K"
    PROMPT_MD = "prompt.md"
    OUTPUT_FILE = "dataset_captions.json"

    process_dataset(DATASET_PATH, PROMPT_MD, OUTPUT_FILE)
