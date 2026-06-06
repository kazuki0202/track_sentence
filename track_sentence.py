"""Generate English description sentences for tracks using a Qwen LLM.

Usage:
    python track_sentence.py                                        # defaults
    python track_sentence.py --model Qwen/Qwen2.5-3B-Instruct
    python track_sentence.py --model Qwen/Qwen2.5-7B-Instruct --batch-size 64
    python track_sentence.py --backend transformers --resume

The script:
1. Loads track metadata from the HuggingFace dataset
2. Filters out subjective / opinion / listening-medium tags
3. Batches track metadata into LLM prompts
4. Generates a concise English description per track
5. Outputs a CSV: track_id, sentence
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

# =========================================================================
# Tag filtering
# =========================================================================

SUBJECTIVE_EXACT: set[str] = {
    # --- ratings ---
    *(f"{n} of 10 stars" for n in range(1, 11)),
    "1 star", "2 stars", "3 stars", "4 stars", "5 stars",
    # --- opinion / preference ---
    "favorites", "favorite", "favourites", "favourite",
    "fav", "favs", "faves", "fave",
    "favorite songs", "favourite songs", "favorite song", "favourite song",
    "favorite tracks", "favorite artists", "favorite bands",
    "personal favourites", "all time favourites", "all time favorites",
    "my favorites", "my favorite", "my music", "my faves",
    "awesome", "amazing", "beautiful", "cool", "epic", "great", "good", "nice",
    "brilliant", "genius", "best", "perfect", "perfection", "lovely", "pretty",
    "cute", "hot", "wow", "fantastic", "incredible", "wonderful", "superb",
    "masterpiece", "legendary", "legend",
    "fucking awesome", "badass", "kick ass",
    "good stuff", "good shit", "good song", "good music",
    "great song", "great songs", "great lyrics", "great lyricists",
    "best song ever", "best songs ever", "the best",
    "top quality", "top", "top 40",
    "love", "loved", "love it", "love at first listen",
    "love songs", "love song", "songs i love",
    "songs i absolutely love", "i love this song",
    "like", "i like", "addictive", "eargasm", "memorable",
    "guilty pleasure", "guilty pleasures",
    "albums i own", "rocking out", "heard live",
    "all the best", "others", "other",
    "new", "played", "test", "tag", "names",
    "yes", "yeah", "shit",
    "fun", "feel good", "feelgood", "makes me happy", "makes me cry",
    "life is easy", "conscious",
    # --- negative / worst-side opinions ---
    "worst", "terrible", "horrible", "awful", "bad", "trash", "garbage",
    "boring", "overrated", "sucks", "hate", "hated", "dislike",
    "crap", "crappy", "rubbish", "annoying", "lame", "mediocre", "meh", "ugh",
    "worst song ever", "worst song", "worst ever",
    "not good", "not my thing", "skip", "skippable",
    "weak", "bland", "dull", "forgettable", "unlistenable",
    "shitty", "disappointing", "overhyped", "overplayed",
    # --- user-specific gibberish ---
    "amayzes loved", "slgdmbestof", "davaho53", "vugube62", "k1r7m",
    "eclectonia", "tantotempotaste", "friendsofthekingofrummelpop",
    "i am a party girl here is my soundtrack", "soundtrack of my life",
    "sound storm", "british i like", "awesome guitar jams",
    "ion b chill station", "9 lbs hammer", "rock band dlc", "rock band",
}

MEDIUM_EXACT: set[str] = {
    "via pandora", "heard on pandora", "pandora",
    "radio paradise", "radioparadise",
    "fip", "somafm", "fm4",
    "radio", "stream", "download", "import",
    "wrif-fm", "bagel",
}

_REMOVE_EXACT = {t.lower() for t in SUBJECTIVE_EXACT | MEDIUM_EXACT}

_REMOVE_REGEX: list[re.Pattern] = [
    re.compile(r"^\d+ of 10 stars$", re.I),
    re.compile(r"^\d+ stars?$", re.I),
    re.compile(r"^my .*(favorite|favourite|fav).*$", re.I),
    re.compile(r"^(best|great|good|top|worst)\b.*$", re.I),
    re.compile(r"^(i love|i like|songs i)\b.*$", re.I),
    re.compile(r".*\btag$", re.I),
]


def _should_remove(tag: str) -> bool:
    t = tag.lower().strip()
    if len(t) <= 1:
        return True
    if t in _REMOVE_EXACT:
        return True
    return any(p.match(t) for p in _REMOVE_REGEX)


def clean_tags(tags: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for t in tags:
        tl = t.lower().strip()
        if tl and tl not in seen and not _should_remove(t):
            seen.add(tl)
            out.append(tl)
    return out


# =========================================================================
# Prompt construction
# =========================================================================

SYSTEM_PROMPT = (
    "You are a music metadata writer. Given structured information about a music track, "
    "write a single concise English sentence (150 words) that naturally describes the track. "
    "Include the track name, artist, album, release year, duration when available, "
    "and musical style/genre from the tags. "
    "Do NOT list tags verbatim; instead weave them into a natural description. "
    "Output ONLY the sentence, nothing else."
)


def _listval(v: Any) -> str:
    if isinstance(v, list):
        return v[0] if v else ""
    return str(v) if v else ""


def _first_existing(item: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = item.get(key)
        if value not in (None, ""):
            return value
    return None


def _format_duration(value: Any) -> str:
    value = _listval(value)
    if not value:
        return "unknown"

    if isinstance(value, str):
        value = value.strip()
        if not value:
            return "unknown"
        if re.fullmatch(r"\d{1,2}:\d{2}(?::\d{2})?", value):
            return value
        try:
            duration = float(value)
        except ValueError:
            return value
    else:
        duration = float(value)

    if duration <= 0 or duration != duration:
        return "unknown"

    seconds = int(round(duration / 1000 if duration > 10000 else duration))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes}:{seconds:02d}"


def build_user_prompt(item: dict[str, Any], cleaned_tags: list[str]) -> str:
    track = _listval(item.get("track_name", ""))
    artist = _listval(item.get("artist_name", ""))
    album = _listval(item.get("album_name", ""))
    release = str(item.get("release_date", ""))[:4]
    popularity = item.get("popularity")
    pop_str = f"{popularity:.0f}/100" if popularity is not None else "unknown"
    duration = _first_existing(
        item,
        ("duration_ms", "duration", "track_duration_ms", "track_duration", "length_ms", "length"),
    )
    duration_str = _format_duration(duration)
    tags_str = ", ".join(cleaned_tags) if cleaned_tags else "none"

    return (
        f"Track: {track}\n"
        f"Artist: {artist}\n"
        f"Album: {album}\n"
        f"Release year: {release}\n"
        f"Duration: {duration_str}\n"
        f"Popularity: {pop_str}\n"
        f"Tags: {tags_str}"
    )


# =========================================================================
# Inference
# =========================================================================

def _dtype_to_torch(dtype_name: str) -> torch.dtype:
    return torch.bfloat16 if dtype_name == "bfloat16" else torch.float16


def _dtype_to_vllm(dtype_name: str) -> str:
    return "bfloat16" if dtype_name == "bfloat16" else "float16"


def load_transformers_model(model_name: str, device: str, dtype: torch.dtype):
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading Transformers model: {model_name}  (dtype={dtype})")
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    return tokenizer, model


def load_vllm_model(
    model_name: str,
    dtype_name: str,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    max_model_len: int | None,
):
    try:
        from vllm import LLM
    except ImportError as exc:
        raise RuntimeError(
            "vLLM is not installed. Install it with a CUDA-compatible environment, "
            "or run with --backend transformers."
        ) from exc

    print(f"Loading vLLM model: {model_name}  (dtype={dtype_name})")
    kwargs: dict[str, Any] = {
        "model": model_name,
        "dtype": _dtype_to_vllm(dtype_name),
        "trust_remote_code": True,
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": gpu_memory_utilization,
    }
    if max_model_len is not None:
        kwargs["max_model_len"] = max_model_len

    llm = LLM(**kwargs)
    tokenizer = llm.get_tokenizer()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, llm


def _build_chat_texts(tokenizer, prompts: list[list[dict[str, str]]]) -> list[str]:
    return [
        tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
        for p in prompts
    ]


def _clean_generation(text: str) -> str:
    return text.split("\n")[0].strip()


def generate_batch_transformers(
    tokenizer,
    model,
    prompts: list[list[dict[str, str]]],
    max_new_tokens: int = 150,
    max_input_tokens: int | None = None,
) -> list[str]:
    """Generate responses for a batch of chat-format prompts."""
    texts = _build_chat_texts(tokenizer, prompts)
    tokenizer_kwargs: dict[str, Any] = {
        "return_tensors": "pt",
        "padding": True,
    }
    if max_input_tokens is not None:
        tokenizer_kwargs.update({"truncation": True, "max_length": max_input_tokens})
    else:
        tokenizer_kwargs["truncation"] = False
    inputs = tokenizer(texts, **tokenizer_kwargs).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only the generated part (skip input tokens)
    results: list[str] = []
    for i, output in enumerate(outputs):
        input_len = inputs["input_ids"][i].shape[0]
        generated = output[input_len:]
        text = tokenizer.decode(generated, skip_special_tokens=True).strip()
        # Take first sentence / clean up
        text = _clean_generation(text)
        results.append(text)
    return results


def generate_batch_vllm(
    tokenizer,
    llm,
    prompts: list[list[dict[str, str]]],
    max_new_tokens: int = 150,
    max_input_tokens: int | None = None,
) -> list[str]:
    """Generate responses for a batch of chat-format prompts with vLLM."""
    from vllm import SamplingParams

    texts = _build_chat_texts(tokenizer, prompts)
    sampling_kwargs: dict[str, Any] = {
        "max_tokens": max_new_tokens,
        "temperature": 0.0,
    }
    if max_input_tokens is not None:
        sampling_kwargs["truncate_prompt_tokens"] = max_input_tokens

    sampling_params = SamplingParams(**sampling_kwargs)
    outputs = llm.generate(texts, sampling_params, use_tqdm=False)
    return [
        _clean_generation(output.outputs[0].text.strip()) if output.outputs else ""
        for output in outputs
    ]


# =========================================================================
# Checkpoint helpers
# =========================================================================

def save_checkpoint(checkpoint_path: Path, done_ids: set[str], rows: list[tuple[str, str]]):
    data = {"done_ids": list(done_ids), "rows": rows}
    tmp = checkpoint_path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    tmp.replace(checkpoint_path)


def load_checkpoint(checkpoint_path: Path) -> tuple[set[str], list[tuple[str, str]]]:
    if not checkpoint_path.exists():
        return set(), []
    with checkpoint_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return set(data["done_ids"]), [tuple(r) for r in data["rows"]]


# =========================================================================
# Main
# =========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate track descriptions with Qwen LLM.")
    p.add_argument("--model", type=str, default="Qwen/Qwen3.5-9B",
                    help="HuggingFace model name (default: Qwen3.5-9B)")
    p.add_argument("--backend", choices=["vllm", "transformers"], default="vllm",
                    help="Inference backend (default: vllm)")
    p.add_argument("--batch-size", type=int, default=32,
                    help="Batch size for inference (default: 32)")
    p.add_argument("--max-new-tokens", type=int, default=150,
                    help="Max tokens to generate per track (default: 150)")
    p.add_argument("--max-input-tokens", type=int, default=None,
                    help="Max prompt tokens before generation (default: no truncation)")
    p.add_argument("--dtype", choices=["float16", "bfloat16"], default="bfloat16",
                    help="Model dtype (default: bfloat16)")
    p.add_argument("--device", type=str, default="cuda",
                    help="Transformers device map (default: cuda; ignored by vLLM)")
    p.add_argument("--tensor-parallel-size", type=int, default=1,
                    help="vLLM tensor parallel size (default: 1)")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90,
                    help="vLLM GPU memory utilization ratio (default: 0.90)")
    p.add_argument("--max-model-len", type=int, default=None,
                    help="vLLM max model length override (default: model config)")
    p.add_argument("--dataset", type=str,
                    default="talkpl-ai/TalkPlayData-Challenge-Track-Metadata")
    p.add_argument("--output", type=str, default=None,
                    help="Output CSV path (default: experiment/data/track_sentences.csv)")
    p.add_argument("--checkpoint-every", type=int, default=500,
                    help="Save checkpoint every N batches (default: 500)")
    p.add_argument("--resume", action="store_true",
                    help="Resume from checkpoint if available")
    p.add_argument("--max-tracks", type=int, default=None,
                    help="Process only first N tracks (for testing)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    script_dir = Path(__file__).resolve().parent
    output_path = Path(args.output) if args.output else script_dir / "data" / "track_sentences.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_path.with_suffix(".ckpt.json")

    dtype = _dtype_to_torch(args.dtype)

    # --- Load data ---
    print("Loading track metadata...")
    ds = load_dataset(args.dataset)
    all_items: dict[str, dict] = {}
    for split_name in ["all_tracks", "test_tracks"]:
        if split_name in ds:
            for item in ds[split_name]:
                tid = item["track_id"]
                if tid not in all_items:
                    all_items[tid] = item
    print(f"Total unique tracks: {len(all_items)}")

    # --- Resume ---
    done_ids: set[str] = set()
    rows: list[tuple[str, str]] = []
    if args.resume:
        done_ids, rows = load_checkpoint(checkpoint_path)
        print(f"Resumed from checkpoint: {len(done_ids)} tracks already processed")

    # --- Prepare work items ---
    work_items: list[tuple[str, dict, list[str]]] = []
    for tid, item in all_items.items():
        if tid in done_ids:
            continue
        raw_tags = item.get("tag_list", [])
        cleaned = clean_tags(raw_tags)
        work_items.append((tid, item, cleaned))
    if args.max_tracks:
        work_items = work_items[:args.max_tracks]
    print(f"Tracks to process: {len(work_items)}")

    if not work_items:
        print("Nothing to process.")
        return

    # --- Load model ---
    if args.backend == "vllm":
        tokenizer, model = load_vllm_model(
            args.model,
            args.dtype,
            args.tensor_parallel_size,
            args.gpu_memory_utilization,
            args.max_model_len,
        )
        generate_batch_fn = generate_batch_vllm
    else:
        tokenizer, model = load_transformers_model(args.model, args.device, dtype)
        generate_batch_fn = generate_batch_transformers

    # --- Batch inference ---
    total_batches = (len(work_items) + args.batch_size - 1) // args.batch_size
    t0 = time.time()
    processed_this_run = 0

    for batch_idx in tqdm(range(total_batches), desc="Generating"):
        start = batch_idx * args.batch_size
        end = min(start + args.batch_size, len(work_items))
        batch = work_items[start:end]

        # Build prompts
        prompts = []
        for tid, item, cleaned in batch:
            user_msg = build_user_prompt(item, cleaned)
            prompts.append([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ])

        # Generate
        try:
            sentences = generate_batch_fn(
                tokenizer, model, prompts, args.max_new_tokens, args.max_input_tokens
            )
        except torch.cuda.OutOfMemoryError:
            print(f"\nOOM at batch {batch_idx}. Reducing batch, retrying one by one...")
            torch.cuda.empty_cache()
            gc.collect()
            sentences = []
            for p in prompts:
                try:
                    s = generate_batch_fn(
                        tokenizer, model, [p], args.max_new_tokens, args.max_input_tokens
                    )
                    sentences.append(s[0])
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    sentences.append("")

        for (tid, item, cleaned), sent in zip(batch, sentences):
            rows.append((tid, sent))
            done_ids.add(tid)
        processed_this_run += len(batch)

        # Checkpoint
        if (batch_idx + 1) % args.checkpoint_every == 0:
            save_checkpoint(checkpoint_path, done_ids, rows)
            elapsed = time.time() - t0
            speed = processed_this_run / elapsed
            eta = (len(work_items) - processed_this_run) / speed if speed > 0 else 0
            print(f"\n  Checkpoint saved. {processed_this_run}/{len(work_items)} done this run. "
                  f"Speed: {speed:.1f} tracks/s, ETA: {eta/60:.1f} min")

    # --- Write CSV ---
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["track_id", "sentence"])
        writer.writerows(rows)

    # Clean up checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    elapsed = time.time() - t0
    print(f"\nDone! Wrote {len(rows)} rows to: {output_path}")
    print(f"Total time: {elapsed/60:.1f} min ({elapsed/len(rows):.3f} s/track)")

    # --- Samples ---
    print("\n--- Sample sentences ---")
    for tid, sent in rows[:5]:
        print(f"\n[{tid}]")
        print(f"  {sent}")


if __name__ == "__main__":
    main()
