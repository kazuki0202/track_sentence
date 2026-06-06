"""Generate English description sentences for tracks using a Qwen LLM.

Usage:
    python track_sentence2.py                          # RTX A5000-friendly defaults
    python track_sentence2.py --model Qwen/Qwen3.5-9B --batch-size 8
    python track_sentence2.py --load-in-4bit --batch-size 16
    python track_sentence2.py --resume   # resume from checkpoint

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
from transformers import AutoModelForCausalLM, AutoTokenizer

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
    "write a single concise English sentence that naturally describes the track. "
    "Include the track name, artist, album, release year, duration when available, "
    "and musical style/genre from the tags. "
    "Do NOT list tags verbatim; instead weave them into a natural description. "
    "Do NOT output thinking, reasoning, analysis, labels, markdown, or explanations. "
    "Output ONLY the final sentence, nothing else."
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


def build_user_prompt(item: dict[str, Any], cleaned_tags: list[str], max_tags: int) -> str:
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
    tags = cleaned_tags[:max_tags] if max_tags > 0 else cleaned_tags
    tags_str = ", ".join(tags) if tags else "none"

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

def _dtype_from_name(dtype_name: str) -> torch.dtype:
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float32":
        return torch.float32
    return torch.float16


def _enable_cuda_fast_math() -> None:
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except AttributeError:
        pass


def load_model(
    model_name: str,
    device: str,
    dtype: torch.dtype,
    attn_implementation: str,
    load_in_4bit: bool,
):
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    _enable_cuda_fast_math()

    print(f"Loading model: {model_name}  (dtype={dtype}, attn={attn_implementation})")
    model_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "device_map": device,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "attn_implementation": attn_implementation,
    }

    if load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise RuntimeError(
                "--load-in-4bit requires bitsandbytes and a recent transformers version."
            ) from exc
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )

    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    except (ImportError, ValueError) as exc:
        if attn_implementation == "flash_attention_2":
            print("flash_attention_2 is unavailable; retrying with sdpa.")
            model_kwargs["attn_implementation"] = "sdpa"
            model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        else:
            raise exc
    model.eval()
    return tokenizer, model


def _apply_chat_template_no_thinking(tokenizer, prompt: list[dict[str, str]]) -> str:
    try:
        return tokenizer.apply_chat_template(
            prompt,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            prompt,
            tokenize=False,
            add_generation_prompt=True,
        )


def _clean_generated_sentence(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.I | re.S).strip()
    text = re.sub(r".*</think>", "", text, flags=re.I | re.S).strip()

    final_markers = (
        r"final answer\s*:",
        r"final sentence\s*:",
        r"answer\s*:",
        r"sentence\s*:",
    )
    for marker in final_markers:
        matches = list(re.finditer(marker, text, flags=re.I))
        if matches:
            text = text[matches[-1].end():].strip()
            break

    lines = []
    skip_prefixes = (
        "thinking process",
        "thought process",
        "reasoning",
        "analysis",
        "final answer",
        "final sentence",
        "answer",
        "sentence",
    )
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        lowered = line.lower().strip(":： ")
        if any(lowered.startswith(prefix) for prefix in skip_prefixes):
            continue
        lines.append(line)

    if not lines:
        return ""

    text = lines[0].strip().strip('"')
    if _is_bad_sentence(text):
        return ""
    return text


def _is_bad_sentence(sentence: str) -> bool:
    text = sentence.strip()
    if not text:
        return True
    lowered = text.lower()
    bad_starts = (
        "thinking process",
        "thought process",
        "reasoning",
        "analysis",
        "<think",
        "</think",
    )
    return lowered.startswith(bad_starts)


def generate_batch(
    tokenizer,
    model,
    prompts: list[list[dict[str, str]]],
    max_new_tokens: int = 150,
    max_input_tokens: int | None = None,
) -> list[str]:
    """Generate responses for a batch of chat-format prompts."""
    texts = [
        _apply_chat_template_no_thinking(tokenizer, p)
        for p in prompts
    ]
    tokenizer_kwargs: dict[str, Any] = {
        "return_tensors": "pt",
        "padding": True,
    }
    if max_input_tokens is not None:
        tokenizer_kwargs.update({"truncation": True, "max_length": max_input_tokens})
    else:
        tokenizer_kwargs["truncation"] = False
    inputs = tokenizer(texts, **tokenizer_kwargs).to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

    # Decode only the generated part (skip input tokens)
    results: list[str] = []
    for i, output in enumerate(outputs):
        input_len = inputs["input_ids"][i].shape[0]
        generated = output[input_len:]
        text = tokenizer.decode(generated, skip_special_tokens=True).strip()
        text = _clean_generated_sentence(text)
        results.append(text)
    return results


def generate_with_auto_split(
    tokenizer,
    model,
    prompts: list[list[dict[str, str]]],
    max_new_tokens: int,
    max_input_tokens: int | None,
) -> list[str]:
    try:
        return generate_batch(tokenizer, model, prompts, max_new_tokens, max_input_tokens)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        gc.collect()
        if len(prompts) == 1:
            print("OOM on a single prompt; writing an empty sentence for this track.")
            return [""]

        mid = len(prompts) // 2
        print(f"\nOOM with batch size {len(prompts)}; retrying as {mid} + {len(prompts) - mid}.")
        return (
            generate_with_auto_split(tokenizer, model, prompts[:mid], max_new_tokens, max_input_tokens)
            + generate_with_auto_split(tokenizer, model, prompts[mid:], max_new_tokens, max_input_tokens)
        )


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

    valid_rows: list[tuple[str, str]] = []
    dropped = 0
    for row in data["rows"]:
        if len(row) < 2:
            dropped += 1
            continue
        track_id, sentence = str(row[0]), str(row[1])
        cleaned_sentence = _clean_generated_sentence(sentence)
        if _is_bad_sentence(cleaned_sentence):
            dropped += 1
            continue
        valid_rows.append((track_id, cleaned_sentence))

    if dropped:
        print(
            f"Checkpoint cleanup: dropped {dropped} invalid/thinking rows; "
            "they will be regenerated."
        )

    done_ids = {track_id for track_id, _ in valid_rows}
    return done_ids, valid_rows


# =========================================================================
# Main
# =========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate track descriptions with Qwen LLM.")
    p.add_argument("--model", type=str, default="Qwen/Qwen3.5-9B",
                    help="HuggingFace model name (default: Qwen3.5-9B)")
    p.add_argument("--batch-size", type=int, default=8,
                    help="Batch size for inference (default: 8; RTX A5000 friendly)")
    p.add_argument("--max-new-tokens", type=int, default=80,
                    help="Max tokens to generate per track (default: 80)")
    p.add_argument("--max-input-tokens", type=int, default=512,
                    help="Max prompt tokens before generation (default: 512)")
    p.add_argument("--max-tags", type=int, default=24,
                    help="Max cleaned tags included in each prompt (default: 24)")
    p.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16",
                    help="Model dtype (default: float16; recommended for RTX A5000)")
    p.add_argument("--device", type=str, default="cuda",
                    help="Transformers device map (default: cuda)")
    p.add_argument("--attn-implementation", choices=["sdpa", "flash_attention_2", "eager"],
                    default="sdpa",
                    help="Attention implementation (default: sdpa)")
    p.add_argument("--load-in-4bit", action="store_true",
                    help="Use bitsandbytes 4-bit quantization to reduce VRAM use")
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

    dtype = _dtype_from_name(args.dtype)

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
    tokenizer, model = load_model(
        args.model,
        args.device,
        dtype,
        args.attn_implementation,
        args.load_in_4bit,
    )

    # --- Batch inference ---
    total_batches = (len(work_items) + args.batch_size - 1) // args.batch_size
    t0 = time.time()
    processed_this_run = 0
    invalid_this_run = 0

    for batch_idx in tqdm(range(total_batches), desc="Generating"):
        start = batch_idx * args.batch_size
        end = min(start + args.batch_size, len(work_items))
        batch = work_items[start:end]

        # Build prompts
        prompts = []
        for tid, item, cleaned in batch:
            user_msg = build_user_prompt(item, cleaned, args.max_tags)
            prompts.append([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ])

        # Generate
        sentences = generate_with_auto_split(
            tokenizer, model, prompts, args.max_new_tokens, args.max_input_tokens
        )

        for (tid, item, cleaned), sent in zip(batch, sentences):
            if _is_bad_sentence(sent):
                invalid_this_run += 1
                continue
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

    # Clean up checkpoint only when every attempted track produced a valid sentence.
    if invalid_this_run == 0 and checkpoint_path.exists():
        checkpoint_path.unlink()
    elif invalid_this_run:
        save_checkpoint(checkpoint_path, done_ids, rows)
        print(
            f"\nWarning: {invalid_this_run} invalid/thinking outputs were not marked done. "
            f"Checkpoint kept for another --resume attempt: {checkpoint_path}"
        )

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
