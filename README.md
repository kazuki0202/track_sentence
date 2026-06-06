# track_sentence

Generate concise English track-description sentences from track metadata.

The default inference backend is vLLM:

```bash
python track_sentence.py --model Qwen/Qwen2.5-7B-Instruct --batch-size 64
```

Useful vLLM options:

```bash
python track_sentence.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --batch-size 128 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.95 \
  --resume
```

Fallback to the original Transformers backend:

```bash
python track_sentence.py --backend transformers --batch-size 32
```

RTX A5000 / no-vLLM path:

```bash
python track_sentence2.py --model Qwen/Qwen3.5-9B --batch-size 8 --resume
```

If 24GB VRAM is still tight, use 4-bit quantization:

```bash
python track_sentence2.py --model Qwen/Qwen3.5-9B --load-in-4bit --batch-size 16 --resume
```
