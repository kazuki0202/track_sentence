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
