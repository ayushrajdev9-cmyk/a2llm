# 🧠 A2LLM — the LLM family that runs on anything

**A2LLM** is a from-scratch language-model project — no API wrappers, no copied
weights. One core architecture, many sizes, one repo:

```
a2llm/
├── a2llm-esp32/     ✅ DONE — tiny int8 model + pure-C engine for ESP32
├── a2llm-base/      🔜 next — local LLM for PC / VPS / Colab
├── a2llm-android/   🔜 future — on-device mobile inference
└── a2llm-elite/     🔜 future — bigger, distilled, instruction-tuned
```

Every variant shares the same principles: small, efficient, understandable,
reproducible, measurable, honest about its capabilities.

> **First milestone achieved:** A2LLM trains itself to predict the next token,
> saves a checkpoint, reloads it, generates text locally — and the same
> checkpoint runs on an ESP32 microcontroller.

---

# 📖 Complete instructions

## 1. What's inside `a2llm-esp32/`

| Piece | What it is |
|---|---|
| `a2llm/` | Python library: config, tokenizer, dataset, attention, model, training, evaluation, checkpointing, generation, int8 quantization |
| `scripts/` | CLI entry points: train, evaluate, generate, export_esp32, prepare_data |
| `esp32/` | ESP-IDF project + **pure-C inference engine** (no malloc, no dependencies) |
| `tests/` | 45 tests including the overfit correctness test |
| `setup_colab.sh` | One-cell Google Colab setup (free T4 GPU) |

## 2. Architecture (decoder-only Transformer)

```
Input tokens (byte ids)
     ↓
Token Embeddings ──┐
Position Embeddings┘ (+)                # learned positional info
     ↓
Transformer Block × N
     │   LayerNorm → Multi-Head Self-Attention → + residual
     │   LayerNorm → Feed-Forward (GELU)       → + residual
     ↓
Final LayerNorm
     ↓
LM Head (tied to token embeddings)
     ↓
Logits → softmax → sample next token
```

**Attention in one line of math:** for each position, `weights = softmax((Q·Kᵀ)/√d)` 
then `output = weights·V` — each token asks *"how much should I look at every
previous token?"*. A **causal mask** forces `score(i,j) = −∞` for `j > i` so the
model never sees the future. Training on `"The cat sat"` predicts
`cat, sat, ...` — plain cross-entropy on shifted sequences.

## 3. Install

```bash
git clone https://github.com/ayushrajdev9-cmyk/a2llm.git
cd a2llm/a2llm-esp32
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU-only torch
```

Requirements: Python ≥ 3.10, torch ≥ 2.1, numpy. CUDA is used automatically
when available (never assumed).

## 4. Dataset

```bash
python scripts/prepare_data.py
```

Downloads **Tiny Shakespeare** (public domain, ~1.1 MB). If the download fails
it writes a bundled public-domain fallback — the pipeline always works offline.

## 5. Train

```bash
python scripts/train.py --preset mini     # PC normal: 838K params, ~10 min CPU
```

Presets (one codebase, six sizes):

| preset | params  | ctx | dim | layers | where it runs |
|--------|--------:|----:|----:|-------:|---------------|
| nano   |   34K   |  32 |  32 |   2    | **ESP32** (int8, 33 KB flash) |
| micro  |  119K   |  64 |  64 |   2    | any laptop, seconds |
| mini   |  838K   | 128 | 128 |   4    | normal PC (default) |
| small  |  2.8M   | 256 | 192 |   6    | better PC / entry GPU |
| base   |  4.9M   | 256 | 256 |   6    | VPS / Colab T4 |
| large  |  14.5M  | 512 | 384 |   8    | best VPS / GPU |

Useful flags: `--steps N`, `--batch-size N`, `--lr X`, `--ctx N`, `--dim N`,
`--layers N`, `--device cpu|cuda|auto`, `--seed N`, `--config file.json`,
`--resume checkpoints/latest.pt`, `--grad-accum N`.

The console shows loss, LR, tok/s, validation loss + perplexity and live
samples. Every run is logged to `experiments/exp_NNNN/`. Checkpoints:
`best.pt` (lowest val loss) and `latest.pt` (resumable).

## 6. Evaluate

```bash
python scripts/evaluate.py checkpoints/best.pt
```

Real report from the shipped nano checkpoint:

```
A2LM evaluation report
--------------------------------------------
Checkpoint:       checkpoints/best.pt
Parameters:       34,432 (trainable 34,432)
Model size:       33.6 KB int8 | 134.5 KB fp32
Validation loss:  3.3193
Perplexity:       27.64
Generation speed: 5785 tok/s (cpu)
Vocab / context:  256 / 32
--------------------------------------------
Baseline: uniform random over the vocab has perplexity = vocab size.
```

Perplexity = `exp(loss)` — the effective branching factor at each token.
A random model over 256 tokens scores 256; a perfect model scores 1.0.
All metrics come from real runs. Text quality is judged by reading samples.

## 7. Generate

```bash
python scripts/generate.py checkpoints/best.pt \
    --prompt "ROMEO:" --max-new-tokens 200 \
    --temperature 0.9 --top-k 50
```

Flags: `--top-p` (nucleus sampling), `--stop` (stop at text), `--seed`
(reproducible). Sampling math: logits ÷ temperature → top-k filter → optional
top-p filter → sample.

## 8. The `a2llm` CLI (after `pip install -e .`)

```bash
a2llm train --preset mini
a2llm evaluate checkpoints/best.pt
a2llm generate checkpoints/best.pt --prompt "To be, or"
a2llm count-params --preset nano
a2llm inspect checkpoints/best.pt
```

## 9. ESP32 — A2LLM-ESP32 (the light version)

Train the `nano` tier (34K params), export to int8, flash, done:

```bash
# 1. train the nano tier:
python scripts/train.py --preset nano --out checkpoints/nano

# 2. export weights to C:
python scripts/export_esp32.py checkpoints/nano/best.pt   # -> esp32/main/weights.h

# 3. build & flash (ESP-IDF v5.x):
cd esp32
idf.py set-target esp32s3        # or esp32 / esp32c3
idf.py build flash monitor
```

On boot the device prints model stats, then a `A2LM>` UART prompt (115200 baud).
Type a prompt; after 5 s idle it runs the default `ROMEO:`. It streams
generated text and reports tok/s + free heap.

**Verified parity with PyTorch** (host build of the same C code):
max logit difference vs fp32 **0.025** (pure int8 quantization error),
greedy token sequences identical.

```bash
# host-side check without an ESP32:
cd esp32/host_test && gcc -O2 -I../main ../main/a2lm.c main.c -lm -o host_a2lm
./host_a2lm logits "ROMEO:"        # dump 256 logits
./host_a2lm generate "ROMEO:"      # greedy
./host_a2lm generate "ROMEO:" 1.0  # temperature sampling
```

**Why it won't burn your ESP32:** 33 KB of flash constants, ~30 KB static RAM,
~1M int8 MACs per token (a few ms at 240 MHz), then idle. No overclocking,
stock 3.3 V. Works on ESP32, ESP32-S3, ESP32-C3.

## 10. Google Colab (free T4 GPU)

```bash
!bash setup_colab.sh               # clones repo, installs torch (CUDA), data
!python scripts/train.py --preset base --device auto
```

## 11. Tests

```bash
python -m pytest tests/ -q          # 45 tests
```

Includes the critical **overfit test**: a tiny model must drive loss < 0.05 on
a fixed batch — if architecture or training loop is broken, it fails. Plus
tokenizer round-trips, causal-mask blocking, tensor shapes, deterministic
generation, checkpoint round-trips.

## 12. Configuration

Everything lives in `a2llm/config.py` (dataclasses) — nothing hard-coded.
Override via CLI flags or a JSON config (`configs/tiny.json` = nano tier,
`configs/default.json` = mini tier). Every experiment is reproducible from its
configuration, stored in the checkpoint.

## 13. Hardware requirements

| tier    | RAM  | time (4-core CPU) | notes |
|---------|------|-------------------|-------|
| nano    | 1 GB | ~6 min            | also runs on ESP32 |
| micro   | 1 GB | ~3 min            | |
| mini    | 2 GB | ~10 min           | default |
| small   | 4 GB | ~30 min           | |
| base    | 8 GB | ~1–2 h CPU / ~10 min T4 | |
| large   | 16 GB| hours CPU / ~40 min T4 | |

## 14. Known limitations (honest)

* Presets are **tiny by design** — they learn statistics, not meaning.
  Expect degenerate output at temperature 1; lower temperature + top-k helps.
* Byte tokenizer: 1 token/byte, no word boundaries (fine at this scale).
* `nano` context is 32 tokens — older context is forgotten.
* Greedy nano after "ROMEO:" outputs spaces — that is literally what it
  learned, and the C engine reproduces it identically (that's the parity test).
* Dataset is a shuffled token stream (toy pipeline, documented in code).

## 15. Roadmap

```
A2LLM-0.01 ✅ → a2llm-base 🔜 → a2llm-android → a2llm-elite → A2LLM-GOAT 🐐
```

Next steps, each measured not assumed: KV-cache generation, BPE tokenizer,
bigger public-domain corpora, int4 quantization, distillation, instruction
tuning, Android/iOS on-device runtime.

## License

MIT — free to use, learn from, and break.
Built by **Ayush Rajdev & Anzar Iqbal**.