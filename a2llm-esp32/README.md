# 🧠 A2LM — a genuine, from-scratch language model

**A2LM** is a small decoder-only Transformer (next-token prediction) built from
zero — no API wrappers, no pretrained weights, no shortcuts. It trains locally,
runs on any PC / VPS / Colab, and exports to an **int8 engine that runs on an
ESP32 microcontroller**.

Built and verified end-to-end: dataset → tokenizer → transformer → training →
checkpoint → evaluation → generation → ESP32 export. Every number in this README
comes from an actual run.

```
                    A2LM
                      │
          ┌───────────┼───────────┐
          │           │           │
       A2LM-ESP    A2LM-Base   A2LM-Android
          │           │           │
       Tiny/MCU     Local LLM    Mobile   (roadmap)
```

---

## Why it exists

Most "build your own LLM" projects end up wrapping an API. A2LM is the opposite:
the whole pipeline is ours — tokenizer, attention, training loop, sampler, and a
C inference engine for microcontrollers. It exists to be **understood**.

---

## Architecture (decoder-only Transformer)

```
Input tokens (byte ids)
     ↓
Token Embeddings ──┐
Position Embeddings┘ (+)
     ↓
Transformer Block × N
     │   LayerNorm → Multi-Head Self-Attention → + residual
     │   LayerNorm → Feed-Forward (GELU)       → + residual
     ↓
Final LayerNorm
     ↓
LM Head (tied to token embeddings)
     ↓
Logits over the vocabulary → softmax → next token
```

### Attention, in beginner-friendly math

For each position, the model asks: *"how much should I look at every previous
position?"*

```
score(q, k) = (q · k) / √d_head      # similarity, scaled to stable values
weights    = softmax(score)          # normalized to sum to 1
output     = Σ weights · v           # weighted mix of the values
```

**Causal masking** sets `score(i, j) = −∞` for `j > i`, so a token can never
see the future. Training on `"The cat sat"` means predicting:

```
input  = [The, cat, sat]      target = [cat, sat, ...]
```

The loss is plain cross-entropy between predicted and actual next tokens.

---

## Repo layout

```
a2lm/
├── a2lm/            # the library
│   ├── config.py    # dataclass config + 6 presets (nano → large)
│   ├── tokenizer.py # byte (ESP32-ready) + char tokenizers, save/load
│   ├── dataset.py   # load → tokenize → train/val split → batches
│   ├── attention.py # causal multi-head self-attention (explicit math)
│   ├── model.py     # A2LM: embeddings, blocks, head, generate
│   ├── generate.py  # temperature / top-k / top-p / stop tokens
│   ├── training.py  # AdamW, cosine LR + warmup, AMP, resume, experiments
│   ├── evaluation.py# loss, perplexity, sizes, tok/s report
│   ├── checkpoint.py# save/load/resume
│   ├── quantize.py  # int8 per-channel export for the ESP32
│   ├── cli.py       # `a2llm train|evaluate|generate|count-params|inspect`
│   └── utils.py     # param counting, sizes, throughput
├── scripts/         # thin CLI entry points (also usable directly)
├── configs/         # tiny.json (ESP32) / default.json (PC)
├── esp32/           # ESP-IDF project + pure-C inference engine
├── data/            # tiny_shakespeare.txt (public domain)
├── checkpoints/     # best.pt / latest.pt per run
├── tests/           # 45 tests incl. the overfit correctness test
├── setup_colab.sh   # one-cell Colab setup (free T4 GPU)
├── requirements.txt
└── pyproject.toml   # installs the `a2lm` console command
```

---

## Install

```bash
git clone https://github.com/ayushrajdev9-cmyk/a2llm.git
cd a2llm/a2llm-esp32
cd a2lm
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # or: pip install -e .
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU-only torch
```

Python ≥ 3.10. Torch ≥ 2.1. CUDA is used automatically when available.

## Dataset

```bash
python scripts/prepare_data.py    # downloads Tiny Shakespeare (public domain,
                                  # ~1.1 MB); falls back to a bundled text offline
```

## Train

```bash
# PC normal (default): 837K params, ~10 min on a 4-core CPU
python scripts/train.py --preset mini

# ESP32-light tier: 34K params (what the ESP32 engine runs)
python scripts/train.py --preset nano --out checkpoints/nano

# Colab / VPS with a free T4 GPU
python scripts/train.py --preset base --device auto     # 4.8M params
python scripts/train.py --preset large --device auto    # 14.5M params

# everything is overridable
python scripts/train.py --preset small --steps 10000 --lr 1e-3 --batch-size 64

# resume, or run from a JSON config
python scripts/train.py --preset mini --resume checkpoints/mini/latest.pt
python scripts/train.py --config configs/default.json
```

Presets:

| preset | params  | ctx | dim | layers | where it runs |
|--------|--------:|----:|----:|-------:|---------------|
| nano   |   34K   |  32 |  32 |   2    | **ESP32** (int8, 33 KB flash) |
| micro  |  119K   |  64 |  64 |   2    | any laptop, seconds |
| mini   |  1.06M  | 128 | 144 |   4    | normal PC (default, ~1M params) |
| small  |  2.8M   | 256 | 192 |   6    | better PC / entry GPU |
| base   |  4.9M   | 256 | 256 |   6    | VPS / Colab T4 |
| large  |  14.5M  | 512 | 384 |   8    | best VPS / GPU |

The console prints loss, LR, tok/s, validation loss + perplexity, and live
samples. Every run is tracked under `experiments/exp_NNNN/` (config, metrics,
samples). Checkpoints: `best.pt` (lowest val loss) and `latest.pt` (resumable).

## Evaluate

```bash
python scripts/evaluate.py checkpoints/best.pt
```

Example report (actual run, nano tier):

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

Perplexity = `exp(loss)`: the effective branching factor per token. A random
model over 256 tokens scores 256; a perfect model scores 1.0. A2LM reports
metrics, not vibes — generated text quality is judged by looking at samples.

## Generate

```bash
python scripts/generate.py checkpoints/best.pt \
    --prompt "ROMEO:" --max-new-tokens 200 \
    --temperature 0.9 --top-k 50
```

Flags: `--top-p` (nucleus sampling), `--stop` (stop at text, e.g. `"\n\n"`),
`--seed` (reproducible runs). Same via the CLI: `a2llm generate ...`.

## The `a2lm` CLI

```bash
a2lm train --preset mini
a2lm evaluate checkpoints/best.pt
a2lm generate checkpoints/best.pt --prompt "To be, or"
a2lm count-params --preset nano
a2lm inspect checkpoints/best.pt
```

## ESP32 — A2LM-ESP (the light version)

The `nano` tier runs on an ESP32 (ESP32 / S3 / C3). Weights are quantized to
int8 (one float scale per output row — see `a2lm/quantize.py`), activations
stay float32, and the whole engine is **pure C with static buffers (~30 KB
RAM), no malloc, no dependencies**.

```bash
# 1. train the nano tier, then export:
python scripts/export_esp32.py checkpoints/best.pt        # -> esp32/main/weights.h

# 2. build & flash (ESP-IDF v5.x):
cd esp32
idf.py set-target esp32s3        # or esp32 / esp32c3
idf.py build flash monitor
```

The device prints model stats, then generates from a prompt typed over UART
(115200 baud, `A2LM>` prompt; after 5 s idle it runs the default prompt).

**Verified parity with PyTorch** (host build of the same C code):
max logit difference vs fp32 **0.025** (pure int8 quantization error), greedy
token sequences identical.

```bash
# host-side parity check, no ESP32 needed:
cd esp32/host_test && gcc -O2 -I../main ../main/a2lm.c main.c -lm -o host_a2lm
./host_a2lm logits "ROMEO:"        # dump 256 logits
./host_a2lm generate "ROMEO:"      # greedy
./host_a2lm generate "ROMEO:" 1.0  # temperature sampling
```

**Why it won't burn your ESP32:** the whole model is 33 KB of flash constants;
each generated token is ~1M int8 MACs — a few ms of work at 240 MHz, then the
chip idles. Thermal load is trivial; no overclocking, stock 3.3 V.

## Google Colab (free GPU)

```bash
!bash setup_colab.sh
!python scripts/train.py --preset base --device auto
```

## Tests

```bash
python -m pytest tests/ -q      # 45 tests
```

Includes the critical **overfit test**: a tiny model must drive loss below
0.05 on a fixed batch — if the architecture or training loop is broken, this
test fails. Plus tokenizer round-trips, causal-mask blocking, tensor shapes,
deterministic generation, checkpoint round-trips.

## Colab / VPS tiers

One codebase, six presets — nano (ESP32) through large (VPS). The exact same
checkpoint format works everywhere: train on Colab, evaluate on your laptop,
export to an ESP32.

---

## Hardware requirements

| tier     | RAM   | time (4-core CPU) | notes |
|----------|-------|-------------------|-------|
| nano     | 1 GB  | ~6 min            | also runs on an ESP32 |
| micro    | 1 GB  | ~3 min            | |
| mini     | 2 GB  | ~10 min           | default |
| small    | 4 GB  | ~30 min           | |
| base     | 8 GB  | ~1–2 h CPU, ~10 min T4 | |
| large    | 16 GB | hours CPU, ~40 min T4 | |

CUDA (any GPU) is used automatically; AMP (fp16) is enabled on CUDA.

## Known limitations (honest)

* The presets are **tiny by design** — they memorize statistics, not meaning.
  Expect degenerate output (repeated tokens) at temperature 1; lower
  temperature and top-k help.
* Byte tokenizer: 1 token/byte; no word boundaries (fine at this scale).
* `nano` context is 32 tokens; the model forgets everything older.
* Greedy generation of a 34K model after "ROMEO:" produces spaces — that is
  literally what it learned; the same behavior is reproduced identically by
  the C engine (which is the point of the parity test).
* The dataset is a shuffled token stream (toy pipeline, documented in code).

## Roadmap

```
A2LM-0.01 (this) → A2LM-0.1 → A2LM-1 → A2LM-Android → A2LM-ESP → A2LM-Elite → A2LM-GOAT 🐐
```

Next: KV-cache generation, BPE tokenizer, larger public-domain corpora, int4
quantization, distillation, instruction tuning. Each step is measured, not
assumed.

## License

MIT — open source, free to use, learn from, and break. Built by
**Ayush Rajdev & Anzar Iqbal**.