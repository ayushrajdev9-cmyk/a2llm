# A2LLM

From-scratch, open-source language models — six sizes, one architecture.
No wrappers, no copied weights.

## Models & status

| Model | Params | Status | Release |
|-------|-------:|--------|---------|
| a2llm-esp32-nano | 34,432 | ✅ trained | [v0.1.0](https://github.com/ayushrajdev9-cmyk/a2llm/releases/tag/v0.1.0) |
| a2llm-micro | 120,064 | ✅ trained (multilingual) | [v0.3.0](https://github.com/ayushrajdev9-cmyk/a2llm/releases/tag/v0.3.0) |
| a2llm-mini | 1,056,096 | ✅ trained | [v0.2.0](https://github.com/ayushrajdev9-cmyk/a2llm/releases/tag/v0.2.0) |
| a2llm-small | 2,763,264 | ⏳ training | — |
| a2llm-base | 4,864,000 | ⏳ not trained | — |
| a2llm-large | 14,479,104 | ⏳ not trained | — |

## Releases

| Release | Assets | Result |
|---------|--------|--------|
| [v0.3.0](https://github.com/ayushrajdev9-cmyk/a2llm/releases/tag/v0.3.0) | `a2llm-micro.pt` (+ nano, mini) | micro: val_loss 1.5686, ppl 4.80 |
| [v0.2.0](https://github.com/ayushrajdev9-cmyk/a2llm/releases/tag/v0.2.0) | `a2llm-mini.pt` | mini: val_loss 3.3162, ppl 27.56 |
| [v0.1.0](https://github.com/ayushrajdev9-cmyk/a2llm/releases/tag/v0.1.0) | `a2llm-esp32-nano.pt`, `a2llm-micro.pt` | first trained checkpoints |

## Download

```bash
python scripts/download_model.py            # list
python scripts/download_model.py nano       # ESP32 model
python scripts/download_model.py micro      # laptop model
python scripts/download_model.py mini       # default PC model
```

## Train (Colab, free T4)

- [colab_nano.ipynb](https://colab.research.google.com/github/ayushrajdev9-cmyk/a2llm/blob/main/colab_nano.ipynb) (~18 min)
- [colab_micro.ipynb](https://colab.research.google.com/github/ayushrajdev9-cmyk/a2llm/blob/main/colab_micro.ipynb) (~40 min)
- [colab_mini.ipynb](https://colab.research.google.com/github/ayushrajdev9-cmyk/a2llm/blob/main/colab_mini.ipynb) (~30 min)
- [colab_small.ipynb](https://colab.research.google.com/github/ayushrajdev9-cmyk/a2llm/blob/main/colab_small.ipynb) (~1.25 h)

## Local

```bash
pip install -r requirements.txt
python scripts/train.py --preset mini       # train
python scripts/generate.py checkpoints/best.pt --prompt "ROMEO:"
python -m pytest tests/ -q                  # 45 tests
```

ESP32 port: `esp32/` (int8 engine for ESP32 / S3 / C3).

MIT — Ayush Rajdev & Anzar Iqbal.