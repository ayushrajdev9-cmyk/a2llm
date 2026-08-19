# A2LLM model lineup

======================================================================
A2LLM - MODELS & PARAMETERS  (updated)
Repo: https://github.com/ayushrajdev9-cmyk/a2llm
Releases: https://github.com/ayushrajdev9-cmyk/a2llm/releases
======================================================================

model                     params       fp32       int8   where it runs
----------------------------------------------------------------------
A2LLM-ESP32 (nano)        34,432      0.1 MB    33.6 KB   ESP32 microcontroller
a2llm-micro              120,064      0.5 MB   117.2 KB   any laptop, ~90 sec
a2llm-mini             1,056,096      4.0 MB  1031.3 KB   normal PC (default)
a2llm-small            2,763,264     10.5 MB  2698.5 KB   better PC / entry GPU
a2llm-base             4,864,000     18.6 MB  4750.0 KB   VPS / Colab T4
a2llm-large           14,479,104     55.2 MB 14139.8 KB   best VPS / GPU

STATUS:
  - A2LLM-ESP32 (nano): TRAINED ✅  (checkpoints/nano_best.pt, GitHub release v0.1.0)
  - a2llm-micro: TRAINED ✅  (GitHub release v0.3.0, MULTILINGUAL corpus, val_loss 1.5686, ppl 4.80, 200K steps on T4)
  - a2llm-mini: TRAINED ✅  (GitHub release v0.2.0, val_loss 3.3162, ppl 27.56, 4000 steps on T4 in 67.6s)
  - a2llm-small: config ready - not trained yet
  - a2llm-base: config ready - not trained yet (best for Colab T4)
  - a2llm-large: config ready - not trained yet

DOWNLOAD trained models:
  python scripts/download_model.py            # list
  python scripts/download_model.py nano       # ESP32 model
  python scripts/download_model.py micro      # laptop model (multilingual, v0.3.0)
  python scripts/download_model.py mini        # default PC model (T4 trained)

TRAIN any model (Colab T4 recommended):
  python scripts/train.py --preset <nano|micro|mini|small|base|large>

ESP32 export: python scripts/export_esp32.py checkpoints/best.pt
Generate:     python scripts/generate.py checkpoints/best.pt --prompt 'ROMEO:'

