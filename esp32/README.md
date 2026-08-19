# a2llm-esp32

The `nano` tier (34,432 params, int8) running on an ESP32 / S3 / C3.

Trained model: `a2llm-esp32-nano.pt` — [v0.1.0](https://github.com/ayushrajdev9-cmyk/a2llm/releases/tag/v0.1.0).

```bash
python scripts/export_esp32.py checkpoints/nano/best.pt   # -> esp32/main/weights.h
cd esp32
idf.py set-target esp32s3        # or esp32 / esp32c3
idf.py build flash monitor
```

UART 115200 baud, `A2LM>` prompt. Host parity check:

```bash
cd esp32/host_test && gcc -O2 -I../main ../main/a2lm.c main.c -lm -o host_a2lm
./host_a2lm generate "ROMEO:"
```