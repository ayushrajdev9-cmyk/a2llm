#!/usr/bin/env bash
# One-cell setup for Google Colab (free T4 GPU): clone, install, train.
# Paste this into a Colab cell and run:
#
#     !bash setup_colab.sh
#
# Then train, e.g.:  !python scripts/train.py --preset base --device auto
set -e
cd /content
if [ ! -d a2lm ]; then
  git clone https://github.com/ayushrajdev9-cmyk/a2lm.git
fi
cd a2lm
pip install -q torch --index-url https://download.pytorch.org/whl/cu124 2>/dev/null || pip install -q torch
pip install -q numpy pytest
python scripts/prepare_data.py
echo ""
echo "A2LM ready on Colab. Train with:"
echo "  python scripts/train.py --preset base --device auto   # VPS-grade, free GPU"
echo "  python scripts/train.py --preset large --device auto  # biggest tier"