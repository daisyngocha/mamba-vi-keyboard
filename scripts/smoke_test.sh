#!/usr/bin/env bash
# Kiểm tra pipeline: config -> data -> model -> train -> eval -> checkpoint -> log. Chạy < 1 phút, CPU cũng chạy.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=.; [ -d /kaggle/working ] && OUT=/kaggle/working   # train.py đặt output dưới /kaggle/working khi chạy trên Kaggle
python train.py --config configs/smoke.yaml --no_resume
test -f "$OUT/logs/_smoke/smoke.csv" && test -f "$OUT/checkpoints/smoke/ckpt_latest.pt" && echo "SMOKE TEST OK"
