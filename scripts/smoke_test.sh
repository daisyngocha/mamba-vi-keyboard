#!/usr/bin/env bash
# Kiểm tra pipeline: config -> data -> model -> train -> eval -> checkpoint. Chạy < 1 phút.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m src.train --config configs/smoke.yaml
test -f results/smoke/summary.json && echo "SMOKE TEST OK"
