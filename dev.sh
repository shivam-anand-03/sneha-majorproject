#!/usr/bin/env bash
# One entry point for the project. Everything (venv, models, datasets, caches) stays inside this folder.
#   ./dev.sh            -> setup if needed, then start the app on http://127.0.0.1:8000
#   ./dev.sh setup      -> create .venv (Python 3.11), install deps, fetch fonts
#   ./dev.sh start      -> start FastAPI + web UI
#   ./dev.sh demo       -> start the app with the demo pipeline (no trained models needed)
#   ./dev.sh train      -> fine-tune the cross-encoder + fit combiner weights (GPU recommended)
#   ./dev.sh synthetic  -> generate synthetic answer sheets
#   ./dev.sh eval       -> run all evaluation scripts
#   ./dev.sh test       -> run the fast test suite
set -euo pipefail
cd "$(dirname "$0")"

export UV_CACHE_DIR="$PWD/.cache/uv" UV_PYTHON_INSTALL_DIR="$PWD/.cache/python" HF_HOME="$PWD/.cache/huggingface" \
       PADDLE_PDX_CACHE_HOME="$PWD/.cache/paddlex" TORCH_HOME="$PWD/.cache/torch" \
       HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
PY=.venv/bin/python
PORT="${PORT:-8000}"

setup() {
  command -v uv >/dev/null || { echo "install uv first: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }
  [ -x "$PY" ] || uv venv --python 3.11 .venv
  uv pip install --python "$PY" torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
  grep -vE '^(torch|torchvision)==' requirements.txt | uv pip install --python "$PY" -r /dev/stdin
  fonts
}

fonts() {
  mkdir -p data/fonts
  local base=https://raw.githubusercontent.com/google/fonts/main
  for f in "ofl/caveat/Caveat%5Bwght%5D.ttf:Caveat" "ofl/indieflower/IndieFlower-Regular.ttf:IndieFlower" \
           "ofl/shadowsintolight/ShadowsIntoLight.ttf:ShadowsIntoLight" "ofl/reeniebeanie/ReenieBeanie.ttf:ReenieBeanie" \
           "ofl/patrickhand/PatrickHand-Regular.ttf:PatrickHand" "apache/homemadeapple/HomemadeApple-Regular.ttf:HomemadeApple"; do
    [ -f "data/fonts/${f##*:}.ttf" ] || curl -sfL -o "data/fonts/${f##*:}.ttf" "$base/${f%%:*}"
  done
}

start() {
  if [ ! -f models/text_combiner.json ] && [ -z "${AES_DEMO_PIPELINE:-}" ]; then
    echo "note: no trained text grader in models/ -> run ./dev.sh train, or ./dev.sh demo for the demo pipeline"
  fi
  echo "open http://127.0.0.1:$PORT"
  exec .venv/bin/uvicorn app.api:app --host 127.0.0.1 --port "$PORT"
}

case "${1:-auto}" in
  auto)      [ -x "$PY" ] || setup; [ -d data/fonts ] || fonts; start ;;
  setup)     setup ;;
  start)     start ;;
  demo)      AES_DEMO_PIPELINE=1 start ;;
  train)     $PY -m scripts.train_text_grader ;;
  synthetic) fonts; $PY -m scripts.make_synthetic --n-iam 30 --n-answer 60 ;;
  eval)      $PY -m scripts.eval_segmentation --methods projection paddle
             $PY -m scripts.eval_ocr --engines trocr paddle vlm
             $PY -m scripts.train_text_grader --skip-train
             $PY -m scripts.eval_diagram
             $PY -m scripts.eval_xai
             $PY -m scripts.eval_history
             $PY -m scripts.eval_e2e ;;
  test)      $PY -m pytest ;;
  *)         sed -n '2,10p' "$0"; exit 1 ;;
esac
