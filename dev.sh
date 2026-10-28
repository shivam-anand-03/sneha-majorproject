#!/usr/bin/env bash
# One entry point for the project. Everything (venv, models, datasets, caches) stays inside this folder.
#   ./dev.sh            -> setup if needed, then start the app on http://127.0.0.1:8000
#   ./dev.sh setup      -> first run after cloning: install uv if missing, create .venv (Python 3.11),
#                          install deps (CUDA torch if an NVIDIA GPU is found, else CPU), fetch fonts, run tests
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
  command -v curl >/dev/null || { echo "curl is required (sudo apt install curl)"; exit 1; }
  if ! command -v uv >/dev/null; then
    echo "uv not found -> installing it to ~/.local/bin"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
  fi
  [ -x "$PY" ] || uv venv --python 3.11 .venv
  # TORCH_INDEX=cu124|cpu overrides the auto choice (CUDA wheels only when an NVIDIA driver answers)
  local idx="${TORCH_INDEX:-cpu}" skip='^(torch|torchvision)=='
  [ -z "${TORCH_INDEX:-}" ] && nvidia-smi -L >/dev/null 2>&1 && idx=cu124
  [ "$(uname -s)" = Darwin ] && skip='^(torch|torchvision|bitsandbytes)=='   # no 4-bit on macOS
  echo "installing torch ($idx wheels)"
  uv pip install --python "$PY" torch==2.6.0 torchvision==0.21.0 --index-url "https://download.pytorch.org/whl/$idx"
  grep -vE "$skip" requirements.txt | uv pip install --python "$PY" -r /dev/stdin
  fonts
  mkdir -p outputs/logs
  $PY -c "import torch; print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available())"
  $PY -m pytest
  echo
  echo "setup done. next:"
  echo "  ./dev.sh demo    # try the UI now (no trained models needed)"
  echo "  ./dev.sh train   # train the text grader (~15 min on GPU), then ./dev.sh start"
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
  *)         sed -n '2,11p' "$0"; exit 1 ;;
esac
