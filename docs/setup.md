# Setup

## One command after cloning

```bash
git clone https://github.com/shivam-anand-03/sneha-majorproject.git
cd sneha-majorproject
./dev.sh setup
```

`./dev.sh setup` does everything needed on a fresh machine:

1. Installs [uv](https://docs.astral.sh/uv/) into `~/.local/bin` if it is missing.
2. Creates `.venv` with Python 3.11. uv downloads Python itself into `.cache/python`, so the system Python version doesn't matter.
3. Installs PyTorch 2.6. It picks CUDA 12.4 wheels if `nvidia-smi` works, and CPU wheels otherwise.
4. Installs everything else from `requirements.txt`.
5. Downloads the handwriting fonts used by the synthetic data generator into `data/fonts/`.
6. Prints the torch version and whether CUDA is visible, then runs the 52 fast tests (about 10 s).

When it finishes:

```bash
./dev.sh demo     # try the UI right away with the demo pipeline → http://127.0.0.1:8000
./dev.sh train    # train the text grader (needed for real grading, ~15 min on an RTX 4050)
./dev.sh          # start the app with the full pipeline
```

Trained models (`models/`), datasets, outputs and caches are git-ignored, so a fresh clone always needs
`./dev.sh train` before the full pipeline can grade text. Until then, use `./dev.sh demo`.

## Requirements

| | Minimum | Used for the reported results |
|---|---|---|
| OS | Linux or WSL2 (macOS works on CPU, see below) | Ubuntu, x86_64 |
| Tools | `bash`, `curl`, `git` | |
| GPU | none (CPU works, but slowly) | NVIDIA RTX 4050, 6 GB |
| NVIDIA driver | supports CUDA 12.4 (driver ≥ 550) | |
| Disk | ~15 GB (venv ~7 GB, models and datasets ~8 GB) | |
| RAM | 16 GB | |

Everything stays inside the project folder. `dev.sh` and `src/__init__.py` point `HF_HOME`, `TORCH_HOME`,
`PADDLE_PDX_CACHE_HOME`, `UV_CACHE_DIR` and the uv Python install directory at `./.cache`.
Deleting the folder removes everything.

## Options

| Variable | Effect |
|---|---|
| `TORCH_INDEX=cpu ./dev.sh setup` | Force CPU torch even when a GPU is present |
| `TORCH_INDEX=cu124 ./dev.sh setup` | Force CUDA torch (e.g. the driver is installed but `nvidia-smi` is not on `PATH`) |
| `PORT=9000 ./dev.sh start` | Serve on another port |

Setup is safe to re-run. It reuses `.venv` and skips fonts that are already downloaded.
To start over completely: `rm -rf .venv .cache && ./dev.sh setup`.

## Optional extras

- **Gemini backend.** Run `export GEMINI_API_KEY=...` and set `external_api.allow_gemini: true` in
  `configs/config.yaml`. Only line and diagram crops are sent, never full pages.
- **Anonymisation salt** for real student data: `export AES_ANON_SALT=<random secret>`. Never commit it.
- **direnv users:** `.envrc` exports the same cache variables, so ad-hoc `uv pip install` calls also cache inside the project.

## Platform notes

- **macOS:** setup installs CPU torch and skips `bitsandbytes`, so the 4-bit Qwen2.5-VL models can't load.
  Set `ocr.vlm.load_in_4bit: false` (needs much more memory), or use the Gemini backend.
- **WSL2:** install the NVIDIA Windows driver. Don't install a Linux driver inside WSL. `nvidia-smi` should then work inside WSL, and setup will choose CUDA.
- **No GPU:** everything runs, but training the cross-encoder on CPU takes hours. Use `./dev.sh demo` to explore the UI.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `uv: command not found` right after setup in a new shell | Add `~/.local/bin` to `PATH` (the uv installer prints the line for your shell). |
| `cuda available: False` on a GPU machine | Check `nvidia-smi`. If it works, re-run with `TORCH_INDEX=cu124`. Then `rm -rf .venv` and run setup again. |
| `CUDA out of memory` during training or VLM calls | Close other GPU programs. Lower `grading_text.cross_encoder.batch_size` in `configs/config.yaml`. |
| Hugging Face downloads stall | `HF_HUB_DISABLE_XET=1` is already set. Re-run, because downloads resume from `.cache/huggingface`. |
| Port 8000 already in use | `PORT=8001 ./dev.sh start` |
| App says "no trained text grader" | Run `./dev.sh train`, or use `./dev.sh demo`. |
