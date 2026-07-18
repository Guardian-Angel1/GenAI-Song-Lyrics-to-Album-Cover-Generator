# 🎨 Song-to-Album Cover Generator

Generate album cover art from song lyrics and/or audio using audio feature extraction, NLP-based lyric analysis, and Stable Diffusion image generation.

![Python](https://img.shields.io/badge/python-3.10-blue)
![Streamlit](https://img.shields.io/badge/frontend-Streamlit-ff4b4b)
![FastAPI](https://img.shields.io/badge/backend-FastAPI-009688)
![Status](https://img.shields.io/badge/status-working%20locally-brightgreen)

---

## Overview

This project was built for **DA 627 – Building Multimodal GenAI Systems**. It takes song lyrics (required) and audio (optional) as input, extracts musical and lyrical features, turns them into a structured text prompt, and feeds that prompt into a lightweight diffusion model to generate album cover artwork — all through a simple Streamlit UI.

A full write-up of the design and results is available in [`Report_Prakhar_Punj_220150011.pdf`](./Report_Prakhar_Punj_220150011.pdf).

## Table of Contents

- [Features](#features)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Running the App](#running-the-app)
- [Usage](#usage)
- [Configuration](#configuration)
- [Performance](#performance)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)
- [Acknowledgments](#acknowledgments)
- [License](#license)

## Features

- 🎧 **Audio analysis** — tempo, energy, spectral texture, and mood extraction via Librosa
- 📝 **Lyrics analysis** — sentiment classification and theme/keyword extraction
- 🧠 **Automatic prompt construction** — combines audio + lyrics insights into a single diffusion prompt
- 🖼️ **Fast image generation** — lightweight diffusion model, tuned for consumer GPUs
- 🎛️ **Adjustable parameters** — art style, mood intensity, color temperature, lighting, guidance scale, and number of variations
- 💾 **PNG export** of generated covers

## How It Works

```
Lyrics (required) + Audio (optional)
            │
            ▼
  ┌───────────────────────┐
  │   Audio Processing     │  Librosa → tempo, energy, mood
  └───────────────────────┘
            │
  ┌───────────────────────┐
  │  Lyrics Processing      │  DistilBERT (sentiment) + MiniLM (embeddings)
  └───────────────────────┘
            │
  ┌───────────────────────┐
  │   Prompt Generator      │  Combines insights + UI params into a prompt
  └───────────────────────┘
            │
  ┌───────────────────────┐
  │   Image Generator       │  Diffusers pipeline → album cover (PNG)
  └───────────────────────┘
```

The frontend (Streamlit) talks to the backend (FastAPI) over HTTP on localhost. If the backend isn't reachable, the frontend falls back to running the generation pipeline in-process.

### Models used

| Component | Model | Notes |
|---|---|---|
| Image generation | [`nota-ai/bk-sdm-small`](https://huggingface.co/nota-ai/bk-sdm-small) (default) | ~2.4 GB, swappable via `SD_MODEL_ID` env var for higher-quality/heavier models |
| Sentiment analysis | `distilbert-base-uncased-finetuned-sst-2-english` | Lightweight BERT variant |
| Text embeddings | `sentence-transformers/all-MiniLM-L6-v2` | 33M params, used for theme/semantic extraction |
| Audio features | Librosa | Signal-processing based, no model download needed |

> The default generation model is optimized for low-VRAM GPUs (tested on an RTX 3050, 4 GB VRAM). You can point `SD_MODEL_ID` at a larger model if you have more VRAM to spare.

## Project Structure

```
.
├── app.py                   # Thin entrypoint wrapper (calls frontend.app.main)
├── check_gpu.py              # GPU/CUDA diagnostics script
├── requirements.txt           # Python dependencies
├── .streamlit/                # Streamlit configuration
├── backend/
│   ├── main.py                # FastAPI app: /health, /generate, pipeline orchestration
│   ├── audio_processing.py    # Librosa-based feature extraction
│   ├── lyrics_processing.py   # Sentiment analysis + theme/keyword extraction
│   ├── prompt_generator.py    # Combines audio + lyrics insight into a prompt
│   └── image_generator.py     # Diffusers pipeline wrapper
└── frontend/
    └── app.py                 # Streamlit UI
```

## Getting Started

### Prerequisites

- Python 3.10
- ~3–5 GB free disk space (first run downloads the diffusion model)
- NVIDIA GPU with CUDA recommended for fast generation (works on CPU too, just slower)

### Installation

```bash
# Clone the repo
git clone https://github.com/Guardian-Angel1/GenAI-Song-Lyrics-to-Album-Cover-Generator.git
cd <folder>

# Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### GPU setup (optional but recommended)

`requirements.txt` pulls a CPU-only build of PyTorch by default on Windows. To enable GPU acceleration:

```bash
# Check your GPU and driver
nvidia-smi

# Reinstall PyTorch with a CUDA build matching your driver
pip uninstall torch torchaudio -y
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Adjust the `cu124` suffix (`cu118`, `cu121`, `cu126`, etc.) to match the CUDA version your GPU driver supports. Verify with:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Or simply run:

```bash
python check_gpu.py
```

## Running the App

The backend and frontend run as two separate processes. **Both must be launched from the project root** (not from inside `backend/` or `frontend/`), since imports are package-style (`from backend.xxx import yyy`).

**Terminal 1 — backend:**
```bash
venv\Scripts\activate
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

**Terminal 2 — frontend:**
```bash
venv\Scripts\activate
streamlit run frontend/app.py
```

- Backend: `http://127.0.0.1:8000`
- Frontend: `http://localhost:8501`

If the backend isn't running, the frontend will still work by falling back to a local (in-process) pipeline call.

## Usage

1. **Enter lyrics** (required) — a few lines work best
2. **Upload audio** (optional, WAV/MP3) — tempo, energy, and mood are auto-detected
3. **Tune generation parameters:**
   - Art style: Realistic, Anime, Cyberpunk, Abstract, Vintage, Minimalist
   - Mood intensity: 0.0–1.0
   - Color temperature: Warm, Cool, Neon, Monochrome
   - Lighting: Dark, Bright, Sunset, Night
   - Guidance scale: 1.0–15.0 (default 7.0; 8–9 gives tighter prompt adherence, avoid going much above ~12–13)
   - Variations: 1–4 (capped at 2 on GPU to stay within a 4 GB VRAM budget)
4. **Generate** and download the resulting PNG(s)

## Configuration

Override the default diffusion model via environment variable:

```bash
export SD_MODEL_ID="runwayml/stable-diffusion-v1-5"   # or another Diffusers-compatible model
streamlit run frontend/app.py
```

Larger models increase quality at the cost of speed and VRAM — check it fits your GPU's budget at your chosen resolution/variation count.

## Performance

| Hardware | Time (1 image, 512×512) |
|---|---|
| CPU (i7/i9) | ~90–120s |
| RTX 3050 (4 GB) | ~15–20s (expected) |
| Higher-end GPUs | Faster, roughly scaling with VRAM/compute |

CPU timing above is measured directly; GPU timing is the expected range for this hardware class and may vary based on driver/CUDA setup.

## Troubleshooting

**`ModuleNotFoundError: No module named 'backend'`**
You likely ran a command from inside `backend/` or `frontend/`. Always run from the project root.

**Backend seems to do nothing when started**
Use `python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload` — there's no bare `python backend/main.py` entrypoint.

**Frontend says "FastAPI backend is not running and local import failed"**
Make sure the backend is started first (from the project root), then start the frontend (also from the project root).

**GPU not detected**
Run `python check_gpu.py`. If a GPU is present but not detected, your `torch` install is likely CPU-only — see [GPU setup](#gpu-setup-optional-but-recommended) above.

**Slow generation despite having a GPU**
Confirm GPU utilization with `nvidia-smi`, close other GPU-heavy applications, and re-verify `torch.cuda.is_available()`.

**Cosmetic startup warning about `torch.classes`**
A harmless interaction between Streamlit's file watcher and PyTorch internals. Safe to ignore, or suppress with `streamlit run frontend/app.py --server.fileWatcherType none`.

## Roadmap

- [ ] Genre classification from audio
- [ ] Multi-language lyrics support
- [ ] Text overlay / typography customization on generated covers
- [ ] Generation history and metadata storage
- [ ] Batch processing for multiple songs
- [ ] Deployment (e.g., single-process Hugging Face Spaces build)

## Acknowledgments

- [Streamlit](https://streamlit.io/) — web app framework
- [Hugging Face Diffusers](https://github.com/huggingface/diffusers) — diffusion model pipeline
- [Librosa](https://librosa.org/) — audio processing
- [PyTorch](https://pytorch.org/) — deep learning framework
