# Song-to-Album Cover Generator

A multimodal GenAI project that generates creative album cover concepts from audio and lyrics using state-of-the-art diffusion models and transformers.

## Project Overview

Song-to-Album Cover Generator combines audio analysis, NLP sentiment analysis, and image generation to create artistic album covers from musical input. The system analyzes musical features (tempo, energy, mood) and lyrical themes to craft detailed prompts for a lightweight diffusion model, resulting in high-quality album artwork generated in 10-20 seconds.

### System Capabilities

- Audio Analysis: Extract tempo, energy, spectral texture, and mood from MP3/WAV files
- Lyrics Analysis: Sentiment analysis, theme detection, and semantic embeddings
- Prompt Generation: Combine audio and lyrical insights into detailed image prompts
- Fast Generation: 10-20 seconds per image on GPU, optimized for consumer hardware
- Customizable Parameters: Adjust art style, mood, colors, lighting, and generation parameters
- Output Export: High-quality PNG album covers

---

## Quick Start

### Prerequisites
- Python 3.9+
- 4GB RAM minimum (GPU recommended for speed)
- Internet connection (first run downloads ~3.5GB model)

### Installation & Running

**Step 1: Setup Environment**
```bash
# Navigate to project
cd "d:\CPi\8th Sem\DA 627 Building Multimodal GenAI\Course Project\Course Project"

# Create virtual environment
python -m venv venv

# Activate it
venv\Scripts\activate
```

**Step 2: Install Dependencies**
```bash
pip install -r requirements.txt
```

**Step 3: (GPU Users Only) Install PyTorch with CUDA**
```bash
# Check GPU:
nvidia-smi

# If you have NVIDIA GPU, reinstall PyTorch:
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Step 4: Run the Application**
```bash
streamlit run app.py --server.fileWatcherType none
```

The application opens at `http://localhost:8501`

---

## Usage Instructions

### Web Interface

The Streamlit interface provides controls for album cover generation:

1. **Lyrics Input** (required)
   - Text field for song lyrics (2-4 lines recommended)

2. **Audio Upload** (optional)
   - Accepts WAV or MP3 audio files
   - System automatically extracts tempo, energy, and mood

3. **Generation Parameters**
   - Art Style: Realistic, Anime, Cyberpunk, Abstract, Vintage, Minimalist
   - Mood Intensity: 0.0 to 1.0 scale
   - Color Temperature: Warm, Cool, Neon, Monochrome
   - Lighting Style: Dark, Bright, Sunset, Night
   - Guidance Scale: 1.0 to 15.0 (default: 7.0)
   - Variations: 1 to 4 images per generation

4. **Generation Process**
   - Typical generation time: 10-20 seconds (GPU), 30-120 seconds (CPU)
   - Progress indicator displays during processing

5. **Output Management**
   - Generated images available for download as PNG files
   - Metadata and analysis data provided for reference

---

## AI Model Specifications

### Image Generation Model: **SSD-1B (Segmind's Stable Diffusion 1B)**

The project uses **SSD-1B**, a distilled variant of Stable Diffusion specifically optimized for speed and efficiency.

#### Technical Specifications
- **Model Size:** 3.5 GB (vs 7GB for Stable Diffusion v1.5)
- **Architecture:** Distilled from Stable Diffusion 2.1
- **Inference Steps:** 16 steps (vs 30-50 in standard SD)
- **Precision:** float16 on GPU, float32 on CPU
- **VRAM Usage:** ~3-4 GB on GPU
- **Generation Speed:** 10-20 seconds per 512×512 image on RTX 3050+

#### Why SSD-1B > Other Models

| Aspect | SSD-1B | SD v1.5 | SD v2.1 | SD XL |
|--------|--------|---------|---------|-------|
| **Model Size** | 3.5 GB | 4-7 GB | 5-7 GB | 6-7 GB |
| **Speed** | Fast | Medium | Slow | Very Slow |
| **Quality** | Very Good | Good | Very Good | Excellent |
| **VRAM (GPU)** | 3-4 GB | 6-8 GB | 6-8 GB | 8-10 GB |
| **Best For** | **Fast, Consumer GPUs** | Legacy | High-quality | Enterprise |

**SSD-1B Selection Rationale:**
1. Optimized model architecture for computational efficiency
2. Quality improvements over Stable Diffusion v1.5 through Stable Diffusion 2.1 foundation
3. Consumer GPU compatibility with 4GB VRAM requirements
4. 16 inference steps sufficient with modern scheduler implementations
5. Optimal balance between generation speed and output quality

### Text Analysis Models

#### Sentiment Analysis: **DistilBERT** (distilbert-base-uncased-finetuned-sst-2-english)
- Lightweight BERT variant, 40% smaller than BERT
- Fast sentiment classification (positive/negative/neutral)
- Pre-trained on SST-2 sentiment corpus

#### Embeddings: **Sentence-Transformers MiniLM** (sentence-transformers/all-MiniLM-L6-v2)
- 33M parameters (efficient for CPU)
- Semantic similarity for theme extraction
- Pre-trained on 215M sentence pairs

#### Audio Analysis: **Librosa**
- Open-source audio processing library
- Fast feature extraction: tempo, energy, spectral centroid, zero-crossing rate
- No ML model required - signal processing based

---

## Performance Metrics

### Generation Times (512×512 images, 16 steps)

| Hardware | Time | Speed |
|----------|------|-------|
| **RTX 3050 (4GB)** | 12-18s | Fast |
| **RTX 3060 (12GB)** | 8-12s | Fast |
| **RTX 3080 (10GB)** | 5-8s | Very Fast |
| **CPU (i7/i9)** | 60-120s | Slow |

### Memory Usage

- **Model Loading:** ~3.5 GB
- **Generation:** ~0.5-1.0 GB additional
- **Total Needed:** 4 GB minimum (GPU), 8 GB recommended (RAM for CPU)

---

## System Architecture

```
┌─────────────────────────────────────────┐
│        Streamlit Frontend (UI)          │
│  • File upload, sliders, text inputs    │
│  • Real-time visualization              │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│      FastAPI Backend (Optional)         │
│  • Handles requests & file processing   │
│  • Enables multi-user support           │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│     Audio Processing (librosa)          │
│  • Extract: tempo, energy, mood         │
│  • Output: AudioAnalysisResult          │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│   Lyrics Processing (Transformers)      │
│  • Sentiment analysis (DistilBERT)      │
│  • Theme extraction & embeddings        │
│  • Output: LyricsAnalysisResult         │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│   Prompt Generation (Rule-based)        │
│  • Combine insights into detailed prompt│
│  • Apply style/mood/color preferences   │
│  • Output: Structured prompt text       │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│   Image Generation (SSD-1B Diffusion)   │
│  • 16 inference steps                   │
│  • GPU-optimized with xFormers          │
│  • Output: 512×512 PNG images           │
└─────────────────────────────────────────┘
```

---

## Project Structure

```
Course Project/
├── backend/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application & endpoints
│   ├── audio_processing.py     # Librosa-based audio analysis
│   ├── lyrics_processing.py    # DistilBERT sentiment & embeddings
│   ├── prompt_generator.py     # Combine insights into prompts
│   └── image_generator.py      # SSD-1B diffusion pipeline
├── frontend/
│   ├── __init__.py
│   └── app.py                  # Streamlit UI
├── app.py                      # Entry point
├── check_gpu.py                # GPU diagnostics script
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── GPU_SETUP.md               # GPU configuration guide
├── SETUP_GUIDE.md             # Detailed setup instructions
├── QUICK_START.md             # 2-minute quick start
├── OPTIMIZATION_SUMMARY.md    # Performance optimization details
├── MODEL_ALTERNATIVES.md      # Alternative model information
└── PROJECT_REPORT.md          # Comprehensive project report
```

---

## Advanced Configuration

### Custom Model Configuration

Alternative models can be specified via environment variables:
```bash
export SD_MODEL_ID="dreamshaper-8"
streamlit run app.py
```
Note: Custom models must be compatible with Hugging Face Diffusers format.

### GPU Diagnostics and Verification

System diagnostic information can be obtained:
```bash
python check_gpu.py
```
Provides VRAM availability, CUDA version, GPU capabilities, and configuration recommendations.

### Multiple Variations

Multiple variations can be requested for comparison:
- Two variations: approximately 20-40 seconds generation time
- Four variations: approximately 40-80 seconds generation time
- Guidance scale parameter controls output detail level (recommended range: 7-15)

### Deterministic Generation

Repeatability can be achieved by specifying a seed value:
- Identical seed and input parameters produce identical output
- Useful for testing and consistency verification

---

## Model Selection Justification

### Why Not Use Other Models?

**Stable Diffusion v1.5**
- Requires 30+ inference steps
- Model size: 7GB (vs 3.5GB for SSD-1B)
- Not optimized for consumer hardware with limited VRAM

**Stable Diffusion XL**
- Very large model (6-7GB)
- Slow performance on consumer GPUs
- Designed for high-end systems

**Leonardo Diffusion**
- Closed-source implementation
- API-based access requiring subscription

**LocalDiffusion / TinySD**
- Limited quality output
- Insufficient for professional album cover generation

**SSD-1B (Selected Model)**
- Optimal balance of speed and quality
- Optimized for consumer GPU hardware
- Based on improved Stable Diffusion 2.1 architecture
- 3.5GB model size (manageable on 4GB VRAM systems)
- 16-step generation sufficient for high-quality output
- Best suited for album cover generation requirements

---

## Generation Workflow

```
1. Lyrics Input
   Input: "Dancing in the night, feeling so alive"
   
2. Audio Feature Analysis (Optional)
   Tempo: 120 BPM
   Energy Level: 0.75
   Detected Mood: Happy
   
3. Lyrics Semantic Analysis
   Sentiment Classification: Positive
   Detected Themes: Party, Dance, Joy
   Key Terms: Dancing, Night, Alive, Feeling
   
4. Prompt Composition
   Synthesized Prompt: "Emotionally expressive album cover, happy mood
   atmosphere, realistic composition, cool color palette, bright lighting,
   fast tempo, high energy, dance party theme"
   
5. Image Generation
   Model: SSD-1B Diffusion
   Steps: 16
   Output Resolution: 512×512 pixels
   Format: PNG
   
6. Final Deliverables
   Generated album cover image
   Generation prompt documentation
   Processing metadata
```

---

## Performance Optimizations

1. Model Distillation: SSD-1B architecture provides reduced computational requirements
2. Inference Steps: Configured to 16 steps for optimal convergence
3. Precision: Float16 on GPU for reduced memory consumption
4. Memory Optimization: Attention slicing and VAE slicing for consumer hardware
5. Lightweight NLP: DistilBERT for efficient sentiment analysis
6. Efficient Embeddings: MiniLM for semantic representation

---

## Troubleshooting

### Generation Failure
If generation process fails:
1. Reduce number of variations to 1
2. Lower guidance scale to 7.0
3. Restart application
4. Clear model cache: `Remove-Item -Recurse $env:USERPROFILE\.cache\huggingface`

### GPU Detection Issues
If GPU is not detected:
```bash
python check_gpu.py
```
If GPU is available but not detected, reinstall PyTorch with CUDA support:
```bash
pip uninstall torch -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Slow Generation Performance
If generation is slower than expected:
- Verify GPU utilization: `nvidia-smi`
- Close competing GPU-intensive applications
- Confirm PyTorch GPU support installation

### Memory Constraints
If out-of-memory errors occur:
- Reduce number of image variations
- Lower guidance scale parameter
- Use CPU processing (slower but requires less VRAM)

---

## Supporting Documentation

- **GPU_SETUP.md** - GPU configuration and installation procedures
- **SETUP_GUIDE.md** - Detailed system setup instructions
- **QUICK_START.md** - Quick reference guide
- **OPTIMIZATION_SUMMARY.md** - Performance optimization details
- **MODEL_ALTERNATIVES.md** - Comparison of alternative model options
- **PROJECT_REPORT.md** - Comprehensive technical project documentation

---

## Software Dependencies

Primary dependencies:
- PyTorch 2.0+ - Deep learning framework
- Diffusers 0.32+ - Diffusion model library
- Transformers 4.48+ - NLP model implementations
- Sentence-Transformers - Semantic embedding models
- Librosa 0.10+ - Audio signal processing
- Streamlit 1.42+ - Web application framework
- FastAPI 0.115+ - REST API framework (optional)

---

## Planned Enhancements

- Genre classification from audio analysis
- Multi-language lyrics support
- Real-time audio stream input processing
- Text overlay and typography customization
- Generation history database and metadata storage
- Batch processing for multiple songs
- Music platform integration (Spotify, SoundCloud)
- Support for additional diffusion models
- Advanced prompt refinement using language models

---

## Acknowledgments

- Streamlit - Web application framework
- Hugging Face Diffusers - Diffusion model implementations
- SSD-1B Model - Segmind Stable Diffusion 1B
- Librosa - Audio processing library
- PyTorch - Deep learning framework

---

## Support and Diagnostics

For technical issues:
1. Consult the Troubleshooting section in this document
2. Review GPU_SETUP.md for GPU-specific configuration
3. Execute `python check_gpu.py` for system diagnostics
4. Review terminal output and logs for detailed error information


