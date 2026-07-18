from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import base64
import os
import tempfile
import warnings
from io import BytesIO
from typing import Any, Dict, List, Optional

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*symlinks.*")
warnings.filterwarnings("ignore", message=".*Xet Storage.*")

import requests
import streamlit as st
from PIL import Image

# Try API first, fall back to local import
try:
    from backend.main import run_generation_pipeline
    _LOCAL_IMPORT_OK = True
except ImportError:
    _LOCAL_IMPORT_OK = False


BACKEND_URL = "http://127.0.0.1:8000"
ART_STYLES = ["Realistic", "Anime", "Cyberpunk", "Abstract", "Vintage", "Minimalist"]
COLOR_TEMPERATURES = ["Warm", "Cool", "Neon", "Monochrome"]
LIGHTING_STYLES = ["Dark", "Bright", "Sunset", "Night"]


def check_backend() -> bool:
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=2)
        return response.ok
    except requests.RequestException:
        return False


def decode_images(images_base64: List[str]) -> List[Image.Image]:
    images: List[Image.Image] = []
    for encoded in images_base64:
        image_bytes = base64.b64decode(encoded)
        images.append(Image.open(BytesIO(image_bytes)))
    return images


def generate_via_api(
    audio_bytes: Optional[bytes],
    audio_name: Optional[str],
    lyrics: str,
    art_style: str,
    mood_intensity: float,
    color_temperature: str,
    guidance_scale: float,
    num_variations: int,
    seed: Optional[int],
    lighting_style: str,
) -> Dict[str, Any]:
    data = {
        "lyrics": lyrics,
        "art_style": art_style,
        "mood_intensity": str(mood_intensity),
        "color_temperature": color_temperature,
        "guidance_scale": str(guidance_scale),
        "num_variations": str(num_variations),
        "lighting_style": lighting_style,
    }
    if seed is not None:
        data["seed"] = str(seed)
    files = {}
    if audio_bytes and audio_name:
        files["audio_file"] = (audio_name, audio_bytes)

    response = requests.post(f"{BACKEND_URL}/generate", data=data, files=files, timeout=2400)
    response.raise_for_status()
    payload = response.json()
    payload["images"] = decode_images(payload.get("images_base64", []))
    return payload


def generate_locally(
    audio_bytes: Optional[bytes],
    audio_name: Optional[str],
    lyrics: str,
    art_style: str,
    mood_intensity: float,
    color_temperature: str,
    guidance_scale: float,
    num_variations: int,
    seed: Optional[int],
    lighting_style: str,
) -> Dict[str, Any]:
    if not _LOCAL_IMPORT_OK:
        raise RuntimeError(
            "FastAPI backend is not running and local import failed. "
            "Start the backend: uvicorn backend.main:app --reload"
        )
    temp_path = None
    try:
        if audio_bytes and audio_name:
            suffix = f".{audio_name.rsplit('.', 1)[-1]}" if "." in audio_name else ".wav"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(audio_bytes)
                temp_path = tmp.name

        return run_generation_pipeline(
            art_style=art_style,
            mood_intensity=mood_intensity,
            color_temperature=color_temperature,
            guidance_scale=guidance_scale,
            num_variations=num_variations,
            lighting_style=lighting_style,
            lyrics=lyrics,
            audio_path=temp_path,
            seed=seed,
        )
    finally:
        if temp_path:
            try:
                os.unlink(temp_path)
            except OSError:
                pass


def render_analysis(label: str, payload: Optional[Dict[str, Any]]) -> None:
    with st.expander(label, expanded=False):
        if not payload:
            st.info(f"No {label.lower()} available.")
        else:
            st.json(payload)


def main() -> None:
    st.set_page_config(
        page_title="Song-to-Album Cover Generator",
        page_icon="🎵",
        layout="wide",
    )
    st.title("🎵 Song-to-Album Cover Generator")
    st.caption("Generate album cover concepts from audio and lyrics using multimodal AI.")

    with st.sidebar:
        st.header("🎨 Creative Controls")
        art_style = st.selectbox("Art Style", ART_STYLES)
        mood_intensity = st.slider("Mood Intensity", 0.0, 1.0, 0.65, 0.05)
        color_temperature = st.selectbox("Color Temperature", COLOR_TEMPERATURES)
        lighting_style = st.selectbox("Lighting Style", LIGHTING_STYLES)
        guidance_scale = st.slider(
            "Guidance Scale",
            min_value=1.0, max_value=15.0, value=7.0, step=0.5,
            help="7.0 is recommended for 4GB VRAM. Higher values use more memory.",
        )
        # Default to 1 variation to stay safe on 4GB VRAM
        num_variations = st.slider(
            "Variations",
            min_value=1, max_value=2, value=1, step=1,
            help="Keep at 1 for 4GB VRAM. Max 2 supported.",
        )
        seed_input = st.text_input("Seed (optional)", placeholder="Leave blank for random")

        st.divider()
        use_api = check_backend()
        if use_api:
            st.success("✅ FastAPI backend running")
        else:
            st.warning("⚠️ No FastAPI backend — running locally")

    col1, col2 = st.columns([1, 1])

    with col1:
        audio_file = st.file_uploader("Upload Audio (MP3 / WAV)", type=["mp3", "wav"])
        lyrics = st.text_area(
            "Lyrics (optional)",
            height=220,
            placeholder="Paste song lyrics here, or leave blank to use audio only.",
        )

    with col2:
        st.markdown(
            """
            ### How it works
            1. **Audio** features (tempo, energy, mood) are extracted with `librosa`.
            2. **Lyrics** are analyzed for sentiment, themes, and semantic keywords.
            3. Both signals are merged into a structured **Stable Diffusion prompt**.
            4. `nota-ai/bk-sdm-small` generates square album cover concepts (~2.4GB VRAM).

            **Tips for 4GB VRAM:**
            - Keep Variations at **1**
            - Guidance Scale at **7.0**
            - Restart Streamlit between runs if you hit memory errors
            """
        )

    generate_clicked = st.button(
        "🎨 Generate Album Cover", type="primary", use_container_width=True
    )

    if generate_clicked:
        if not audio_file and not lyrics.strip():
            st.error("Upload an audio file, enter lyrics, or provide both.")
            return

        seed: Optional[int] = int(seed_input.strip()) if seed_input.strip().isdigit() else None
        audio_bytes = audio_file.getvalue() if audio_file else None
        audio_name = audio_file.name if audio_file else None

        try:
            with st.spinner("Analyzing music and generating album cover…"):
                if use_api:
                    result = generate_via_api(
                        audio_bytes=audio_bytes,
                        audio_name=audio_name,
                        lyrics=lyrics,
                        art_style=art_style,
                        mood_intensity=mood_intensity,
                        color_temperature=color_temperature,
                        guidance_scale=guidance_scale,
                        num_variations=num_variations,
                        seed=seed,
                        lighting_style=lighting_style,
                    )
                else:
                    result = generate_locally(
                        audio_bytes=audio_bytes,
                        audio_name=audio_name,
                        lyrics=lyrics,
                        art_style=art_style,
                        mood_intensity=mood_intensity,
                        color_temperature=color_temperature,
                        guidance_scale=guidance_scale,
                        num_variations=num_variations,
                        seed=seed,
                        lighting_style=lighting_style,
                    )
        except requests.HTTPError as exc:
            try:
                detail = exc.response.json().get("detail", str(exc))
            except Exception:
                detail = str(exc)
            st.error(f"🔴 Generation Failed:\n{detail}")
            st.info(
                "**Quick fixes:**\n"
                "1. Set Variations to **1**\n"
                "2. Set Guidance Scale to **7.0**\n"
                "3. Restart Streamlit: `Ctrl+C` → `streamlit run app.py --server.fileWatcherType none`"
            )
            return
        except Exception as exc:
            st.error(f"🔴 Generation Failed:\n{exc}")
            st.info(
                "**Quick fixes:**\n"
                "1. Set Variations to **1**\n"
                "2. Restart Streamlit to flush GPU memory\n"
                "3. Make sure you're using `nota-ai/bk-sdm-small` (default)"
            )
            return

        st.subheader("Generated Prompt")
        st.code(result.get("prompt", ""), language="text")

        render_analysis("Audio Analysis", result.get("audio_analysis"))
        render_analysis("Lyrics Analysis", result.get("lyrics_analysis"))

        images = result.get("images", [])
        if not images:
            st.warning("No images were returned.")
            return

        st.subheader("Generated Album Covers")
        cols = st.columns(len(images))
        for idx, image in enumerate(images):
            with cols[idx]:
                st.image(image, caption=f"Variation {idx + 1}", use_container_width=True)
                buf = BytesIO()
                image.save(buf, format="PNG")
                st.download_button(
                    label=f"⬇️ Download Variation {idx + 1}",
                    data=buf.getvalue(),
                    file_name=f"album_cover_{idx + 1}.png",
                    mime="image/png",
                    use_container_width=True,
                )


if __name__ == "__main__":
    main()
