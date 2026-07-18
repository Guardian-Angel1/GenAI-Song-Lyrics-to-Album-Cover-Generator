from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.audio_processing import AudioAnalysisResult, AudioProcessingError, analyze_audio
from backend.image_generator import ImageGenerationError, generate_images, image_to_png_bytes
from backend.lyrics_processing import LyricsAnalysisResult, analyze_lyrics
from backend.prompt_generator import generate_prompt


app = FastAPI(title="Song-to-Album Cover Generator", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerationResponse(BaseModel):
    prompt: str
    negative_prompt: str
    audio_analysis: Optional[Dict[str, Any]] = None
    lyrics_analysis: Optional[Dict[str, Any]] = None
    images_base64: List[str]


class HealthResponse(BaseModel):
    status: str = "ok"


def _serialize_audio(result: Optional[AudioAnalysisResult]) -> Optional[Dict[str, Any]]:
    if not result:
        return None
    return {
        "tempo": result.tempo,
        "energy": result.energy,
        "spectral_centroid": result.spectral_centroid,
        "zero_crossing_rate": result.zero_crossing_rate,
        "mood": result.mood,
        "descriptors": result.descriptors,
    }


def _serialize_lyrics(result: Optional[LyricsAnalysisResult]) -> Optional[Dict[str, Any]]:
    if not result:
        return None
    return {
        "sentiment_label": result.sentiment_label,
        "sentiment_score": result.sentiment_score,
        "themes": result.themes,
        "summary_keywords": result.summary_keywords,
        "embedding_preview": result.embedding_preview,
    }


def run_generation_pipeline(
    art_style: str,
    mood_intensity: float,
    color_temperature: str,
    guidance_scale: float,
    num_variations: int,
    lighting_style: str,
    lyrics: Optional[str] = None,
    audio_path: Optional[str] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    if not audio_path and not (lyrics and lyrics.strip()):
        raise ValueError("Provide an audio file, lyrics, or both.")

    audio_result = analyze_audio(audio_path) if audio_path else None
    lyrics_result = analyze_lyrics(lyrics) if lyrics and lyrics.strip() else None

    prompt_payload = generate_prompt(
        audio_result=audio_result,
        lyrics_result=lyrics_result,
        art_style=art_style,
        mood_intensity=mood_intensity,
        color_temperature=color_temperature,
        lighting_style=lighting_style,
    )
    images = generate_images(
        prompt=prompt_payload["prompt"],
        negative_prompt=prompt_payload["negative_prompt"],
        guidance_scale=guidance_scale,
        num_images=num_variations,
        seed=seed,
    )

    return {
        "prompt": prompt_payload["prompt"],
        "negative_prompt": prompt_payload["negative_prompt"],
        "audio_analysis": _serialize_audio(audio_result),
        "lyrics_analysis": _serialize_lyrics(lyrics_result),
        "images": images,
    }


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    return HealthResponse()


@app.post("/generate")
async def generate_album_cover(
    art_style: str = Form(...),
    mood_intensity: float = Form(..., ge=0.0, le=1.0),
    color_temperature: str = Form(...),
    guidance_scale: float = Form(..., ge=1.0, le=20.0),
    num_variations: int = Form(..., ge=1, le=4),
    lighting_style: str = Form(...),
    lyrics: Optional[str] = Form(None),
    seed: Optional[int] = Form(None),
    audio_file: Optional[UploadFile] = File(None),
) -> GenerationResponse:
    temp_path: Optional[Path] = None

    try:
        if audio_file and audio_file.filename:
            suffix = Path(audio_file.filename).suffix or ".wav"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                shutil.copyfileobj(audio_file.file, temp_file)
                temp_path = Path(temp_file.name)

        result = run_generation_pipeline(
            art_style=art_style,
            mood_intensity=mood_intensity,
            color_temperature=color_temperature,
            guidance_scale=guidance_scale,
            num_variations=num_variations,
            lighting_style=lighting_style,
            lyrics=lyrics,
            audio_path=str(temp_path) if temp_path else None,
            seed=seed,
        )
    except (ValueError, AudioProcessingError, ImageGenerationError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink(missing_ok=True)

    import base64

    images_base64 = [base64.b64encode(image_to_png_bytes(image)).decode("utf-8") for image in result["images"]]
    return GenerationResponse(
        prompt=result["prompt"],
        negative_prompt=result["negative_prompt"],
        audio_analysis=result["audio_analysis"],
        lyrics_analysis=result["lyrics_analysis"],
        images_base64=images_base64,
    )
