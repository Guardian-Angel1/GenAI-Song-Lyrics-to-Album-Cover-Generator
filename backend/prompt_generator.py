from __future__ import annotations

from typing import Any, Dict, List, Optional

from backend.audio_processing import AudioAnalysisResult
from backend.lyrics_processing import LyricsAnalysisResult


STYLE_DESCRIPTORS = {
    "Realistic": "photorealistic composition, premium album artwork, cinematic detail",
    "Anime": "anime-inspired illustration, expressive linework, polished shading",
    "Cyberpunk": "cyberpunk aesthetic, futuristic city textures, holographic atmosphere",
    "Abstract": "abstract visual language, layered symbolism, artistic composition",
    "Vintage": "vintage poster art, textured grain, timeless analog feel",
    "Minimalist": "minimalist cover design, clean geometry, strong focal subject",
}

COLOR_DESCRIPTORS = {
    "Warm": "warm color palette, amber, crimson, gold highlights",
    "Cool": "cool color palette, blue, teal, silver hues",
    "Neon": "neon glow, electric magenta and cyan accents",
    "Monochrome": "monochrome palette, tonal contrast, restrained colors",
}

LIGHTING_DESCRIPTORS = {
    "Dark": "low-key lighting, dramatic shadows",
    "Bright": "bright lighting, vivid highlights",
    "Sunset": "sunset lighting, golden hour atmosphere",
    "Night": "night scene, moody artificial light",
}


def _intensity_phrase(mood_intensity: float) -> str:
    if mood_intensity >= 0.8:
        return "emotionally intense"
    if mood_intensity >= 0.5:
        return "emotionally expressive"
    return "subtly emotional"


def _build_subject_tokens(
    audio_result: Optional[AudioAnalysisResult],
    lyrics_result: Optional[LyricsAnalysisResult],
) -> List[str]:
    tokens: List[str] = []
    if audio_result:
        tokens.extend(
            [
                f"{audio_result.mood} mood",
                audio_result.descriptors["tempo_descriptor"],
                audio_result.descriptors["energy_descriptor"],
                audio_result.descriptors["texture_descriptor"],
            ]
        )
    if lyrics_result:
        tokens.append(f"{lyrics_result.sentiment_label} lyrical tone")
        tokens.extend(lyrics_result.themes[:3])
        tokens.extend(lyrics_result.summary_keywords[:3])
    if not tokens:
        tokens.append("expressive musical storytelling")
    return tokens


def generate_prompt(
    audio_result: Optional[AudioAnalysisResult],
    lyrics_result: Optional[LyricsAnalysisResult],
    art_style: str,
    mood_intensity: float,
    color_temperature: str,
    lighting_style: str,
) -> Dict[str, Any]:
    """Convert multimodal insights into a structured image prompt."""
    subject_tokens = _build_subject_tokens(audio_result, lyrics_result)

    primary_mood = audio_result.mood if audio_result else "evocative"
    if lyrics_result and lyrics_result.sentiment_label == "negative":
        primary_mood = "melancholic"
    elif lyrics_result and lyrics_result.sentiment_label == "positive":
        primary_mood = "uplifting"

    prompt_parts = [
        f"{_intensity_phrase(mood_intensity)} album cover",
        f"{primary_mood} atmosphere",
        STYLE_DESCRIPTORS.get(art_style, STYLE_DESCRIPTORS["Realistic"]),
        COLOR_DESCRIPTORS.get(color_temperature, COLOR_DESCRIPTORS["Cool"]),
        LIGHTING_DESCRIPTORS.get(lighting_style, LIGHTING_DESCRIPTORS["Night"]),
        ", ".join(subject_tokens[:8]),
        "centered composition, square album art, highly detailed, professional cover design",
    ]

    prompt = ", ".join(part for part in prompt_parts if part)
    negative_prompt = (
        "blurry, distorted face, watermark, text, logo, low quality, duplicate elements, "
        "cropped, deformed anatomy, noisy, oversaturated"
    )

    return {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "insights": {
            "audio_mood": audio_result.mood if audio_result else None,
            "lyrics_sentiment": lyrics_result.sentiment_label if lyrics_result else None,
            "themes": lyrics_result.themes if lyrics_result else [],
            "keywords": lyrics_result.summary_keywords if lyrics_result else [],
        },
    }
