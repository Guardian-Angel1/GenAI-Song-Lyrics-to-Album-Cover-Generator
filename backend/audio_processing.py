from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict

import librosa
import numpy as np


@dataclass
class AudioAnalysisResult:
    tempo: float
    energy: float
    spectral_centroid: float
    zero_crossing_rate: float
    mood: str
    descriptors: Dict[str, str]


class AudioProcessingError(Exception):
    """Raised when the uploaded audio cannot be processed."""


def _safe_normalize(value: float, minimum: float, maximum: float) -> float:
    if maximum <= minimum:
        return 0.0
    return float(np.clip((value - minimum) / (maximum - minimum), 0.0, 1.0))


def infer_mood(tempo: float, energy: float, spectral_centroid: float) -> str:
    if energy > 0.18 and tempo > 125:
        return "aggressive"
    if energy > 0.12 and spectral_centroid > 2200:
        return "happy"
    if energy < 0.06 and tempo < 95:
        return "calm"
    return "sad"


def analyze_audio(audio_path: str) -> AudioAnalysisResult:
    if not audio_path or not os.path.exists(audio_path):
        raise AudioProcessingError("Audio file was not found.")

    try:
        signal, sample_rate = librosa.load(audio_path, sr=22050, mono=True)
    except Exception as exc:
        raise AudioProcessingError("Unable to read the audio file. Use WAV or MP3.") from exc

    if signal.size == 0:
        raise AudioProcessingError("Audio file is empty or unreadable.")

    duration = librosa.get_duration(y=signal, sr=sample_rate)
    if duration < 1.0:
        raise AudioProcessingError("Audio file is too short. Upload at least 1 second.")

    # librosa >= 0.10 returns (tempo_array, beats); extract scalar safely
    tempo_result = librosa.beat.beat_track(y=signal, sr=sample_rate)
    raw_tempo = tempo_result[0]
    tempo = float(np.atleast_1d(raw_tempo)[0])  # handles both scalar and 1-element array

    rms = librosa.feature.rms(y=signal)[0]
    spectral_centroid = librosa.feature.spectral_centroid(y=signal, sr=sample_rate)[0]
    zcr = librosa.feature.zero_crossing_rate(signal)[0]

    avg_energy = float(np.mean(rms))
    avg_centroid = float(np.mean(spectral_centroid))
    avg_zcr = float(np.mean(zcr))
    mood = infer_mood(tempo, avg_energy, avg_centroid)

    descriptors = {
        "tempo_descriptor": "fast" if tempo >= 120 else "mid-tempo" if tempo >= 90 else "slow",
        "energy_descriptor": (
            "high energy" if avg_energy >= 0.14 else
            "gentle" if avg_energy < 0.07 else "balanced energy"
        ),
        "texture_descriptor": "bright texture" if avg_centroid >= 2500 else "warm texture",
        "rhythm_descriptor": "percussive" if avg_zcr >= 0.12 else "smooth",
    }

    return AudioAnalysisResult(
        tempo=tempo,
        energy=_safe_normalize(avg_energy, 0.0, 0.25),
        spectral_centroid=avg_centroid,
        zero_crossing_rate=avg_zcr,
        mood=mood,
        descriptors=descriptors,
    )
