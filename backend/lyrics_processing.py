from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline


@dataclass
class LyricsAnalysisResult:
    sentiment_label: str
    sentiment_score: float
    themes: List[str]
    summary_keywords: List[str]
    embedding_preview: List[float]


THEME_KEYWORDS: Dict[str, List[str]] = {
    "love": ["love", "heart", "kiss", "romance", "forever"],
    "heartbreak": ["goodbye", "break", "broken", "tears", "alone"],
    "party": ["dance", "club", "tonight", "party", "lights"],
    "dark": ["shadow", "night", "pain", "blood", "void"],
    "dreamy": ["dream", "sky", "float", "stars", "cloud"],
    "rebellion": ["fire", "fight", "riot", "rage", "freedom"],
    "nostalgia": ["memory", "yesterday", "old", "home", "child"],
}


@lru_cache(maxsize=1)
def get_sentiment_pipeline():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")


@lru_cache(maxsize=1)
def get_embedding_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def extract_themes(text: str) -> List[str]:
    lowered = text.lower()
    themes = [theme for theme, words in THEME_KEYWORDS.items() if any(word in lowered for word in words)]
    return themes[:4] if themes else ["introspection"]


def extract_keywords(text: str) -> List[str]:
    tokens = re.findall(r"[a-zA-Z']{4,}", text.lower())
    filtered = [token for token in tokens if token not in {"this", "that", "with", "your", "from", "into"}]
    ranked = []
    seen = set()
    for token in filtered:
        if token not in seen:
            seen.add(token)
            ranked.append(token)
        if len(ranked) == 6:
            break
    return ranked or ["emotion", "melody"]


def analyze_lyrics(lyrics: str) -> LyricsAnalysisResult:
    normalized = _normalize_text(lyrics)
    if not normalized:
        raise ValueError("Lyrics text is empty.")

    sentiment_result = get_sentiment_pipeline()(normalized[:512], truncation=True)[0]
    embedding = get_embedding_model().encode(normalized)
    embedding_preview = np.asarray(embedding[:8], dtype=float).round(4).tolist()

    label = sentiment_result["label"].lower()
    if label == "positive":
        sentiment_label = "positive"
    elif label == "negative":
        sentiment_label = "negative"
    else:
        sentiment_label = "neutral"

    return LyricsAnalysisResult(
        sentiment_label=sentiment_label,
        sentiment_score=float(sentiment_result["score"]),
        themes=extract_themes(normalized),
        summary_keywords=extract_keywords(normalized),
        embedding_preview=embedding_preview,
    )
