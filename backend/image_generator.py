from __future__ import annotations

import gc
import io
import os
import warnings
from functools import lru_cache
from typing import List, Optional

import torch
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
from PIL import Image


# BK-SDM-Small: distilled SD model, ~2.4GB VRAM, fast on 4GB GPUs
# Fallback: runwayml/stable-diffusion-v1-5 (~3.0GB)
DEFAULT_MODEL_ID = os.getenv("SD_MODEL_ID", "nota-ai/bk-sdm-small")


class ImageGenerationError(Exception):
    """Raised when image generation fails."""


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_vram_usage() -> str:
    if not torch.cuda.is_available():
        return "N/A (CPU mode)"
    allocated = torch.cuda.memory_allocated() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    return f"{allocated:.2f}GB / {total:.2f}GB (Reserved: {reserved:.2f}GB)"


def clear_pipeline_cache() -> None:
    get_pipeline.cache_clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


@lru_cache(maxsize=1)
def get_pipeline(model_id: str = DEFAULT_MODEL_ID) -> StableDiffusionPipeline:
    device = get_device()
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"\n📦 Loading model: {model_id}")
    print(f"🖥️  Device: {device.upper()} | 📊 dtype: {dtype}")

    try:
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            safety_checker=None,          # disable to free VRAM and prevent NoneType crash
            requires_safety_checker=False,
        )
    except Exception as exc:
        raise ImageGenerationError(
            f"Failed to load model '{model_id}'. "
            f"Check your internet connection and HuggingFace access. Error: {exc}"
        ) from exc

    # Euler Ancestral gives good quality at low step counts
    pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipeline.scheduler.config
    )

    pipeline = pipeline.to(device)
    pipeline.enable_attention_slicing(1)  # slice=1 is most memory-efficient
    pipeline.enable_vae_slicing()

    if device == "cuda":
        # xformers is optional — skip silently if absent
        if hasattr(pipeline, "enable_xformers_memory_efficient_attention"):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pipeline.enable_xformers_memory_efficient_attention()
            except Exception:
                pass
        print(f"✅ GPU ready — VRAM: {get_vram_usage()}")
    else:
        print("⚠️  No GPU detected — running on CPU (slow)")

    return pipeline


def generate_images(
    prompt: str,
    negative_prompt: str,
    guidance_scale: float,
    num_images: int,
    seed: Optional[int] = None,
    width: int = 512,
    height: int = 512,
) -> List[Image.Image]:
    """Generate album cover images using Stable Diffusion."""
    device = get_device()

    # On 4GB VRAM cap to 1 image at a time to avoid OOM
    if device == "cuda" and num_images > 1:
        num_images = min(num_images, 2)

    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    print(f"\n🎨 Generating {num_images} image(s)...")
    print(f"📝 Prompt: {prompt[:120]}...")
    print(f"⚙️  guidance={guidance_scale}, steps=20, size={width}x{height}")

    try:
        pipeline = get_pipeline()

        generator = None
        if seed is not None:
            generator = torch.Generator(device=device).manual_seed(int(seed))
            print(f"🔒 Seed: {seed}")

        with torch.no_grad():
            result = pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                guidance_scale=float(guidance_scale),
                num_inference_steps=30,          # 20 steps is sweet spot for quality/speed
                num_images_per_prompt=int(num_images),
                height=int(height),
                width=int(width),
                generator=generator,
            )

        # Robust image extraction — handles both object and tuple returns
        if hasattr(result, "images") and result.images is not None:
            images = result.images
        elif isinstance(result, (list, tuple)) and len(result) > 0:
            images = result[0]
        else:
            raise ImageGenerationError(
                "Pipeline returned no images. Try restarting Streamlit."
            )

        if not images:
            raise ImageGenerationError(
                "Empty image list returned. Try a different seed or lower guidance scale."
            )

        print(f"✅ Generated {len(images)} image(s) — VRAM: {get_vram_usage()}")
        return list(images)

    except ImageGenerationError:
        raise
    except RuntimeError as exc:
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        msg = str(exc).lower()
        if "out of memory" in msg:
            raise ImageGenerationError(
                f"GPU OOM — VRAM: {get_vram_usage()}\n"
                "Fix: set Variations to 1, restart Streamlit."
            ) from exc
        raise ImageGenerationError(f"Runtime error: {exc}") from exc
    except Exception as exc:
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        raise ImageGenerationError(
            f"Generation failed: {exc}\nVRAM: {get_vram_usage()}"
        ) from exc
    finally:
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()


def image_to_png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
