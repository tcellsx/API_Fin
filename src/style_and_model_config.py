"""
Style and model configuration module.

Defines style presets and model-specific prompt/parameter mappings as described
in `docs/05_style_and_model_config.md`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from .config import GLOBAL_STYLE_CONFIG, StyleConfig


@dataclass
class ModelConfig:
    model_name: str
    prompt: str
    max_duration_sec: Optional[float] = None

    def to_dict(self) -> Dict[str, str | float | None]:
        return {
            "model_name": self.model_name,
            "prompt": self.prompt,
            "max_duration_sec": self.max_duration_sec,
        }


@dataclass
class StylePreset:
    name: str
    description: str
    mood: str
    tempo_bpm: int
    instruments: List[str]
    model_configs: Dict[str, ModelConfig]

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "description": self.description,
            "mood": self.mood,
            "tempo_bpm": self.tempo_bpm,
            "instruments": self.instruments,
            "model_configs": {k: v.to_dict() for k, v in self.model_configs.items()},
        }


DEFAULT_STYLES: Dict[str, StylePreset] = {
    "lofi": StylePreset(
        name="lofi",
        description="Relaxed, mellow lofi hip-hop with soft drums and warm textures.",
        mood="chill",
        tempo_bpm=80,
        instruments=["soft drums", "electric piano", "vinyl noise"],
        model_configs={
            "stub": ModelConfig(
                model_name="stub",
                prompt="A chill lofi hip-hop track with soft drums and warm textures.",
                max_duration_sec=20,
            )
        },
    ),
    "orchestral": StylePreset(
        name="orchestral",
        description="Cinematic orchestral piece with strings, brass, and percussion.",
        mood="epic",
        tempo_bpm=110,
        instruments=["strings", "brass", "timpani"],
        model_configs={
            "stub": ModelConfig(
                model_name="stub",
                prompt="An epic orchestral score with sweeping strings, brass fanfares, and powerful percussion.",
                max_duration_sec=25,
            )
        },
    ),
    "8bit": StylePreset(
        name="8bit",
        description="Retro chiptune with bright square leads and arpeggios.",
        mood="playful",
        tempo_bpm=130,
        instruments=["square lead", "noise snare", "arp"],
        model_configs={
            "stub": ModelConfig(
                model_name="stub",
                prompt="A playful 8-bit chiptune with bright square leads and retro arpeggios.",
                max_duration_sec=20,
            )
        },
    ),
    "rock": StylePreset(
        name="rock",
        description="Energetic rock track with guitars and punchy drums.",
        mood="energetic",
        tempo_bpm=120,
        instruments=["electric guitar", "bass", "drums"],
        model_configs={
            "stub": ModelConfig(
                model_name="stub",
                prompt="An energetic rock track with crunchy guitars, driving bass, and punchy drums.",
                max_duration_sec=22,
            )
        },
    ),
    "ambient": StylePreset(
        name="ambient",
        description="Ethereal ambient soundscape with pads and evolving textures.",
        mood="calm",
        tempo_bpm=70,
        instruments=["pads", "drones", "textures"],
        model_configs={
            "stub": ModelConfig(
                model_name="stub",
                prompt="A calm ambient soundscape with airy pads and evolving textures.",
                max_duration_sec=30,
            )
        },
    ),
}


class StyleConfigManager:
    """
    Provides lookup and listing utilities for style presets.
    """

    def __init__(self, styles: Optional[Dict[str, StylePreset]] = None, style_config: StyleConfig = GLOBAL_STYLE_CONFIG):
        self.styles = styles or DEFAULT_STYLES
        self.style_config = style_config

    def list_styles(self) -> List[str]:
        return sorted(self.styles.keys())

    def get_style(self, name: Optional[str] = None) -> StylePreset:
        style_name = name or self.style_config.default_style
        if style_name not in self.styles:
            raise KeyError(f"Unknown style '{style_name}'. Available: {', '.join(self.list_styles())}")
        return self.styles[style_name]

    def get_model_config(self, style_name: str, model_name: Optional[str] = None) -> ModelConfig:
        style = self.get_style(style_name)
        model_key = model_name or self.style_config.default_model_name
        if model_key not in style.model_configs:
            raise KeyError(
                f"Style '{style_name}' does not define model '{model_key}'. "
                f"Available: {', '.join(style.model_configs.keys())}"
            )
        return style.model_configs[model_key]


def list_available_styles(manager: Optional[StyleConfigManager] = None) -> List[str]:
    manager = manager or StyleConfigManager()
    return manager.list_styles()


def get_style_config(style_name: Optional[str] = None, manager: Optional[StyleConfigManager] = None) -> Dict:
    manager = manager or StyleConfigManager()
    return manager.get_style(style_name).to_dict()
