"""
Music generation module: combine melody representation with style configs and call model adapters.

Implements the responsibilities from `docs/06_music_generation.md`.

This implementation provides a stub generator that synthesizes a simple sine wave
as a placeholder for a real model, while keeping the interface ready for swap-in
of actual adapters (e.g., MusicGen, Riffusion).
"""

from __future__ import annotations

import logging
import math
import wave
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .config import (
    GENERATED_AUDIO_DIR,
    GLOBAL_AUDIO_CONFIG,
    GLOBAL_STYLE_CONFIG,
    GlobalAudioConfig,
    StyleConfig,
)
from .style_and_model_config import ModelConfig, StyleConfigManager, StylePreset

try:
    import numpy as np
except ImportError:  # pragma: no cover - dependency guard
    np = None  # type: ignore

log = logging.getLogger(__name__)


class MusicGenerationError(Exception):
    """Raised when generation cannot proceed."""


@dataclass
class GenerationResult:
    audio_path: Path
    model_name: str
    style_name: str
    duration_sec: float
    sample_rate: int
    generation_metadata: Dict

    def to_dict(self) -> Dict:
        data = asdict(self)
        data["audio_path"] = str(self.audio_path)
        return data


class BaseGeneratorAdapter:
    model_name: str = "base"

    def generate(
        self,
        melody_representation: Dict,
        style: StylePreset,
        model_cfg: ModelConfig,
        global_audio: GlobalAudioConfig,
        output_dir: Path,
    ) -> GenerationResult:
        raise NotImplementedError


class StubSineGenerator(BaseGeneratorAdapter):
    """
    Minimal adapter that writes a sine wave WAV file as a stand-in for real models.
    """

    model_name = "stub"

    def generate(
        self,
        melody_representation: Dict,
        style: StylePreset,
        model_cfg: ModelConfig,
        global_audio: GlobalAudioConfig,
        output_dir: Path,
    ) -> GenerationResult:
        if np is None:
            raise MusicGenerationError("numpy is required for stub generation. Install via `pip install numpy`.")

        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"gen_{style.name}_{self.model_name}_{timestamp}.wav"
        out_path = output_dir / filename

        duration = self._estimate_duration(melody_representation, model_cfg, global_audio)
        freq = self._estimate_pitch(melody_representation)
        sample_rate = global_audio.sample_rate

        t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
        waveform = 0.2 * np.sin(2 * math.pi * freq * t)

        # Write 16-bit PCM WAV
        with wave.open(str(out_path), "w") as wav_file:
            wav_file.setnchannels(1 if global_audio.mono else 2)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            pcm = np.int16(waveform * 32767)
            wav_file.writeframes(pcm.tobytes())

        metadata = {
            "prompt": model_cfg.prompt,
            "seed": None,
            "timestamp": timestamp,
        }

        return GenerationResult(
            audio_path=out_path,
            model_name=self.model_name,
            style_name=style.name,
            duration_sec=duration,
            sample_rate=sample_rate,
            generation_metadata=metadata,
        )

    def _estimate_duration(
        self,
        melody_representation: Dict,
        model_cfg: ModelConfig,
        global_audio: GlobalAudioConfig,
    ) -> float:
        max_allowed = global_audio.max_input_duration
        if model_cfg.max_duration_sec is not None:
            max_allowed = min(max_allowed, model_cfg.max_duration_sec)

        notes = melody_representation.get("note_sequence", [])
        if not notes:
            return min(5.0, max_allowed)
        end_times = [(n["start"] + n["duration"]) for n in notes if "start" in n and "duration" in n]
        est = max(end_times) + 1.0
        return min(est, max_allowed)

    def _estimate_pitch(self, melody_representation: Dict) -> float:
        notes = melody_representation.get("note_sequence", [])
        if not notes:
            return 220.0
        midi_vals = [n.get("pitch_midi", 60) for n in notes]
        midi_mean = sum(midi_vals) / len(midi_vals)
        return 440.0 * (2 ** ((midi_mean - 69) / 12))


class MusicGenerator:
    """
    High-level generation orchestrator that selects an adapter and returns standardized results.
    """

    def __init__(
        self,
        adapters: Optional[Dict[str, BaseGeneratorAdapter]] = None,
        style_manager: Optional[StyleConfigManager] = None,
        style_config: StyleConfig = GLOBAL_STYLE_CONFIG,
        global_audio: GlobalAudioConfig = GLOBAL_AUDIO_CONFIG,
        output_dir: Path = GENERATED_AUDIO_DIR,
    ) -> None:
        self.adapters = adapters or {StubSineGenerator.model_name: StubSineGenerator()}
        self.style_manager = style_manager or StyleConfigManager()
        self.style_config = style_config
        self.global_audio = global_audio
        self.output_dir = output_dir

    def generate(
        self,
        melody_representation: Dict,
        style_name: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> GenerationResult:
        style = self.style_manager.get_style(style_name)
        model_cfg = self.style_manager.get_model_config(style.name, model_name)

        adapter = self._get_adapter(model_cfg.model_name)
        return adapter.generate(
            melody_representation=melody_representation,
            style=style,
            model_cfg=model_cfg,
            global_audio=self.global_audio,
            output_dir=self.output_dir,
        )

    def generate_multi(
        self,
        melody_representation: Dict,
        style_names: List[str],
        model_name: Optional[str] = None,
    ) -> List[GenerationResult]:
        results = []
        for s in style_names:
            try:
                results.append(self.generate(melody_representation, style_name=s, model_name=model_name))
            except Exception as exc:
                log.error("Generation failed for style %s: %s", s, exc)
        return results

    def _get_adapter(self, model_name: str) -> BaseGeneratorAdapter:
        if model_name not in self.adapters:
            raise MusicGenerationError(
                f"No adapter registered for model '{model_name}'. "
                f"Available: {', '.join(self.adapters.keys())}"
            )
        return self.adapters[model_name]


def generate_track(
    melody_representation: Dict,
    style_name: Optional[str] = None,
    model_name: Optional[str] = None,
    generator: Optional[MusicGenerator] = None,
) -> Dict:
    generator = generator or MusicGenerator()
    result = generator.generate(
        melody_representation=melody_representation,
        style_name=style_name,
        model_name=model_name,
    )
    return result.to_dict()


def generate_multi_style(
    melody_representation: Dict,
    style_names: List[str],
    model_name: Optional[str] = None,
    generator: Optional[MusicGenerator] = None,
) -> List[Dict]:
    generator = generator or MusicGenerator()
    results = generator.generate_multi(
        melody_representation=melody_representation,
        style_names=style_names,
        model_name=model_name,
    )
    return [r.to_dict() for r in results]
