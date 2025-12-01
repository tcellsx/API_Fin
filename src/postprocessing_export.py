"""
Post-processing and export module.

Applies loudness normalization, fades, and exports generated audio as per
`docs/07_postprocessing_export.md`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

from .config import (
    DEFAULT_POSTPROCESSING_CONFIG,
    GLOBAL_AUDIO_CONFIG,
    POSTPROCESSED_AUDIO_DIR,
    GlobalAudioConfig,
    PostprocessingConfig,
)

try:
    from pydub import AudioSegment, effects
except ImportError:  # pragma: no cover - dependency guard
    AudioSegment = None  # type: ignore
    effects = None  # type: ignore

log = logging.getLogger(__name__)


class PostprocessingError(Exception):
    """Raised when post-processing fails."""


@dataclass
class PostprocessingResult:
    final_audio_path: Path
    final_audio_path_mp3: Optional[Path]
    duration_sec: float
    sample_rate: int
    postprocessing_applied: List[str]
    style_name: str
    model_name: str

    def to_dict(self) -> Dict:
        data = asdict(self)
        data["final_audio_path"] = str(self.final_audio_path)
        if self.final_audio_path_mp3:
            data["final_audio_path_mp3"] = str(self.final_audio_path_mp3)
        return data


class Postprocessor:
    """
    Polishes generated audio and exports final files.
    """

    def __init__(
        self,
        config: PostprocessingConfig = DEFAULT_POSTPROCESSING_CONFIG,
        global_audio: GlobalAudioConfig = GLOBAL_AUDIO_CONFIG,
        output_dir: Path = POSTPROCESSED_AUDIO_DIR,
    ) -> None:
        if AudioSegment is None:
            raise PostprocessingError("pydub is required for postprocessing. Install via `pip install pydub`.")
        self.config = config
        self.global_audio = global_audio
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def process(
        self,
        generated_audio_path: Path | str,
        style_name: str,
        model_name: str,
    ) -> PostprocessingResult:
        in_path = Path(generated_audio_path)
        if not in_path.exists():
            raise PostprocessingError(f"Generated audio not found: {in_path}")

        segment = AudioSegment.from_file(in_path)
        applied: List[str] = []

        segment = self._ensure_sr_channels(segment)

        segment = self._normalize(segment)
        applied.append("normalize")

        if self.config.fade_in_ms > 0:
            segment = segment.fade_in(self.config.fade_in_ms)
            applied.append("fade_in")
        if self.config.fade_out_ms > 0:
            segment = segment.fade_out(self.config.fade_out_ms)
            applied.append("fade_out")

        base_name = in_path.stem
        wav_out = self.output_dir / f"{base_name}_final.wav"
        segment.export(wav_out, format="wav")

        mp3_out: Optional[Path] = None
        if self.config.export_mp3:
            mp3_out = self.output_dir / f"{base_name}_final.mp3"
            segment.export(mp3_out, format="mp3")
            applied.append("export_mp3")

        return PostprocessingResult(
            final_audio_path=wav_out,
            final_audio_path_mp3=mp3_out,
            duration_sec=len(segment) / 1000.0,
            sample_rate=segment.frame_rate,
            postprocessing_applied=applied,
            style_name=style_name,
            model_name=model_name,
        )

    # Internal helpers ------------------------------------------------
    def _ensure_sr_channels(self, segment: AudioSegment) -> AudioSegment:
        segment = segment.set_frame_rate(self.global_audio.sample_rate)
        if self.global_audio.mono:
            segment = segment.set_channels(1)
        return segment

    def _normalize(self, segment: AudioSegment) -> AudioSegment:
        if segment.rms == 0:
            raise PostprocessingError("Audio is silent; cannot normalize.")

        gain_db = self.config.target_rms_db - segment.dBFS
        segment = segment.apply_gain(gain_db)
        segment = effects.normalize(segment, headroom=0.1)
        return segment


def postprocess_audio(
    generated_audio_path: Path | str,
    style_name: str,
    model_name: str,
    postprocessor: Optional[Postprocessor] = None,
) -> Dict:
    postprocessor = postprocessor or Postprocessor()
    result = postprocessor.process(generated_audio_path, style_name=style_name, model_name=model_name)
    return result.to_dict()
