"""
Preprocessing module: trim silence, simple filtering, loudness normalization, and length enforcement.

Implements the behaviors described in `docs/02_preprocessing.md`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

from .config import (
    DEFAULT_PREPROCESSING_CONFIG,
    GLOBAL_AUDIO_CONFIG,
    PROCESSED_AUDIO_DIR,
    GlobalAudioConfig,
    PreprocessingConfig,
)

try:
    from pydub import AudioSegment, effects
    from pydub.silence import detect_nonsilent
except ImportError:  # pragma: no cover - dependency guard
    AudioSegment = None  # type: ignore
    effects = None  # type: ignore
    detect_nonsilent = None  # type: ignore

log = logging.getLogger(__name__)


class PreprocessingError(Exception):
    """Raised when preprocessing cannot be completed."""


@dataclass
class PreprocessingMetadata:
    path: Path
    original_duration_sec: float
    processed_duration_sec: float
    sample_rate: int
    applied_steps: List[str]
    notes: str = ""

    def to_dict(self) -> Dict[str, str | float | int | List[str]]:
        data = asdict(self)
        data["path"] = str(self.path)
        return data


class Preprocessor:
    """
    Preprocess normalized audio for downstream melody extraction.

    Typical use:
        pre = Preprocessor()
        meta = pre.preprocess("data/raw/raw_input_...wav")
    """

    def __init__(
        self,
        global_config: GlobalAudioConfig = GLOBAL_AUDIO_CONFIG,
        preprocess_config: PreprocessingConfig = DEFAULT_PREPROCESSING_CONFIG,
        processed_dir: Path | str = PROCESSED_AUDIO_DIR,
    ) -> None:
        self.global_config = global_config
        self.config = preprocess_config
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        if AudioSegment is None:
            raise PreprocessingError("pydub is required for preprocessing. Install via `pip install pydub`.")

    def preprocess(self, raw_audio_path: Path | str) -> PreprocessingMetadata:
        raw_audio_path = Path(raw_audio_path)
        if not raw_audio_path.exists():
            raise PreprocessingError(f"Raw audio not found: {raw_audio_path}")

        segment = AudioSegment.from_file(raw_audio_path)
        segment = self._ensure_global_format(segment)
        original_duration = len(segment) / 1000.0
        applied_steps: List[str] = []
        notes: List[str] = []

        if self.config.trim_silence:
            trimmed, trimmed_flag, trim_note = self._trim_silence(segment)
            segment = trimmed
            if trimmed_flag:
                applied_steps.append("trim_silence")
            if trim_note:
                notes.append(trim_note)

        if self.config.apply_highpass:
            segment = segment.high_pass_filter(self.config.highpass_cutoff_hz)
            applied_steps.append("highpass")

        if self.config.apply_lowpass:
            segment = segment.low_pass_filter(self.config.lowpass_cutoff_hz)
            applied_steps.append("lowpass")

        segment = self._normalize_loudness(segment)
        applied_steps.append("normalize")

        segment, truncated_note = self._enforce_length(segment)
        if truncated_note:
            notes.append(truncated_note)
            applied_steps.append("truncate")

        dest_path = self._output_path(raw_audio_path)
        segment.export(dest_path, format="wav")

        return PreprocessingMetadata(
            path=dest_path,
            original_duration_sec=original_duration,
            processed_duration_sec=len(segment) / 1000.0,
            sample_rate=segment.frame_rate,
            applied_steps=applied_steps,
            notes="; ".join(notes),
        )

    # Internal helpers ------------------------------------------------
    def _ensure_global_format(self, segment: AudioSegment) -> AudioSegment:
        segment = segment.set_frame_rate(self.global_config.sample_rate)
        if self.global_config.mono:
            segment = segment.set_channels(1)
        return segment

    def _trim_silence(self, segment: AudioSegment) -> tuple[AudioSegment, bool, str]:
        if detect_nonsilent is None:
            return segment, False, "Silence detection unavailable; pydub.silence not imported."

        non_silent = detect_nonsilent(
            segment,
            min_silence_len=self.config.min_silence_len_ms,
            silence_thresh=self.config.silence_thresh_db,
        )
        if not non_silent:
            return segment, False, "Silence trim skipped (could not detect non-silent region)."

        start = max(0, non_silent[0][0] - self.config.keep_silence_ms)
        end = min(len(segment), non_silent[-1][1] + self.config.keep_silence_ms)
        trimmed = segment[start:end]
        return trimmed, True, ""

    def _normalize_loudness(self, segment: AudioSegment) -> AudioSegment:
        if segment.rms == 0:
            raise PreprocessingError("Audio appears silent after trimming; cannot normalize.")

        current_db = segment.dBFS
        gain_db = self.config.target_rms_db - current_db
        normalized = segment.apply_gain(gain_db)

        # Secondary normalization to avoid clipping if the gain pushed peaks too high
        normalized = effects.normalize(normalized, headroom=0.1)
        return normalized

    def _enforce_length(self, segment: AudioSegment) -> tuple[AudioSegment, str]:
        if not self.config.enforce_max_duration:
            return segment, ""

        max_ms = int(self.global_config.max_input_duration * 1000)
        if len(segment) <= max_ms:
            return segment, ""

        truncated = segment[:max_ms]
        note = (
            f"Truncated to {self.global_config.max_input_duration}s "
            f"(original {len(segment) / 1000.0:.2f}s after trim)."
        )
        return truncated, note

    def _output_path(self, raw_audio_path: Path) -> Path:
        # Preserve the timestamp part of the raw filename when possible
        stem = raw_audio_path.stem
        if stem.startswith("raw_input_"):
            stem = stem.replace("raw_input_", "processed_input_", 1)
        else:
            stem = f"processed_{stem}"
        return self.processed_dir / f"{stem}.wav"


def preprocess_file(
    raw_audio_path: Path | str,
    preprocessor: Optional[Preprocessor] = None,
) -> Dict:
    """
    Convenience wrapper that returns metadata as a dict.
    """
    preprocessor = preprocessor or Preprocessor()
    meta = preprocessor.preprocess(raw_audio_path)
    return meta.to_dict()
