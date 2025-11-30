"""
Melody extraction: frame-level F0 estimation, voiced/unvoiced detection, and smoothing.

Implements responsibilities from `docs/03_melody_extraction.md`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

from .config import (
    DEFAULT_MELODY_EXTRACTION_CONFIG,
    GLOBAL_AUDIO_CONFIG,
    GlobalAudioConfig,
    MelodyExtractionConfig,
)

try:
    import numpy as np
    import librosa
except ImportError:  # pragma: no cover - dependency guard
    np = None  # type: ignore
    librosa = None  # type: ignore

log = logging.getLogger(__name__)


class MelodyExtractionError(Exception):
    """Raised when pitch extraction fails."""


@dataclass
class MelodyContour:
    time: List[float]
    f0_hz: List[float]
    f0_midi: List[float]
    voiced: List[int]
    metadata: Dict[str, float | int]

    def to_dict(self) -> Dict:
        return {
            "time": self.time,
            "f0_hz": self.f0_hz,
            "f0_midi": self.f0_midi,
            "voiced": self.voiced,
            "metadata": self.metadata,
        }


class MelodyExtractor:
    """
    Estimate pitch contour (Hz and MIDI) plus voiced mask.

    Uses librosa.pyin for F0 estimation with post-processing to smooth
    and clamp values into a valid range.
    """

    def __init__(
        self,
        global_config: GlobalAudioConfig = GLOBAL_AUDIO_CONFIG,
        config: MelodyExtractionConfig = DEFAULT_MELODY_EXTRACTION_CONFIG,
    ) -> None:
        if librosa is None or np is None:
            raise MelodyExtractionError("librosa and numpy are required. Install via `pip install librosa numpy`.")
        self.global_config = global_config
        self.config = config

    def extract(self, preprocessed_audio_path: Path | str) -> MelodyContour:
        audio_path = Path(preprocessed_audio_path)
        if not audio_path.exists():
            raise MelodyExtractionError(f"Audio file not found: {audio_path}")

        y, sr = librosa.load(audio_path, sr=self.global_config.sample_rate, mono=True)
        if y.size == 0:
            raise MelodyExtractionError("Loaded audio is empty.")

        f0, voiced_flag, voiced_prob = librosa.pyin(
            y,
            fmin=self.config.min_f0_hz,
            fmax=self.config.max_f0_hz,
            sr=sr,
            frame_length=self.config.frame_length,
            hop_length=self.config.hop_length,
        )

        time_axis = librosa.times_like(f0, sr=sr, hop_length=self.config.hop_length).tolist()

        # voiced_flag can be None on some librosa versions; fall back to nan-based mask
        if voiced_flag is None:
            mask = ~np.isnan(f0)
        else:
            mask = voiced_flag.astype(bool)

        f0 = np.nan_to_num(f0, nan=0.0)
        f0 = np.clip(f0, self.config.min_f0_hz, self.config.max_f0_hz)

        f0 = self._smooth_contour(f0, mask)
        mask = self._remove_short_voiced(mask, sr)

        f0_midi = self._hz_to_midi(f0, mask)
        voiced_int = mask.astype(int).tolist()

        metadata = {
            "frame_size": self.config.frame_length,
            "hop_size": self.config.hop_length,
            "min_f0": self.config.min_f0_hz,
            "max_f0": self.config.max_f0_hz,
            "smoothing_window": self.config.smoothing_window,
            "sr": sr,
        }

        return MelodyContour(
            time=time_axis,
            f0_hz=f0.tolist(),
            f0_midi=f0_midi.tolist(),
            voiced=voiced_int,
            metadata=metadata,
        )

    # Internal helpers ------------------------------------------------
    def _smooth_contour(self, f0: "np.ndarray", mask: "np.ndarray") -> "np.ndarray":
        if self.config.smoothing_window <= 1:
            return f0

        k = self.config.smoothing_window
        pad = k // 2
        padded = np.pad(f0, (pad, pad), mode="edge")
        try:
            windows = np.lib.stride_tricks.sliding_window_view(padded, window_shape=k)
            median_vals = np.median(windows, axis=-1)
        except Exception:
            # Fallback to simple loop if sliding_window_view unavailable
            median_vals = []
            for i in range(len(f0)):
                window = padded[i : i + k]
                median_vals.append(float(np.median(window)))
            median_vals = np.array(median_vals)

        smoothed = f0.copy()
        smoothed[mask] = median_vals[mask]
        return smoothed

    def _remove_short_voiced(self, mask: "np.ndarray", sr: int) -> "np.ndarray":
        min_frames = int(self.config.min_voiced_duration_sec * sr / self.config.hop_length)
        if min_frames <= 1:
            return mask

        cleaned = mask.copy()
        start = None
        for idx, val in enumerate(mask):
            if val and start is None:
                start = idx
            elif not val and start is not None:
                length = idx - start
                if length < min_frames:
                    cleaned[start:idx] = False
                start = None
        if start is not None:
            length = len(mask) - start
            if length < min_frames:
                cleaned[start:] = False
        return cleaned

    def _hz_to_midi(self, f0: "np.ndarray", mask: "np.ndarray") -> "np.ndarray":
        midi = np.full_like(f0, fill_value=0.0)
        voiced_vals = f0[mask]
        if voiced_vals.size:
            midi_voiced = librosa.hz_to_midi(voiced_vals)
            midi[mask] = midi_voiced
        return midi


def extract_melody(preprocessed_audio_path: Path | str, extractor: Optional[MelodyExtractor] = None) -> Dict:
    """
    Convenience wrapper returning the contour as a serializable dict.
    """
    extractor = extractor or MelodyExtractor()
    contour = extractor.extract(preprocessed_audio_path)
    return contour.to_dict()
