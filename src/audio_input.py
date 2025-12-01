"""
Audio Input module: recording, uploading, normalization, validation, and metadata extraction.

Implements the responsibilities described in `docs/01_audio_input.md`:
- record or ingest uploaded audio
- normalize to a consistent WAV/mono/sample-rate format
- enforce duration constraints
- extract metadata for downstream modules
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from .config import GLOBAL_AUDIO_CONFIG, RAW_AUDIO_DIR, GlobalAudioConfig

try:
    from pydub import AudioSegment
except ImportError:  # pragma: no cover - dependency guard
    AudioSegment = None  # type: ignore

try:
    import numpy as np
    import sounddevice as sd
except ImportError:  # pragma: no cover - dependency guard
    np = None  # type: ignore
    sd = None  # type: ignore


log = logging.getLogger(__name__)


class AudioInputError(Exception):
    """Raised when audio ingestion or validation fails."""


@dataclass
class AudioMetadata:
    path: Path
    duration_sec: float
    sample_rate: int
    channels: int
    format: str
    source_type: str  # "recorded" | "uploaded"

    def to_dict(self) -> Dict[str, str | float | int]:
        data = asdict(self)
        data["path"] = str(self.path)
        return data


class AudioInputManager:
    """
    Entry point for ingesting user audio.

    Typical use:
        manager = AudioInputManager()
        metadata = manager.ingest_upload("path/to/file.mp3")
    """

    def __init__(
        self,
        config: GlobalAudioConfig = GLOBAL_AUDIO_CONFIG,
        raw_audio_dir: Path | str = RAW_AUDIO_DIR,
    ) -> None:
        self.config = config
        self.raw_audio_dir = Path(raw_audio_dir)
        self.raw_audio_dir.mkdir(parents=True, exist_ok=True)

    # Public API -----------------------------------------------------
    def ingest_upload(self, upload_path: Path | str, session_id: str = "0") -> AudioMetadata:
        """
        Normalize and store an uploaded audio file.
        """
        if AudioSegment is None:
            raise AudioInputError("pydub is required for ingesting uploads. Install via `pip install pydub`.")

        upload_path = Path(upload_path)
        if not upload_path.exists():
            raise AudioInputError(f"File not found: {upload_path}")

        if upload_path.suffix.replace(".", "").lower() not in [fmt.lower() for fmt in self.config.supported_formats]:
            raise AudioInputError(
                f"Unsupported format for {upload_path}. Supported: {', '.join(self.config.supported_formats)}"
            )

        segment = AudioSegment.from_file(upload_path)
        dest_path = self._normalized_output_path(session_id)
        return self._normalize_and_persist(segment, dest_path, source_type="uploaded")

    def record_from_microphone(self, duration_sec: float, session_id: str = "0") -> AudioMetadata:
        """
        Record audio from the default microphone for `duration_sec` seconds.
        """
        if sd is None or np is None:
            raise AudioInputError("Recording requires `sounddevice` and `numpy`. Install them to enable recording.")
        if duration_sec <= 0:
            raise AudioInputError("Recording duration must be positive.")

        channels = 1 if self.config.mono else 2
        frames = int(duration_sec * self.config.sample_rate)
        log.info("Recording audio: %s seconds @ %s Hz", duration_sec, self.config.sample_rate)

        recording = sd.rec(
            frames,
            samplerate=self.config.sample_rate,
            channels=channels,
            dtype="int16",
        )
        sd.wait()

        segment = AudioSegment(
            recording.tobytes(),
            frame_rate=self.config.sample_rate,
            sample_width=recording.dtype.itemsize,
            channels=channels,
        )

        dest_path = self._normalized_output_path(session_id)
        return self._normalize_and_persist(segment, dest_path, source_type="recorded")

    # Internal helpers ----------------------------------------------
    def _normalized_output_path(self, session_id: str) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"raw_input_{timestamp}_session{session_id}.wav"
        return self.raw_audio_dir / filename

    def _normalize_and_persist(self, segment: AudioSegment, dest_path: Path, source_type: str) -> AudioMetadata:
        normalized = self._normalize(segment)
        self._validate_duration(normalized)
        normalized.export(dest_path, format="wav")
        return self._extract_metadata(dest_path, normalized, source_type)

    def _normalize(self, segment: AudioSegment) -> AudioSegment:
        segment = segment.set_frame_rate(self.config.sample_rate)
        if self.config.mono:
            segment = segment.set_channels(1)
        return segment

    def _validate_duration(self, segment: AudioSegment) -> None:
        duration_sec = len(segment) / 1000.0  # pydub duration in ms
        if duration_sec < self.config.min_input_duration:
            raise AudioInputError(
                f"Audio too short ({duration_sec:.2f}s). Minimum: {self.config.min_input_duration}s"
            )
        if duration_sec > self.config.max_input_duration:
            raise AudioInputError(
                f"Audio too long ({duration_sec:.2f}s). Maximum: {self.config.max_input_duration}s"
            )

    def _extract_metadata(self, path: Path, segment: AudioSegment, source_type: str) -> AudioMetadata:
        duration_sec = len(segment) / 1000.0
        sample_rate = segment.frame_rate
        channels = segment.channels
        return AudioMetadata(
            path=path,
            duration_sec=duration_sec,
            sample_rate=sample_rate,
            channels=channels,
            format="wav",
            source_type=source_type,
        )


def ingest_file(upload_path: Path | str, session_id: str = "0", manager: Optional[AudioInputManager] = None) -> Dict:
    """
    Convenience wrapper to ingest a file and return metadata as a dict.
    """
    manager = manager or AudioInputManager()
    metadata = manager.ingest_upload(upload_path, session_id=session_id)
    return metadata.to_dict()

