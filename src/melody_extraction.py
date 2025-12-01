"""
Melody extraction: Convert audio to MIDI using Spotify's basic-pitch.

Implements responsibilities from `docs/03_melody_extraction.md`.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from .config import (
    DEFAULT_MELODY_EXTRACTION_CONFIG,
    GLOBAL_AUDIO_CONFIG,
    GlobalAudioConfig,
    MelodyExtractionConfig,
)

try:
    from basic_pitch.inference import predict_and_save, Model
    from basic_pitch import ICASSP_2022_MODEL_PATH
except ImportError:
    predict_and_save = None
    Model = None
    ICASSP_2022_MODEL_PATH = None

log = logging.getLogger(__name__)


class MelodyExtractionError(Exception):
    """Raised when pitch extraction fails."""


@dataclass
class MelodyContour:
    midi_path: str
    metadata: Dict[str, float | int | str]

    def to_dict(self) -> Dict:
        return {
            "midi_path": self.midi_path,
            "metadata": self.metadata,
        }


class MelodyExtractor:
  
    def __init__(
        self,
        global_config: GlobalAudioConfig = GLOBAL_AUDIO_CONFIG,
        config: MelodyExtractionConfig = DEFAULT_MELODY_EXTRACTION_CONFIG,
    ) -> None:
        if predict_and_save is None:
            raise MelodyExtractionError(
                "basic_pitch is required. Install via `pip install basic-pitch onnxruntime`."
            )
        self.global_config = global_config
        self.config = config
        
        try:
            self.basic_pitch_model = Model(ICASSP_2022_MODEL_PATH)
        except Exception as e:
            raise MelodyExtractionError(f"Failed to load basic-pitch model: {e}")


    def extract(self, preprocessed_audio_path: Path | str, output_dir: Optional[Path | str] = None) -> MelodyContour:
        audio_path = Path(preprocessed_audio_path)
        if not audio_path.exists():
            raise MelodyExtractionError(f"Audio file not found: {audio_path}")

        if output_dir is None:
            output_dir = audio_path.parent
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        expected_midi_filename = audio_path.stem + "_basic_pitch.mid"
        midi_path = output_dir / expected_midi_filename

        if midi_path.exists():
            try:
                os.remove(midi_path)
                log.info(f"Removed existing MIDI file to allow overwrite: {midi_path}")
            except OSError as e:
                log.warning(f"Could not remove existing MIDI file: {e}")

        try:
            predict_and_save(
                audio_path_list=[str(audio_path)],
                output_directory=str(output_dir),
                save_midi=True,
                sonify_midi=False,
                save_model_outputs=False,
                save_notes=False,
                model_or_model_path=self.basic_pitch_model
            )
        except Exception as e:
            raise MelodyExtractionError(f"basic_pitch failed: {e}")

        if not midi_path.exists():
             raise MelodyExtractionError(f"Expected MIDI file was not created: {midi_path}")

        metadata = {
            "extractor": "basic_pitch (onnx)",
            "source_audio": str(audio_path),
        }

        return MelodyContour(
            midi_path=str(midi_path),
            metadata=metadata,
        )


def extract_melody(
    preprocessed_audio_path: Path | str,
    extractor: Optional[MelodyExtractor] = None,
    output_dir: Optional[Path | str] = None
) -> Dict:

    extractor = extractor or MelodyExtractor()
    contour = extractor.extract(preprocessed_audio_path, output_dir=output_dir)
    return contour.to_dict()