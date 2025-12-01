"""
Shared configuration objects for the humming2music prototype.

The values here mirror the expectations described in the documentation
(`docs/00_overview.md` and module-specific specs). They are intended to be
imported by the individual pipeline modules.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class GlobalAudioConfig:
    """Global audio parameters used across the pipeline."""

    sample_rate: int = 16_000
    mono: bool = True
    max_input_duration: float = 30.0  # seconds
    min_input_duration: float = 0.5  # seconds
    supported_formats: List[str] = (
        "wav",
        "mp3",
        "m4a",
        "flac",
        "ogg",
        "aiff",
    )


@dataclass(frozen=True)
class PreprocessingConfig:
    """Configurable parameters for the preprocessing stage."""

    # Silence trimming
    trim_silence: bool = True
    silence_thresh_db: float = -40.0
    min_silence_len_ms: int = 100
    keep_silence_ms: int = 50

    # Filtering
    apply_highpass: bool = True
    highpass_cutoff_hz: int = 80
    apply_lowpass: bool = False
    lowpass_cutoff_hz: int = 8_000

    # Loudness
    target_rms_db: float = -20.0

    # Length enforcement toggles (uses global max/min durations)
    enforce_max_duration: bool = True


@dataclass(frozen=True)
class MelodyExtractionConfig:
    """Parameters for frame-based pitch tracking."""

    frame_length: int = 2048
    hop_length: int = 256
    min_f0_hz: float = 50.0
    max_f0_hz: float = 1000.0
    smoothing_window: int = 5  # frames (odd number recommended)
    min_voiced_duration_sec: float = 0.1  # drop shorter voiced islands


@dataclass(frozen=True)
class MelodyRepresentationConfig:
    """Parameters controlling note segmentation and summarization."""

    min_note_duration_sec: float = 0.12
    pitch_merge_tolerance_semitones: float = 0.35
    quantization_step_sec: float = 0.25
    enable_embedding: bool = True


@dataclass(frozen=True)
class StyleConfig:
    """Global style/model configuration options."""

    default_style: str = "lofi"
    default_model_name: str = "stub"
    max_generation_duration: float = 30.0  # seconds


@dataclass(frozen=True)
class PostprocessingConfig:
    """Parameters for post-generation audio polishing."""

    target_rms_db: float = -14.0
    fade_in_ms: int = 80
    fade_out_ms: int = 120
    export_mp3: bool = False  # requires ffmpeg/lame when True


@dataclass(frozen=True)
class SimilarityConfig:
    """Parameters for similarity evaluation."""

    pitch_weight: float = 0.7
    rhythm_weight: float = 0.3
    dtw_cost: str = "euclidean"


# Project-level directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_AUDIO_DIR = DATA_DIR / "raw"
PROCESSED_AUDIO_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
GENERATED_AUDIO_DIR = OUTPUTS_DIR / "generated"
POSTPROCESSED_AUDIO_DIR = OUTPUTS_DIR / "final"
EVAL_OUTPUT_DIR = OUTPUTS_DIR / "evaluation"

# Default global config used by modules
GLOBAL_AUDIO_CONFIG = GlobalAudioConfig()
DEFAULT_PREPROCESSING_CONFIG = PreprocessingConfig()
DEFAULT_MELODY_EXTRACTION_CONFIG = MelodyExtractionConfig()
DEFAULT_MELODY_REPRESENTATION_CONFIG = MelodyRepresentationConfig()
GLOBAL_STYLE_CONFIG = StyleConfig()
DEFAULT_POSTPROCESSING_CONFIG = PostprocessingConfig()
DEFAULT_SIMILARITY_CONFIG = SimilarityConfig()
