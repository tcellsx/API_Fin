"""
Similarity and evaluation module.

Compares generated tracks to the original melody using pitch and rhythm metrics,
as outlined in `docs/08_similarity_evaluation.md`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

from .config import (
    DEFAULT_SIMILARITY_CONFIG,
    GLOBAL_AUDIO_CONFIG,
    EVAL_OUTPUT_DIR,
    GlobalAudioConfig,
    SimilarityConfig,
)
from .melody_extraction import MelodyExtractor
from .melody_representation import MelodyRepresenter

try:
    import numpy as np
    import librosa
except ImportError:  # pragma: no cover - dependency guard
    np = None  # type: ignore
    librosa = None  # type: ignore

log = logging.getLogger(__name__)


class SimilarityError(Exception):
    """Raised when similarity evaluation fails."""


@dataclass
class SimilarityReport:
    style_name: str
    model_name: str
    pitch_similarity: float
    rhythm_similarity: float
    overall_similarity: float
    metadata: Dict

    def to_dict(self) -> Dict:
        return asdict(self)


class SimilarityEvaluator:
    """
    Evaluates similarity between the original input and generated outputs.
    """

    def __init__(
        self,
        similarity_config: SimilarityConfig = DEFAULT_SIMILARITY_CONFIG,
        global_audio: GlobalAudioConfig = GLOBAL_AUDIO_CONFIG,
        eval_output_dir: Path = EVAL_OUTPUT_DIR,
    ) -> None:
        if np is None or librosa is None:
            raise SimilarityError("librosa and numpy are required. Install via `pip install librosa numpy`.")
        self.config = similarity_config
        self.global_audio = global_audio
        self.eval_output_dir = Path(eval_output_dir)
        self.eval_output_dir.mkdir(parents=True, exist_ok=True)
        self.extractor = MelodyExtractor(global_config=global_audio)
        self.representer = MelodyRepresenter()

    def evaluate(
        self,
        original_processed_audio: Path | str,
        generated_audio: Path | str,
        style_name: str,
        model_name: str,
    ) -> SimilarityReport:
        # Extract contour & representation for original
        orig_contour = self.extractor.extract(original_processed_audio)
        orig_rep = self.representer.represent(
            time=orig_contour.time,
            f0_midi=orig_contour.f0_midi,
            voiced=orig_contour.voiced,
        )

        # Extract contour & representation for generated
        gen_contour = self.extractor.extract(generated_audio)
        gen_rep = self.representer.represent(
            time=gen_contour.time,
            f0_midi=gen_contour.f0_midi,
            voiced=gen_contour.voiced,
        )

        pitch_sim = self._pitch_similarity(orig_contour.f0_midi, gen_contour.f0_midi)
        rhythm_sim = self._rhythm_similarity(orig_rep, gen_rep)

        overall = (
            self.config.pitch_weight * pitch_sim + self.config.rhythm_weight * rhythm_sim
        )

        metadata = {
            "dtw_cost": self.config.dtw_cost,
            "pitch_weight": self.config.pitch_weight,
            "rhythm_weight": self.config.rhythm_weight,
        }

        return SimilarityReport(
            style_name=style_name,
            model_name=model_name,
            pitch_similarity=pitch_sim,
            rhythm_similarity=rhythm_sim,
            overall_similarity=overall,
            metadata=metadata,
        )

    # Internal helpers ------------------------------------------------
    def _pitch_similarity(self, midi_a: List[float], midi_b: List[float]) -> float:
        arr_a = np.array(midi_a)
        arr_b = np.array(midi_b)
        # Align lengths by trimming/padding with zeros
        if len(arr_a) != len(arr_b):
            min_len = min(len(arr_a), len(arr_b))
            arr_a = arr_a[:min_len]
            arr_b = arr_b[:min_len]

        # DTW distance over pitch values; treat zeros as silence (mask them out)
        mask = (arr_a > 0) & (arr_b > 0)
        if not np.any(mask):
            return 0.0
        seq_a = arr_a[mask]
        seq_b = arr_b[mask]

        if seq_a.size == 0 or seq_b.size == 0:
            return 0.0

        dist, _ = librosa.sequence.dtw(seq_a[:, None], seq_b[:, None], metric=self.config.dtw_cost)
        # Normalize distance by path length, then convert to similarity
        norm_dist = dist[-1, -1] / dist.shape[0]
        sim = 1.0 / (1.0 + norm_dist)
        return float(sim)


    # def _rhythm_similarity(self, rep_a: Dict, rep_b: Dict) -> float:
    #     pattern_a = rep_a.rhythm_profile.get("quantized_pattern", []) if hasattr(rep_a, "rhythm_profile") else rep_a.get("rhythm_profile", {}).get("quantized_pattern", [])
    #     pattern_b = rep_b.rhythm_profile.get("quantized_pattern", []) if hasattr(rep_b, "rhythm_profile") else rep_b.get("rhythm_profile", {}).get("quantized_pattern", [])
    #     if not pattern_a or not pattern_b:
    #         return 0.0

    #     # Build simple edit distance over start_step/pitch pairs
    #     seq_a = [(p["start_step"], p["pitch_midi"]) for p in pattern_a]
    #     seq_b = [(p["start_step"], p["pitch_midi"]) for p in pattern_b]
    #     dist = self._levenshtein(seq_a, seq_b)
    #     max_len = max(len(seq_a), len(seq_b), 1)
    #     sim = 1.0 - (dist / max_len)
    #     return float(sim)
    
    def _rhythm_similarity(self, rep_a, rep_b) -> float:
    
        if hasattr(rep_a, "rhythm_profile"):
            pattern_a = rep_a.rhythm_profile.get("quantized_pattern", [])
        else:
            pattern_a = rep_a.get("rhythm_profile", {}).get("quantized_pattern", [])

        if hasattr(rep_b, "rhythm_profile"):
            pattern_b = rep_b.rhythm_profile.get("quantized_pattern", [])
        else:
            pattern_b = rep_b.get("rhythm_profile", {}).get("quantized_pattern", [])

        if not pattern_a or not pattern_b:
            return 0.0

        seq_a = [(p["start_step"], p["pitch_midi"]) for p in pattern_a]
        seq_b = [(p["start_step"], p["pitch_midi"]) for p in pattern_b]
        dist = self._levenshtein(seq_a, seq_b)
        max_len = max(len(seq_a), len(seq_b), 1)
        sim = 1.0 - (dist / max_len)
        return float(sim)


    def _levenshtein(self, seq1: List, seq2: List) -> int:
        if len(seq1) < len(seq2):
            return self._levenshtein(seq2, seq1)
        # len(seq1) >= len(seq2)
        previous_row = list(range(len(seq2) + 1))
        for i, c1 in enumerate(seq1):
            current_row = [i + 1]
            for j, c2 in enumerate(seq2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1]


def evaluate_similarity(
    original_processed_audio: Path | str,
    generated_audio: Path | str,
    style_name: str,
    model_name: str,
    evaluator: Optional[SimilarityEvaluator] = None,
) -> Dict:
    evaluator = evaluator or SimilarityEvaluator()
    report = evaluator.evaluate(
        original_processed_audio=original_processed_audio,
        generated_audio=generated_audio,
        style_name=style_name,
        model_name=model_name,
    )
    return report.to_dict()
