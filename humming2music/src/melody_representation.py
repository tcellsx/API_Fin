"""
Melody representation: convert a pitch contour into note sequences, rhythm profile, and optional embedding.

Implements responsibilities from `docs/04_melody_representation.md`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

from .config import (
    DEFAULT_MELODY_REPRESENTATION_CONFIG,
    MelodyRepresentationConfig,
)

try:
    import numpy as np
except ImportError:  # pragma: no cover - dependency guard
    np = None  # type: ignore

log = logging.getLogger(__name__)


class MelodyRepresentationError(Exception):
    """Raised when melody representation fails."""


@dataclass
class NoteEvent:
    start: float
    duration: float
    pitch_midi: int

    def to_dict(self) -> Dict[str, float | int]:
        return {"start": self.start, "duration": self.duration, "pitch_midi": self.pitch_midi}


@dataclass
class MelodyRepresentation:
    note_sequence: List[NoteEvent]
    rhythm_profile: Dict
    embedding: Optional[List[float]] = None

    def to_dict(self) -> Dict:
        return {
            "note_sequence": [n.to_dict() for n in self.note_sequence],
            "rhythm_profile": self.rhythm_profile,
            "embedding": self.embedding,
        }


class MelodyRepresenter:
    """
    Convert frame-level contour into discrete notes and rhythm descriptors.
    """

    def __init__(self, config: MelodyRepresentationConfig = DEFAULT_MELODY_REPRESENTATION_CONFIG) -> None:
        if np is None:
            raise MelodyRepresentationError("numpy is required. Install via `pip install numpy`.")
        self.config = config

    def represent(
        self,
        time: List[float],
        f0_midi: List[float],
        voiced: List[int],
    ) -> MelodyRepresentation:
        if not (len(time) == len(f0_midi) == len(voiced)):
            raise MelodyRepresentationError("time, f0_midi, and voiced lengths must match.")

        hop_sec = self._infer_hop(time)
        note_events = self._segment_notes(time, np.array(f0_midi), np.array(voiced, dtype=bool), hop_sec)
        rhythm_profile = self._derive_rhythm_profile(note_events)
        embedding = self._embed_melody(note_events) if self.config.enable_embedding else None

        return MelodyRepresentation(
            note_sequence=note_events,
            rhythm_profile=rhythm_profile,
            embedding=embedding,
        )

    # Internal helpers ------------------------------------------------
    def _infer_hop(self, time: List[float]) -> float:
        if len(time) < 2:
            return 0.0
        diffs = [t2 - t1 for t1, t2 in zip(time[:-1], time[1:])]
        # Use median to resist occasional irregularities
        return float(np.median(diffs))

    def _segment_notes(
        self,
        time: List[float],
        f0_midi: "np.ndarray",
        voiced: "np.ndarray",
        hop_sec: float,
    ) -> List[NoteEvent]:
        notes: List[NoteEvent] = []
        start_idx: Optional[int] = None

        for idx, is_voiced in enumerate(voiced):
            if is_voiced and start_idx is None:
                start_idx = idx
            if (not is_voiced or idx == len(voiced) - 1) and start_idx is not None:
                end_idx = idx if not is_voiced else idx + 1
                duration = hop_sec * (end_idx - start_idx)
                if duration >= self.config.min_note_duration_sec:
                    pitch_val = self._stable_pitch(f0_midi[start_idx:end_idx])
                    note = NoteEvent(start=time[start_idx], duration=duration, pitch_midi=int(round(pitch_val)))
                    self._append_or_merge(notes, note)
                start_idx = None
        return notes

    def _stable_pitch(self, segment_pitches: "np.ndarray") -> float:
        # Use median to smooth micro-variations
        return float(np.median(segment_pitches[segment_pitches > 0]))

    def _append_or_merge(self, notes: List[NoteEvent], note: NoteEvent) -> None:
        if not notes:
            notes.append(note)
            return
        prev = notes[-1]
        same_pitch = abs(note.pitch_midi - prev.pitch_midi) <= self.config.pitch_merge_tolerance_semitones
        gap = note.start - (prev.start + prev.duration)
        if same_pitch and gap <= self.config.quantization_step_sec:
            prev.duration += gap + note.duration
        else:
            notes.append(note)

    def _derive_rhythm_profile(self, notes: List[NoteEvent]) -> Dict:
        if not notes:
            return {"estimated_tempo_bpm": 0, "duration_histogram": [], "quantized_pattern": []}

        durations = np.array([n.duration for n in notes])
        starts = np.array([n.start for n in notes])

        # Estimate tempo using median inter-onset interval (simple heuristic)
        iois = np.diff(starts)
        if len(iois) == 0 or np.median(iois) == 0:
            tempo_bpm = 0
        else:
            tempo_bpm = float(60.0 / np.median(iois))

        hist, bin_edges = np.histogram(durations, bins=5, range=(0, durations.max()))
        # duration_hist = [{"bin_start": float(bin_edges[i]), "bin_end": float(bin_edges[i + 1]), "count": int(c)} for i, c in enumerate(hist)]
        duration_hist = [
            {
                "bin_start": float(bin_edges[i]),
                "bin_end": float(bin_edges[i + 1]),
                "count": int(c),
            }
            for i, c in enumerate(hist)
        ]

        quantized = self._quantize_pattern(notes)

        return {
            "estimated_tempo_bpm": tempo_bpm,
            "duration_histogram": duration_hist,
            "quantized_pattern": quantized,
        }

    def _quantize_pattern(self, notes: List[NoteEvent]) -> List[Dict[str, float | int]]:
        if not notes or self.config.quantization_step_sec <= 0:
            return []

        step = self.config.quantization_step_sec
        pattern: List[Dict[str, float | int]] = []
        for note in notes:
            start_step = int(round(note.start / step))
            dur_steps = max(1, int(round(note.duration / step)))
            pattern.append(
                {
                    "start_step": start_step,
                    "dur_steps": dur_steps,
                    "pitch_midi": note.pitch_midi,
                }
            )
        return pattern

    def _embed_melody(self, notes: List[NoteEvent]) -> List[float]:
        if not notes:
            return []

        pitches = np.array([n.pitch_midi for n in notes])
        durations = np.array([n.duration for n in notes])
        return [
            float(np.mean(pitches)),
            float(np.std(pitches)),
            float(np.min(pitches)),
            float(np.max(pitches)),
            float(np.mean(durations)),
            float(np.std(durations)),
            len(notes),
        ]


def represent_melody(
    time: List[float],
    f0_midi: List[float],
    voiced: List[int],
    representer: Optional[MelodyRepresenter] = None,
) -> Dict:
    """
    Convenience wrapper returning the representation as a serializable dict.
    """
    representer = representer or MelodyRepresenter()
    rep = representer.represent(time=time, f0_midi=f0_midi, voiced=voiced)
    return rep.to_dict()
