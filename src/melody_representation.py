"""
Melody representation: convert a MIDI file into note sequences, rhythm profile, and optional embedding.

Implements responsibilities from `docs/04_melody_representation.md`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

from .config import (
    DEFAULT_MELODY_REPRESENTATION_CONFIG,
    MelodyRepresentationConfig,
)

try:
    import numpy as np
    import pretty_midi
except ImportError:  # pragma: no cover - dependency guard
    np = None  # type: ignore
    pretty_midi = None  # type: ignore

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
    midi_path: Optional[str] = None
    embedding: Optional[List[float]] = None

    def to_dict(self) -> Dict:
        return {
            "note_sequence": [n.to_dict() for n in self.note_sequence],
            "rhythm_profile": self.rhythm_profile,
            "midi_path": self.midi_path,
            "embedding": self.embedding,
        }


class MelodyRepresenter:
    """
    Convert MIDI file into discrete notes and rhythm descriptors.
    """

    def __init__(self, config: MelodyRepresentationConfig = DEFAULT_MELODY_REPRESENTATION_CONFIG) -> None:
        if np is None or pretty_midi is None:
            raise MelodyRepresentationError("numpy and pretty_midi are required. Install via `pip install numpy pretty_midi`.")
        self.config = config

    def represent(
        self,
        midi_path: str,
    ) -> MelodyRepresentation:
        try:
            pm = pretty_midi.PrettyMIDI(midi_path)
        except Exception as e:
            raise MelodyRepresentationError(f"Failed to load MIDI file {midi_path}: {e}")

        note_events = self._extract_notes(pm)
        rhythm_profile = self._derive_rhythm_profile(note_events, pm)
        embedding = self._embed_melody(note_events) if self.config.enable_embedding else None

        return MelodyRepresentation(
            note_sequence=note_events,
            rhythm_profile=rhythm_profile,
            midi_path=midi_path,
            embedding=embedding,
        )

    # Internal helpers ------------------------------------------------
    def _extract_notes(self, pm: "pretty_midi.PrettyMIDI") -> List[NoteEvent]:
        notes: List[NoteEvent] = []
        # basic_pitch usually outputs to a single instrument, but we iterate all just in case
        for instrument in pm.instruments:
            if instrument.is_drum:
                continue
            for note in instrument.notes:
                # Filter short notes if needed, though basic_pitch is usually good
                if note.end - note.start >= self.config.min_note_duration_sec:
                    notes.append(NoteEvent(
                        start=note.start,
                        duration=note.end - note.start,
                        pitch_midi=note.pitch
                    ))
        
        # Sort by start time
        notes.sort(key=lambda n: n.start)
        return notes

    def _derive_rhythm_profile(self, notes: List[NoteEvent], pm: "pretty_midi.PrettyMIDI") -> Dict:
        if not notes:
            return {"estimated_tempo_bpm": 0, "duration_histogram": [], "quantized_pattern": []}

        durations = np.array([n.duration for n in notes])
        
        # Use pretty_midi's tempo estimation or fallback
        tempo_bpm = pm.estimate_tempo()
        if tempo_bpm is None or tempo_bpm <= 0:
             # Fallback to simple heuristic if pm fails
             starts = np.array([n.start for n in notes])
             iois = np.diff(starts)
             if len(iois) > 0 and np.median(iois) > 0:
                 tempo_bpm = float(60.0 / np.median(iois))
             else:
                 tempo_bpm = 120.0 # Default

        hist, bin_edges = np.histogram(durations, bins=5, range=(0, durations.max() if len(durations) > 0 else 1))
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
    midi_path: str,
    representer: Optional[MelodyRepresenter] = None,
) -> Dict:
    """
    Convenience wrapper returning the representation as a serializable dict.
    """
    representer = representer or MelodyRepresenter()
    rep = representer.represent(midi_path=midi_path)
    return rep.to_dict()
