
---

```md
<!-- docs/04_melody_representation.md -->

# 04 – Melody Representation Module

## 1. Purpose

Convert the **continuous pitch contour** into a **structured representation** that is easier for:

- Music generation models (as a conditioning signal).
- Similarity metrics and analysis.

This includes:

- Note segmentation.
- Rhythmic profiling.
- Symbolic representation (e.g., pseudo-MIDI).
- Optional vector embeddings.

---

## 2. Responsibilities

1. **Note Segmentation**
   - Use F0 and voiced mask to segment the melody into discrete notes:
     - Each note has:
       - Start time.
       - End time (or duration).
       - Pitch (e.g., MIDI value or rounded semitone).
   - Merge micro-variations:
     - Slight pitch fluctuations within a small range are treated as the same note.
   - Remove segments shorter than a minimum duration threshold.

2. **Rhythm Profile Extraction**
   - Derive basic rhythm characteristics:
     - Approximate tempo (if feasible).
     - Distribution of note durations.
     - Simplified rhythm patterns (e.g., relative durations in a quantized grid).

3. **Symbolic Representation**
   - Produce a structured sequence such as:

     ```text
     [
       { "start": t0, "duration": d0, "pitch": p0 },
       { "start": t1, "duration": d1, "pitch": p1 },
       ...
     ]
     ```

   - This symbolic representation can serve as:
     - Condition input for symbolic or hybrid generation models.
     - Basis for embedding or similarity computations.

4. **Embedding / Fingerprint (Optional)**
   - Convert the note sequence into a fixed-length vector:
     - By statistical summarization or a learned embedding.
   - Used for:
     - Quick melodic similarity comparisons.
     - Indexing multiple examples.

5. **Diagnostic Output**
   - Visualize:
     - Piano-roll plots (pitch vs time as blocks).
   - Print/readable summary:
     - Number of notes, pitch range, average note duration.

---

## 3. Inputs and Outputs

### Inputs

- `pitch_contour` and `voiced_mask` (from `03_melody_extraction`).
- Time axis for frames.

### Outputs

- `note_sequence`
  - List of note dictionaries with start, duration, pitch.
- `rhythm_profile`
  - Structure capturing rhythm characteristics.
- Optional `melody_embedding`.

Example:

```text
{
  "note_sequence": [
    {"start": 0.10, "duration": 0.35, "pitch_midi": 64},
    {"start": 0.50, "duration": 0.20, "pitch_midi": 67},
    ...
  ],
  "rhythm_profile": {
    "estimated_tempo_bpm": 90,
    "duration_histogram": [...],
    "quantized_pattern": [...]
  },
  "embedding": [e0, e1, ..., eK]  # optional
}
