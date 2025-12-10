
---

```md
<!-- docs/03_melody_extraction.md -->

# 03 – Melody Extraction Module

## 1. Purpose

Extract the **melodic contour** from preprocessed vocal audio:

- Estimate pitch over time (F0).
- Determine voiced vs unvoiced segments.
- Produce a **smoothed pitch contour** that reflects the sung/hummed melody.

This module is central to the whole system.

---

## 2. Responsibilities

1. **Frame-Based Pitch Tracking**
   - Analyze the audio in short frames with a given frame size and hop size.
   - Use a chosen pitch estimator (e.g., a pretrained model like CREPE or another F0 estimator).
   - For each frame, output:
     - Estimated F0 (Hz) or a special value (e.g., 0 or NaN) for unvoiced.

2. **Voiced/Unvoiced Decision**
   - Determine if each frame is voiced (contains pitched sound) or unvoiced.
   - Based on estimator confidence and/or F0 range thresholds.
   - Produce a binary mask: `voiced_mask[t] ∈ {0,1}`.

3. **Post-processing and Smoothing**
   - Apply temporal smoothing:
     - Median/mean filtering over a small window (configurable).
   - Remove extremely short voiced segments (likely noise).
   - Clamp F0 values to valid range:
     - Between `MIN_F0` and `MAX_F0`.

4. **Pitch Unit Conversion**
   - Convert F0 (Hz) to:
     - MIDI note numbers, or
     - Semitone scale around a reference (e.g., 440 Hz).
   - Optionally store both Hz and MIDI for later use.

5. **Diagnostic Visualization**
   - Plot:
     - Pitch contour over time.
     - Voiced/unvoiced regions.
   - Useful for route debugging and evaluation.

---

## 3. Inputs and Outputs

### Inputs

- Path to preprocessed audio file (from `02_preprocessing`).
- Sample rate and duration.

### Outputs

- `pitch_contour_hz`:
  - Array of F0 values in Hz (for each time frame).
- `pitch_contour_midi`:
  - Array of pitch values in MIDI space (optional but recommended).
- `voiced_mask`:
  - Binary array indicating voiced frames.
- `time_axis`:
  - Time values for each frame, derived from hop size and sample rate.

Example structure:

```text
{
  "time": [t0, t1, ..., tN],
  "f0_hz": [f0_0, f0_1, ..., f0_N],
  "f0_midi": [midi_0, ..., midi_N],
  "voiced": [0, 1, 1, ..., 0],
  "metadata": {
    "frame_size": <int>,
    "hop_size": <int>,
    "min_f0": <float>,
    "max_f0": <float>
  }
}
