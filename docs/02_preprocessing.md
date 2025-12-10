
---

```md
<!-- docs/02_preprocessing.md -->

# 02 – Preprocessing Module

## 1. Purpose

Convert raw humming/vocal input into a **clean, stable signal** suitable for melody extraction.

Goals:

- Remove leading/trailing silence.
- Reduce background noise.
- Normalize loudness.
- Optionally apply simple filtering.

---

## 2. Responsibilities

1. **Silence Trimming**
   - Detect and remove low-energy regions at the start and end of the recording.
   - Threshold configurable in dB (e.g., -40 dB).
   - Avoid removing intentional low-volume humming.

2. **Noise Reduction (Optional)**
   - Apply basic noise reduction techniques:
     - High-pass filter (remove low-frequency rumble).
     - Optional low-pass filter (remove very high-frequency noise).
     - Simple noise gate if needed.
   - Keep the implementation simple enough to be robust.

3. **Loudness Normalization**
   - Align loudness to a target level (e.g., -20 dB RMS or similar).
   - Avoid clipping and maintain dynamic range as much as possible.

4. **Length Enforcement (Optional)**
   - After trimming:
     - If audio is still too long (> `MAX_INPUT_DURATION`), truncate.
     - If too short but above minimum threshold, still accept.

5. **Diagnostic Visualization (for Notebook)**
   - Provide:
     - Before/after waveform plots.
     - Optional spectrogram views.
   - Not required in production, but helpful for research/debugging.

---

## 3. Inputs and Outputs

### Inputs

- Path to normalized raw audio file (from `01_audio_input`).
- Audio metadata (sample rate, duration).

### Outputs

- Path to **preprocessed audio file**, stored in `PROCESSED_AUDIO_DIR`.
- Updated metadata (duration after trimming, loudness level).
- Optional diagnostic information (for logging/Notebook display).

Example metadata:

```text
{
  "path": "<processed-audio-path>",
  "original_duration_sec": <float>,
  "processed_duration_sec": <float>,
  "sample_rate": <int>,
  "applied_steps": ["trim_silence", "normalize", "highpass"],
  "notes": "Noise reduction skipped due to short duration."
}
