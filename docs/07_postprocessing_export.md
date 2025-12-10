
---

```md
<!-- docs/07_postprocessing_export.md -->

# 07 – Post-processing and Export Module

## 1. Purpose

Take raw generated audio from the music generation module and produce **polished, user-ready audio files**.

Tasks include:

- Loudness normalization.
- Basic dynamic control (optional).
- Fade-in/fade-out.
- Export to chosen file formats (e.g., WAV, MP3).

---

## 2. Responsibilities

1. **Loudness and Level Adjustment**
   - Normalize peak or RMS level to a target loudness.
   - Avoid clipping; optionally apply a simple limiter.

2. **Fade-in / Fade-out**
   - Apply short fades at the start and/or end to avoid clicks and abrupt edges.
   - Fade duration configurable (e.g., 50–200 ms).

3. **Resampling (if necessary)**
   - Ensure output sample rate matches user-facing standard if different from generator output.

4. **File Export**
   - Export in one or more formats:
     - Lossless: WAV.
     - Compressed: MP3 (if desired).
   - Use consistent naming:
     - `<input_id>_<style_name>_<model_name>.wav`.

5. **Metadata and Logging**
   - Provide final audio duration, sample rate, format.
   - Log post-processing steps for reproducibility.

---

## 3. Inputs and Outputs

### Inputs

- Raw generated audio descriptor from `06_music_generation`:
  - Path to intermediate audio file or audio array.
  - Metadata (model, style, initial duration).

### Outputs

- Final exported file(s), e.g.:

```text
{
  "final_audio_path": "<final-wav-path>",
  "final_audio_path_mp3": "<final-mp3-path-if-any>",
  "duration_sec": <float>,
  "sample_rate": <int>,
  "postprocessing_applied": ["normalize", "fade_in", "fade_out"],
  "style_name": "<style>",
  "model_name": "<model>"
}
