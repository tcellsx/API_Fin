
---

```md
<!-- docs/06_music_generation.md -->

# 06 – Music Generation Module

## 1. Purpose

Transform the **melody representation + style configuration** into a **full musical track** using one or more music generation models.

This module:

- Builds model-specific input conditions.
- Calls models to generate audio.
- Handles multiple style variants per input.

---

## 2. Responsibilities

1. **Condition Construction**
   - Combine:
     - `melody_representation` (note sequence, rhythm, embedding).
     - `style_config` (from `05_style_and_model_config`).
   - Build model-specific inputs:
     - Text prompt (for text-conditioned models).
     - Optional symbolic track/melody guide if supported.
     - Target duration, tempo hints, etc.

2. **Model Abstraction Layer**
   - Define a common interface:
     - `generate(melody, style_config, model_name) -> audio`.
   - Implement adapters for each underlying model:
     - `MusicGenAdapter`, `RiffusionAdapter`, etc.
   - Each adapter:
     - Receives standardized inputs.
     - Produces standardized outputs (audio array or audio file).

3. **Multi-Style Generation (Optional)**
   - Allow generating multiple versions for the same input:
     - E.g., “lofi”, “orchestral”, “8bit” all in one go.
   - Manage naming and output directory structure accordingly.

4. **Basic Quality and Safety Checks**
   - Limit generation duration (max seconds).
   - Handle model timeouts or exceptions gracefully.
   - Ensure output sample rate compatibility with post-processing.

---

## 3. Inputs and Outputs

### Inputs

- `melody_representation` from `04_melody_representation`:
  - `note_sequence`, `rhythm_profile`, optional `embedding`.

- `style_config` from `05_style_and_model_config`:
  - Contains mood, tempo hints, model-specific config.

- `model_name` (optional):
  - If not specified, use `GENERATION.DEFAULT_MODEL_NAME`.

### Outputs

- One or more generated audio tracks, each with:

```text
{
  "audio_path": "<path-to-generated-file>",
  "model_name": "<model>",
  "style_name": "<style>",
  "duration_sec": <float>,
  "sample_rate": <int>,
  "generation_metadata": {
    "prompt": "<final-text-prompt-if-used>",
    "seed": <int>,
    "timestamp": "<datetime>",
    ...
  }
}
