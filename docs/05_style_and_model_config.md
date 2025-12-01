
---

```md
<!-- docs/05_style_and_model_config.md -->

# 05 – Style and Model Configuration Module

## 1. Purpose

Provide a **central place to define musical styles** and **map style choices** to the concrete parameters required by the music generation models.

This module does **not** generate audio; it:

- Defines style presets.
- Maps style presets to:
  - Text prompts (for text-conditioned models).
  - Tempo/energy/instrument hints.
  - Any model-specific configuration.

---

## 2. Responsibilities

1. **Style Definition**
   - Maintain a list of available styles, e.g.:
     - `"lofi"`, `"orchestral"`, `"8bit"`, `"rock"`, `"ambient"`, etc.
   - For each style, define:
     - Textual description (for prompts).
     - High-level parameters:
       - Tempo range.
       - Mood (e.g., “chill”, “epic”).
       - Instrumentation hints (e.g., guitars, strings, synths).

2. **Model Configuration Mapping**
   - For each style, specify how it maps to particular model settings:
     - Text-based prompt for text-conditioned models.
     - Control flags / tags.
     - Target duration overrides (if any).
   - Support multiple models by creating sub-configs per model:
     - Example:
       ```text
       {
         "model_name": "musicgen",
         "prompt_template": "A {mood}, {style} track with {instruments}..."
       }
       ```

3. **Interface for UI/Orchestrator**
   - Provide functions/structures for:
     - Listing available styles.
     - Getting full configuration for a chosen style ID.
   - Optionally support:
     - Default style.
     - Multi-style generation (list of styles).

---

## 3. Inputs and Outputs

### Inputs

- Style ID or style name, e.g. `"lofi"`.
- Optional model name (if multiple models exist).

### Outputs

- `style_config` object, e.g.:

```text
{
  "name": "lofi",
  "description": "Relaxed, mellow lofi hip-hop with soft drums and warm textures.",
  "mood": "chill",
  "tempo_bpm": 80,
  "instruments": ["soft drums", "electric piano", "vinyl noise"],
  "model_configs": {
    "musicgen": {
      "prompt": "A chill lofi hip-hop track with soft drums and warm textures.",
      "max_duration_sec": 20
    }
  }
}
