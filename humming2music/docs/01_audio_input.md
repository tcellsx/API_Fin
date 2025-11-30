
---

```md
<!-- docs/01_audio_input.md -->

# 01 – Audio Input Module

## 1. Purpose

Handle **user audio acquisition** in a clean, consistent way:

- Allow users to **record** or **upload** humming/vocal audio.
- Normalize audio format (sample rate, channels, file type).
- Enforce basic constraints (duration, size, supported formats).
- Provide metadata for downstream modules.

This module is the **entry point** of the pipeline.

---

## 2. Responsibilities

1. **Audio Recording Interface**
   - Support recording from microphone (via Notebook cell or GUI control).
   - Save recording to `RAW_AUDIO_DIR` with a standardized naming scheme:
     - Example: `raw_input_<timestamp>_session<id>.wav`.

2. **File Upload Interface**
   - Accept external audio files from the user.
   - Supported file formats (configurable), e.g.: `wav`, `mp3`, `m4a`, `flac`.
   - Store copies in `RAW_AUDIO_DIR`, potentially with conversion.

3. **Format Normalization**
   - Convert all inputs to:
     - Fixed sample rate (e.g., 16 kHz or 22.05 kHz).
     - Mono (single channel).
     - WAV format (recommended for processing).
   - Ensure that the normalized audio adheres to global configuration:
     - `GLOBAL.SAMPLE_RATE`
     - `GLOBAL.MONO`

4. **Basic Validation**
   - Check duration:
     - Minimum duration (e.g., 0.5 s).
     - Maximum duration (`GLOBAL.MAX_INPUT_DURATION`, e.g., 30 s).
   - Handle invalid or corrupted files:
     - Provide informative error messages or warnings.

5. **Metadata Extraction**
   - Extract basic metadata:
     - Duration.
     - Sample rate.
     - Bit depth / encoding.
   - Return as a structured object for logging and downstream use.

---

## 3. Inputs and Outputs

### Inputs

- **User actions**:
  - Start/stop recording.
  - File selection/drag-and-drop upload.

### Outputs

- **Normalized audio file path**
  - e.g., `./data/raw/raw_input_2025-01-01_12-00-00.wav`.

- **Audio metadata** :
  ```text
  {
    "path": "<full-path>",
    "duration_sec": <float>,
    "sample_rate": <int>,
    "channels": 1,
    "format": "wav",
    "source_type": "recorded" | "uploaded"
  }
