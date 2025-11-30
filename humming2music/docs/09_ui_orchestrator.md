
---

```md
<!-- docs/09_ui_orchestrator.md -->

# 09 – UI / Orchestrator Module

## 1. Purpose

Provide the **user-facing entry point** and orchestrate calls between all modules.

Two levels of UI:

1. **Jupyter Notebook pipeline** – primary development and debugging interface.
2. **Optional simple GUI** – higher-level, user-friendly interface.

---

## 2. Responsibilities

1. **Notebook Pipeline Orchestration**
   - Define a clear linear workflow:
     1. Audio acquisition (`01_audio_input`).
     2. Preprocessing (`02_preprocessing`).
     3. Melody extraction (`03_melody_extraction`).
     4. Melody representation (`04_melody_representation`).
     5. Style selection (`05_style_and_model_config`).
     6. Music generation (`06_music_generation`).
     7. Post-processing and export (`07_postprocessing_export`).
     8. Similarity evaluation (`08_similarity_evaluation`).
   - Each step displayed as a separate Notebook section/cell.
   - Show intermediate results:
     - Waveforms, pitch curves, note sequences.
     - Logs and configuration values used.

2. **Simple GUI (Optional)**
   - Provide:
     - Audio input controls:
       - Record button.
       - File upload.
     - Style selection:
       - Drop-down menu for styles.
       - Possibly checkboxes for multiple styles.
     - Generate button.
   - Result display:
     - Audio player(s) for:
       - Original input.
       - Each generated style.
     - Similarity scores in a simple table.
     - Key visualization(s) (e.g., pitch contour comparison).

3. **Session Management**
   - Assign session or input IDs.
   - Keep track of:
     - Associated raw audio.
     - Generated outputs.
     - Evaluation results.
   - Useful for logging and reproducibility.

4. **Configuration Display**
   - Optionally display or log:
     - Important configuration values used in the current run:
       - Sample rate.
       - Max input duration.
       - Pitch extraction parameters.

---

## 3. Inputs and Outputs

### Inputs

- User actions:
  - Recording/upload.
  - Style selection.
  - “Generate” command.

### Outputs

- Chained calls through all internal modules.
- Final artifacts exposed to user:
  - Audio players.
  - Downloadable audio files.
  - Visualization plots.
  - Similarity scores.

---

## 4. Interaction with Other Modules

The Orchestrator coordinates the entire pipeline:

1. **Audio Input** → `01_audio_input`
2. **Preprocessing** → `02_preprocessing`
3. **Melody Extraction** → `03_melody_extraction`
4. **Melody Representation** → `04_melody_representation`
5. **Style Selection & Config** → `05_style_and_model_config`
6. **Music Generation** → `06_music_generation`
7. **Post-processing & Export** → `07_postprocessing_export`
8. **Similarity Evaluation** → `08_similarity_evaluation`

Each step uses the outputs of the previous step as its inputs; orchestrator ensures correct ordering and data passing.

---

## 5. Configuration Parameters

From configuration:

- `GLOBAL` – All global settings (sample rate, max duration, etc.).
- `UI.SHOW_INTERMEDIATE_PLOTS`
  - Whether to render diagnostic plots in Notebook/GUI.
- `UI.ALLOWED_STYLES`
  - Subset of styles exposed in the interface.

---

## 6. Edge Cases and Error Handling

- Missing or failed module output:
  - Orchestrator should halt pipeline and show clear error messages.
  - E.g., “Melody extraction failed due to invalid input.”

- User cancels or changes input mid-pipeline:
  - Provide a way to reset state and start a new session.

---

## 7. Testing Guidelines

- Run complete pipeline on:
  - Multiple different inputs (clean, noisy).
  - Different styles.
- Confirm:
  - No missing links between modules.
  - Errors are surfaced clearly.
  - User can follow a simple, linear workflow without confusion.

---

## 8. Notebook Structuring Suggestions

- Use a top-level outline like:

  1. Setup & Configuration
  2. Audio Input
  3. Preprocessing
  4. Melody Extraction
  5. Melody Representation
  6. Style Selection
  7. Music Generation
  8. Post-processing & Export
  9. Similarity Evaluation
  10. Summary & Discussion

- Each section:
  - Shows current configuration values relevant to that step.
  - Logs important intermediate outputs.
  - Optionally saves results to disk for reuse.
