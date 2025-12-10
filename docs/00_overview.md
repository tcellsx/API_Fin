<!-- docs/00_overview.md -->

# 00 – Project Overview

## 1. Project Goal

This project builds a prototype system that transforms **raw humming or vocal recordings** into **usable, styled musical tracks**.

End-to-end pipeline (first implemented as a Jupyter Notebook, later optionally wrapped in a simple GUI):

1. User records or uploads an audio snippet (humming / singing).
2. System preprocesses the audio (denoising, trimming, normalization).
3. System extracts the melody (pitch contour and rhythm) from the vocal.
4. System represents the extracted melody in a structured form (notes, rhythm profile, optional embedding).
5. User selects a musical style (e.g., lofi, orchestral, 8-bit, rock).
6. A music generation model uses the melody + style to generate a full track.
7. System post-processes and exports the generated audio in user-friendly formats.
8. System evaluates similarity between the original humming and the generated track (melodic similarity).
9. User listens to and compares different style versions.

---

## 2. Architecture

Core modules (each has its own documentation):

- `01_audio_input` – Recording or uploading input audio, format normalization.
- `02_preprocessing` – Noise reduction, silence trimming, loudness normalization.
- `03_melody_extraction` – Pitch tracking, voiced/unvoiced detection, smoothing.
- `04_melody_representation` – Note segmentation, rhythmic profile, symbolic representation, optional embedding.
- `05_style_and_model_config` – Style definitions and mapping to model parameters.
- `06_music_generation` – Conditional music generation based on melody + style.
- `07_postprocessing_export` – Final loudness shaping, fade-in/out, exporting.
- `08_similarity_evaluation` – Melodic similarity metrics and visualizations.
- `09_ui_orchestrator` – Jupyter Notebook pipeline and (later) simple GUI that ties everything together.

Data flow (conceptual):

```text
AudioInput
  → Preprocessing
  → MelodyExtraction
  → MelodyRepresentation
  → StyleAndModelConfig
  → MusicGeneration
  → PostprocessingExport
  → SimilarityEvaluation
  → UIOrchestrator (for display and user interaction)

[UI / Notebook] 
   ↓
[Audio Input]      (录音 / 上传)
   ↓
[Preprocessing]    (降噪、音量 & 截断)
   ↓
[Melody Extraction] (pitch轨迹 + 节奏)
   ↓
[Melody Representation] (MIDI / note 序列等)
   ↓
[Style & Model Config]  (选择风格 & 模型)
   ↓
[Music Generation]      (条件生成整段音乐)
   ↓
[Post-processing]       (淡入淡出、归一、导出)
   ↓
[Similarity & Evaluation] (与原始哼唱的相似度)
   ↓
[Result UI]             (播放器 + 多风格对比)
