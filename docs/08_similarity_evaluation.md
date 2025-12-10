
---

```md
<!-- docs/08_similarity_evaluation.md -->

# 08 – Similarity and Evaluation Module

## 1. Purpose

Evaluate **how similar** the generated tracks are to the original humming in terms of **melody** and, optionally, **rhythm**.

Main goals:

- Provide quantitative scores (e.g., 0–1 or 0–100).
- Provide visual comparisons (pitch contours, note sequences).
- Support comparison across multiple style versions.

---

## 2. Responsibilities

1. **Re-Extraction of Melody from Generated Audio**
   - Use the same melody extraction pipeline (`03_melody_extraction`) on:
     - Original preprocessed input.
     - Each generated track.
   - Ensures fairness of comparison.

2. **Alignment and Distance Metrics**
   - Compare:
     - Original melody representation (from `04`).
     - Generated melody representation (also from `04`).
   - Compute:
     - Pitch contour distance (e.g., DTW over pitch sequences).
     - Optional note sequence edit distance / overlap measures.
   - Normalize distances into similarity scores:
     - Higher score = greater melodic similarity.

3. **Rhythmic Similarity**
   - Compare rhythm profiles:
     - Duration histograms.
     - Quantized patterns.
   - Produce a separate rhythm similarity score if desired.

4. **Aggregate Similarity Scoring**
   - Combine:
     - Pitch similarity.
     - Rhythm similarity.
   - Into an overall score (e.g., weighted sum).

5. **Visualization**
   - Plot:
     - Original vs generated pitch contour over time.
     - Optional piano-roll representations.
   - Provide tabular summaries:
     - For each style:
       - Pitch similarity score.
       - Rhythm similarity score.
       - Overall similarity.

---

## 3. Inputs and Outputs

### Inputs

- Original melody representation and/or original audio:
  - From `03` and `04` (and possibly `02` for the processed audio file path).

- Generated tracks:
  - From `07_postprocessing_export` (final audio paths).

### Outputs

- Per-track evaluation report, e.g.:

```text
{
  "style_name": "<style>",
  "model_name": "<model>",
  "pitch_similarity": 0.82,
  "rhythm_similarity": 0.75,
  "overall_similarity": 0.79,
  "plots": {
    "pitch_contour_comparison": "<path-or-reference>",
    "pianoroll_comparison": "<optional>"
  }
}
