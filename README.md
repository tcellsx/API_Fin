# Humming to Music

A prototype system that transforms hummed melodies into styled musical tracks. This project was developed for the Audio Processing and Indexing course (2526-S1) at Leiden University.

## Overview

The system implements an end-to-end pipeline that:
1. Accepts humming or vocal recordings as input
2. Extracts the melodic pitch contour using neural pitch detection
3. Converts continuous pitch to discrete note sequences
4. Generates styled music based on the extracted melody
5. Evaluates similarity between input and output

## Project Structure

```
humming2music/
├── src/                    # Source code modules
│   ├── audio_input.py      # Audio recording and file ingestion
│   ├── preprocessing.py    # Noise reduction, normalization
│   ├── melody_extraction.py    # Pitch tracking (Basic Pitch)
│   ├── melody_representation.py # Note segmentation
│   ├── style_and_model_config.py # Style presets
│   ├── music_generation.py # Music generation adapters
│   ├── postprocessing_export.py # Final audio polishing
│   ├── similarity_evaluation.py # DTW-based evaluation
│   └── config.py           # Shared configuration
├── data/
│   ├── raw/                # Input audio files
│   └── processed/          # Preprocessed audio
├── outputs/
│   ├── generated/          # Generated audio
│   └── final/              # Post-processed final outputs
└── notebooks/
    └── demo_pipeline.ipynb # Interactive demonstration
```

## Installation

```bash
# Clone the repository
git clone https://github.com/tcellsx/API_Fin.git
cd API_Fin/humming2music

# Install dependencies
pip install numpy librosa pydub sounddevice basic-pitch
```

**Note:** For MP3 export functionality, you also need [FFmpeg](https://ffmpeg.org/) installed on your system.

## Quick Start

### Using the Demo Notebook

Open `notebooks/demo_pipeline.ipynb` in Jupyter and follow the step-by-step demonstration.

### Using the Python API

```python
from src.audio_input import AudioInputManager
from src.preprocessing import Preprocessor
from src.melody_extraction import MelodyExtractor
from src.melody_representation import MelodyRepresenter
from src.music_generation import MusicGenerator
from src.postprocessing_export import Postprocessor
from src.similarity_evaluation import SimilarityEvaluator

# 1. Ingest audio
manager = AudioInputManager()
audio_meta = manager.ingest_upload("path/to/your/humming.wav")

# 2. Preprocess
preprocessor = Preprocessor()
prep_meta = preprocessor.preprocess(audio_meta.path)

# 3. Extract melody
extractor = MelodyExtractor()
contour = extractor.extract(prep_meta.path)

# 4. Convert to notes
representer = MelodyRepresenter()
melody_rep = representer.represent(
    time=contour.time,
    f0_midi=contour.f0_midi,
    voiced=contour.voiced
)

# 5. Generate music
generator = MusicGenerator()
result = generator.generate(
    melody_representation=melody_rep.to_dict(),
    style_name="lofi"  # Options: lofi, orchestral, 8bit, rock, ambient
)

# 6. Post-process
postprocessor = Postprocessor()
final = postprocessor.process(
    result.audio_path,
    style_name=result.style_name,
    model_name=result.model_name
)

print(f"Output saved to: {final.final_audio_path}")
```

## Supported Styles

| Style | Description | Tempo |
|-------|-------------|-------|
| lofi | Relaxed hip-hop with soft drums and warm textures | 80 BPM |
| orchestral | Cinematic with strings, brass, and percussion | 110 BPM |
| 8bit | Retro chiptune with square wave leads | 130 BPM |
| rock | Energetic with guitars and punchy drums | 120 BPM |
| ambient | Ethereal soundscape with pads and drones | 70 BPM |

## Technical Details

### Pitch Extraction
The system uses Basic Pitch, a neural network-based pitch detection model developed by Spotify, for robust F0 estimation even in noisy conditions.

### Similarity Evaluation
- **Pitch Similarity**: Dynamic Time Warping (DTW) on MIDI pitch sequences
- **Rhythm Similarity**: Levenshtein edit distance on quantized note patterns
- **Overall Score**: Weighted combination (70% pitch, 30% rhythm)

## Future Work

- Integration of neural music generation models (MusicGen, Riffusion)
- Comparison with alternative pitch tracking models (CREPE)
- Perceptual evaluation through user studies


## License

This project is developed for educational purposes as part of the Audio Processing and Indexing course at Leiden University.
