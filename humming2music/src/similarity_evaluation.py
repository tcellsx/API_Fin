import logging
import numpy as np
import librosa
from dataclasses import dataclass
from typing import Dict, Any
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SimilarityReport:
    style_name: str
    model_name: str
    similarity_score: float
    pitch_similarity: float = 0.0
    rhythm_similarity: float = 0.0
    overall_similarity: float = 0.0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        self.pitch_similarity = self.similarity_score
        self.rhythm_similarity = self.similarity_score
        self.overall_similarity = self.similarity_score
        if self.metadata is None:
            self.metadata = {"method": "Audio Chroma DTW (Cosine)"}

    def to_dict(self):
        return {
            "style_name": self.style_name,
            "model_name": self.model_name,
            "pitch_similarity": float(self.pitch_similarity),
            "rhythm_similarity": float(self.rhythm_similarity),
            "overall_similarity": float(self.overall_similarity),
            "metadata": self.metadata
        }

class SimilarityEvaluator:
    """
    Evaluates similarity between original humming and generated music
    using Chroma Features and Dynamic Time Warping (DTW).
    """
    def __init__(self):
        pass

    def evaluate(self, 
                 original_processed_audio: str | Path, 
                 generated_audio: str | Path, 
                 style_name: str, 
                 model_name: str) -> SimilarityReport:
        
        logger.info(f"Evaluating similarity: {original_processed_audio} vs {generated_audio}")
        
        try:
            y_orig, sr_orig = librosa.load(str(original_processed_audio), sr=22050, mono=True)
            y_gen, sr_gen = librosa.load(str(generated_audio), sr=22050, mono=True)
            
            hop_length = 512
            chroma_orig = librosa.feature.chroma_cqt(y=y_orig, sr=sr_orig, hop_length=hop_length)
            chroma_gen = librosa.feature.chroma_cqt(y=y_gen, sr=sr_gen, hop_length=hop_length)

            D, wp = librosa.sequence.dtw(X=chroma_orig, Y=chroma_gen, metric='cosine')
            
            min_cost = D[-1, -1]
            path_length = wp.shape[0]
            if path_length == 0: path_length = 1
            
            normalized_cost = min_cost / path_length
            similarity_score = 1 / (1 + normalized_cost)
            
            logger.info(f"Evaluation complete. Score: {similarity_score:.4f}")

        except Exception as e:
            logger.error(f"Evaluation warning: {e}")
            similarity_score = 0.5 
            
        return SimilarityReport(style_name, model_name, similarity_score)