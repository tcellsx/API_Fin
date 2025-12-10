import logging
import torch
import numpy as np
import scipy.io.wavfile
import pretty_midi
from pathlib import Path
from transformers import AutoProcessor, MusicgenForConditionalGeneration

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MusicGenerator:
    def __init__(self, model_size='melody', device='cpu'):
        self.device = device
        self.model_id = f"facebook/musicgen-{model_size}"
        
        logger.info(f"Loading MusicGen model: {self.model_id} on {self.device}...")
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = MusicgenForConditionalGeneration.from_pretrained(self.model_id)
        self.model.to(self.device)
        logger.info("Model loaded.")

    def generate(self, melody_representation, style_name, prompt_text=None, 
                 melody_audio_path=None, duration_sec=10.0, output_dir="outputs/generated"):
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if not prompt_text: prompt_text = f"A music track in {style_name} style"
        logger.info(f"Processing inputs | Prompt: '{prompt_text}'")

        target_sr = 32000
        inputs = None
        
        midi_path = melody_representation.get('midi_path')
        
        if midi_path and Path(midi_path).exists():
            logger.info(f"Detected MIDI input from Basic Pitch: {midi_path}")
            logger.info("Synthesizing MIDI to clean audio for MusicGen conditioning...")
            
            pm = pretty_midi.PrettyMIDI(midi_path)
            
            synthesized_audio = pm.synthesize(fs=target_sr)
            
            max_len = int(target_sr * 30) 
            if len(synthesized_audio) > max_len:
                synthesized_audio = synthesized_audio[:max_len]
            
            audio_tensor = torch.tensor(synthesized_audio).float()
            audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0) # [1, 1, T]
            
            inputs = self.processor(
                text=[prompt_text],
                padding=True,
                return_tensors="pt"
            )
            inputs["input_values"] = audio_tensor
            
        elif melody_audio_path:

            logger.info(f"No MIDI found, falling back to raw audio: {melody_audio_path}")
            import librosa
            audio_values, _ = librosa.load(melody_audio_path, sr=target_sr, mono=True)
            if len(audio_values) > target_sr * 30:
                audio_values = audio_values[:target_sr * 30]
            
            audio_tensor = torch.tensor(audio_values).float().unsqueeze(0).unsqueeze(0)
            
            inputs = self.processor(
                text=[prompt_text],
                padding=True,
                return_tensors="pt"
            )
            inputs["input_values"] = audio_tensor
        else:
            inputs = self.processor(
                text=[prompt_text],
                padding=True,
                return_tensors="pt"
            )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        if "attention_mask" in inputs and "input_values" in inputs:
             del inputs["attention_mask"]

        max_new_tokens = int(duration_sec * 50)
        logger.info("Generating music...")
        
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "guidance_scale": 3.0
        }
        if "input_values" in inputs:
            gen_kwargs["input_values"] = inputs["input_values"]
        if "input_ids" in inputs:
            gen_kwargs["input_ids"] = inputs["input_ids"]

        audio_outputs = self.model.generate(**gen_kwargs)

        sampling_rate = self.model.config.audio_encoder.sampling_rate
        audio_data = audio_outputs[0, 0].cpu().numpy()
        
        import time
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"gen_{style_name}_{timestamp}.wav"
        full_path = output_dir / filename
        scipy.io.wavfile.write(full_path, rate=sampling_rate, data=audio_data)
        logger.info(f"Saved to: {full_path}")

        return SimpleResult("musicgen-transformers", style_name, str(full_path), duration_sec, {"prompt": prompt_text})

class SimpleResult:
    def __init__(self, model_name, style_name, audio_path, duration, metadata):
        self.model_name = model_name
        self.style_name = style_name
        self.audio_path = audio_path
        self.duration_sec = duration
        self.metadata = metadata
    def to_dict(self):
        return {"model_name": self.model_name, "style_name": self.style_name, 
                "audio_path": self.audio_path, "duration_sec": self.duration_sec, 
                "generation_metadata": self.metadata}