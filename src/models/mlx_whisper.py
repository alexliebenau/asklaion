import logging
import numpy as np
from typing import Dict, Any
from .core import BaseModelConfig, HuggingFaceModel, ModelSettings, ModelPerformance

# Example implementation for Whisper models
# FIXME: Add better way to pass customized settings
class MlxWhisperConfig(BaseModelConfig):
    _settings = {
        ModelPerformance.FAST: ModelSettings(
            model_path="mlx-community/whisper-tiny-mlx",
            quantized_path="mlx-community/whisper-tiny-8bit-mlx",
            inference_params={
                "beam_size": 1,
                "word_timestamps": False,
                "temperature": 0.0
            }
        ),
        ModelPerformance.BALANCED: ModelSettings(
            model_path="mlx-community/whisper-base.en-mlx",
            quantized_path="mlx-community/whisper-base-4bit-mlx",
            inference_params={
                "beam_size": 3,
                "word_timestamps": True,
                "temperature": 0.0
            }
        ),
        ModelPerformance.ACCURATE: ModelSettings(
            model_path="mlx-community/whisper-large-v3-mlx",
            quantized_path="mlx-community/whisper-large-v3-8bit-mlx",
            inference_params={
                "beam_size": 5,
                "word_timestamps": True,
                "temperature": 0.2
            }
        )
    }

class MLXWhisperModel(HuggingFaceModel):
    def __init__(self, 
                 performance: ModelPerformance, 
                 use_quantized: bool = False):
        try:
            import mlx_whisper
            self.mlx_whisper = mlx_whisper
        except ImportError:
            raise ImportError("Please install mlx-whisper: pip install mlx-whisper")
        self.use_quantized = use_quantized
        super().__init__(model_config=MlxWhisperConfig, performance=performance)
        

    def setup(self) -> None:
        """Initialize MLX Whisper model"""
        super().setup()
        try:
            model_path = (
                self.config.get_quantized_path(self.performance)
                if self.use_quantized
                else self.config.get_model_path(self.performance)
            )
            
            # Model will be loaded on first transcription
            self.model_path = model_path
            self.inference_params = self.config.get_inference_params(self.performance)

            # send in empty byte array to load model
            self.mlx_whisper.transcribe(
                np.zeros(8),
                path_or_hf_repo=model_path,
                **self.inference_params
            )
            
            logging.info(
                f"MLX Whisper configured with model: {model_path} "
                f"and parameters: {self.inference_params}"
            )
            
        except Exception as e:
            logging.error(f"Failed to configure MLX Whisper: {str(e)}")
            raise

    def query(self, audio_input: np.ndarray) -> Dict[str, Any]:
        """
        Transcribe audio using MLX Whisper
        
        Args:
            audio_input: numpy array containing raw bytes
            
        Returns:
            Dictionary containing:
            - text: transcribed text
            - segments: detailed segment information
            - language: detected language
        """
        if not self.is_ready:
            raise RuntimeError("Model not initialized")

        try:
            # MLX Whisper handles file paths directly
            result = self.mlx_whisper.transcribe(audio_input)

            return {
                'text': result['text'],
                'segments': result['segments'],
                'language': result['language']
            }

        except Exception as e:
            logging.error(f"Transcription failed: {str(e)}")
            raise
