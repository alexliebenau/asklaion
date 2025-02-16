import logging
import numpy as np
from typing import Dict, Any
from .core import BaseModelConfig, HuggingFaceModel, ModelSettings, ModelPerformance

class MlxLlamaConfig(BaseModelConfig):
    _settings = {
        ModelPerformance.FAST: ModelSettings(
            model_path="mlx-community/Llama-3.1-8B-Instruct-mlx",
            quantized_path="mlx-community/Llama-3.1-8B-Instruct-4bit-mlx",
            inference_params={
                "temperature": 0.3,
                "top_p": 0.95,
                "max_tokens": 512,
                "repetition_penalty": 1.05,
                "min_p": 0.05
            }
        ),
        ModelPerformance.BALANCED: ModelSettings(
            model_path="mlx-community/Llama-3.1-70B-Instruct-mlx",
            quantized_path="mlx-community/Llama-3.1-70B-Instruct-4bit-mlx",
            inference_params={
                "temperature": 0.7,
                "top_p": 0.90,
                "max_tokens": 2048,
                "repetition_penalty": 1.02,
                "min_p": 0.10
            }
        ),
        ModelPerformance.ACCURATE: ModelSettings(
            model_path="mlx-community/Llama-3.1-405B-Instruct-mlx",
            quantized_path="mlx-community/Llama-3.1-405B-Instruct-4bit-mlx",
            inference_params={
                "temperature": 0.9,
                "top_p": 0.85,
                "max_tokens": 4096,
                "repetition_penalty": 1.01,
                "min_p": 0.15
            }
        )
    }


class MLXLlamaModel(HuggingFaceModel):
    def __init__(self, 
                 speed: ModelPerformance, 
                 use_quantized: bool = False):
        try:
            import mlx_lm
            self.mlx_lm = mlx_lm
        except ImportError:
            raise ImportError("Please install mlx-llama: pip install mlx-lm")
            
        super().__init__(model_config=MlxLlamaConfig, speed=speed)
        self.use_quantized = use_quantized
        self.tokenizer = None

    def setup(self) -> None:
        """Initialize MLX Llama model"""
        super().setup()
        try:
            model_path = (
                self.config.get_quantized_path(self.performance)
                if self.use_quantized
                else self.config.get_model_path(self.performance)
            )
            
            # Load model and tokenizer
            self.model, self.tokenizer = self.mlx_lm.load(
                path_or_hf_repo=model_path,
                **self.config.get_inference_params(self.performance)
            )
            
            logging.info(
                f"MLX Llama configured with model: {model_path} "
                f"and parameters: {self.config.get_inference_params(self.performance)}"
            )
            
        except Exception as e:
            logging.error(f"Failed to configure MLX Llama: {str(e)}")
            raise

    def query(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Generate text using MLX Llama
        
        Args:
            prompt: Input text prompt
            **kwargs: Additional generation parameters
            
        Returns:
            Dictionary containing:
            - text: generated text
            - tokens: number of generated tokens
            - finish_reason: termination reason
            - logprobs: token probabilities (if enabled)
        """
        if not self.is_ready:
            raise RuntimeError("Model not initialized")

        try:
            # Merge class params with runtime params
            params = {**self.config.get_inference_params(self.performance), **kwargs}
            
            # Tokenize input
            inputs = self.tokenizer.encode(prompt, return_tensors="np")
            
            # Generate output
            outputs = self.model.generate(
                inputs,
                max_length=params.get("max_tokens", 512),
                temperature=params.get("temperature", 0.7),
                top_p=params.get("top_p", 0.9),
                repetition_penalty=params.get("repetition_penalty", 1.0)
            )
            
            # Decode results
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            return {
                'text': generated_text,
                'tokens': len(outputs[0]),
                'finish_reason': 'length' if len(outputs[0]) >= params.get("max_tokens", 512) else 'stop',
                'logprobs': outputs.scores if params.get("return_logprobs", False) else None
            }

        except Exception as e:
            logging.error(f"Generation failed: {str(e)}")
            raise