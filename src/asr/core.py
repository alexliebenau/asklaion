from pathlib import Path
import numpy as np
import os
from huggingface_hub import snapshot_download, HfApi
from abc import ABC, abstractmethod
from typing import Any

class BaseTranscriber(ABC):
    
    @abstractmethod
    def transcribe(self, audio_data: np.ndarray) -> dict:
        pass

    @abstractmethod
    def setup(self, setup_config: dict[str, Any]) -> None:
        '''Loads and prepares model to make it ready for ASR'''
        pass


class HuggingFaceTranscriber(BaseTranscriber):

    def _get_model_path(self, model_name: str) -> Path:
        """Validate local model structure with enhanced checks"""
        model_dir = Path(f"mlx_models/{model_name.replace('/', '--')}")
        
        if not model_dir.exists():
            self._download_model(model_name, model_dir)

        # Check for tokenizer in multiple locations
        tokenizer_paths = [
            model_dir / "tokenizer.json",
            model_dir / "tokenizers/tokenizer.json",
            model_dir / "tokenizer/tokenizer.json"
        ]
        
        if not any(p.exists() for p in tokenizer_paths):
            raise FileNotFoundError(
                f"Tokenizer missing in {model_dir}. Valid paths tried: {tokenizer_paths}"
            )

        required_files = {"config.json", "weights.npz"}
        missing = required_files - set(os.listdir(model_dir))
        if missing:
            raise FileNotFoundError(
                f"Model incomplete in {model_dir}. Missing files: {missing}"
            )
            
        return model_dir

    def _download_model(self, model_name: str, model_dir: Path):
        """Download with model type validation"""
        
        api = HfApi()
        try:
            model_info = api.model_info(model_name)
            if not any(f.rfilename == "config.json" for f in model_info.siblings):
                raise ValueError(f"{model_name} is not a valid MLX Whisper model")
        except Exception as e:
            raise ValueError(f"Invalid model {model_name}: {str(e)}")

        snapshot_download(
            repo_id=model_name,
            local_dir=model_dir,
            allow_patterns=["*.json", "*.npz", "*.md", "*.tiktoken"],
            local_dir_use_symlinks=False,
            resume_download=True
        )

class Transcriber:
    def __init__(
        self, 
        model_name: str,
        dtype: str = "bfloat16",
        beam_size: int = 5,
        word_timestamps: bool = False,
        temperature: float = 0.0
    ):
        self.model_path = self._get_model_path(model_name)
        self.model = load_models.load_model(
            str(self.model_path),
            dtype=mlx.core.float16
        )
        self.config = {
            "beam_size": beam_size,
            "word_timestamps": word_timestamps,
            "temperature": temperature
        }



    def transcribe(self, audio_data: np.ndarray) -> dict:
        """Use pre-loaded model for transcription"""
        return self.model.transcribe(
            audio=audio_data.astype(np.float32) / 32768.0,
            model=self.model,
            **self.config
        )
    
def create_transcriber(preset: str = "balanced", quantized: bool = False) -> Transcriber:
    '''Creates a `Transcriber` class that loads a MLX whisper model based on a 
    preset of 'fast', 'balanced' or 'accurate'. Choose your poison between performance
    and runtime.
    '''
    model_source = QUANTIZED_MODELS if quantized else DEFAULT_MODELS
    config = MODEL_PRESETS[preset]
    # config['files'] = ["config.json", "weights.npz", "tokenizer.json"]
    
    return Transcriber(
        model_name=model_source[preset],
        **config
    )