from pathlib import Path
import os
from huggingface_hub import snapshot_download, HfApi
from abc import ABC, abstractmethod
from typing import Any, Dict, Any, ClassVar, Optional
from dataclasses import dataclass
from enum import Enum
import logging

class ModelPerformance(Enum):
    FAST = "fast"
    BALANCED = "balanced"
    ACCURATE = "accurate"

@dataclass(frozen=True)
class ModelSettings:
    model_path: str
    quantized_path: str | None = None
    inference_params: Dict[str, Any] | None = None


class BaseModelConfig:
    """Base configuration class for models"""
    _settings: ClassVar[Dict[ModelPerformance, ModelSettings]] = {}
    
    @classmethod
    def get_model_path(cls, performance: ModelPerformance) -> str:
        return cls._get_settings(performance).model_path
    
    @classmethod
    def get_quantized_path(cls, performance: ModelPerformance) -> str | None:
        return cls._get_settings(performance).quantized_path
    
    @classmethod
    def get_inference_params(cls, performance: ModelPerformance) -> Dict[str, Any]:
        return cls._get_settings(performance).inference_params or {}
    
    @classmethod
    def get_all_settings(cls, performance: ModelPerformance) -> ModelSettings:
        return cls._get_settings(performance)
    
    @classmethod
    def _get_settings(cls, performance: ModelPerformance) -> ModelSettings:
        if not isinstance(performance, ModelPerformance):
            raise ValueError(f"Invalid performance preset. Must be one of {[s.value for s in ModelPerformance]}")
        
        settings = cls._settings.get(performance)
        if settings is None:
            raise NotImplementedError(f"Settings for performance preset '{performance.value}' are not implemented")
        
        return settings


class BaseModel(ABC):
    """Abstract base class for all models"""
    
    def __init__(self, 
                 model_config: BaseModelConfig, 
                 performance: ModelPerformance,
                 cache_path: Optional[str]):
        self.config = model_config
        self.performance = performance
        self.model: Any = None
        self.is_ready: bool = False
        # FIXME: Add better / global cache for models

        self.model_cache = cache_path or Path(os.path.abspath(__file__)).parent / 'cache'
        self._setup()

    def _setup(self) -> None:
        """Internal setup method that ensures the model is initialized only once"""
        logging.info(f'Model download cache: {self.model_cache}')
        if not self.is_ready:
            try:
                self.setup()
                self.is_ready = True
                logging.info(f"Model initialized successfully with performance preset: {self.performance.value}")
            except Exception as e:
                logging.error(f"Failed to initialize model: {str(e)}")
                raise

    @abstractmethod
    def setup(self) -> None:
        """Initialize and prepare the model for inference"""
        pass

    @abstractmethod
    def query(self, input_data: Any) -> Dict[str, Any]:
        """Process input data and return results"""
        if not self.is_ready:
            raise RuntimeError("Model not initialized. Please ensure setup() is called first.")
        pass

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"\n\tperformance: {self.performance.value}"
            f"\n\tpath: {self.model_cache / self.config.get_model_path(self.performance)}"
            f"\n\tready: {self.is_ready})"
            )
    

class HuggingFaceModel(BaseModel):

    def __init__(self, 
                 model_config: BaseModelConfig, 
                 performance: ModelPerformance,
                 cache_path: Optional[str] = None):
        super().__init__(model_config, performance, cache_path)

    def setup(self) -> Path:
        """Validate local model structure with enhanced checks"""
        model_name = self.config.get_model_path(self.performance)
        model_dir = self.model_cache / f"{model_name.replace('/', '--')}"
        model_dir 
        
        if not model_dir.exists():
            # Download with model type validation
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

        required_files = {"config.json", "weights.npz"}
        missing = required_files - set(os.listdir(model_dir))
        if missing:
            raise FileNotFoundError(
                f"Model incomplete in {model_dir}. Missing files: {missing}"
            )
