import json 
import pyaudio
import numpy as np
from scipy import signal
from dataclasses import dataclass
from typing import Generator, Optional, Any

'''
The Default AudioConfig already provides the correct settings
which are expected by typical ASR models.

You may change it to match your current audio setup, however that might 
have an impact on transcribtion performance.
'''

@dataclass
class AudioConfig:
    sample_size: int = 16 # or 24 or 32 [bit]
    channels: int = 1
    rate: int = 16000
    chunk: int = 2048
    silence_threshold: int = 400
    silence_consecutive: int = 5
    input_device: int = None

    @property
    def format(self) -> int:
        if self.sample_size == 16: return pyaudio.paInt16
        elif self.sample_size == 24: return pyaudio.paInt24
        elif self.sample_size == 32: return pyaudio.paFloat32
        else:
            raise ValueError(f'Invalid config for sample_size: {self.sample_size}. Choose 16, 24 or 32 [bit]')


class AudioConverter:
    """Converts audio chunks between different audio configurations"""

    @staticmethod
    def convert_bit_format(audio_chunk: np.ndarray, 
                           source_format: int, 
                           target_format: int
                           ) -> np.ndarray:
        """
        Convert audio between different bit formats
        
        Supports conversion between:
        - 16-bit int (paInt16)
        - 24-bit int (paInt24)
        - 32-bit float (paFloat32)
        """
        # Define max values for integer formats
        INT16_MAX = 32767.0  # 2^15 - 1
        INT24_MAX = 8388607.0  # 2^23 - 1
        
        # First convert to float32 as intermediate format
        if source_format == pyaudio.paInt16:
            float32_chunk = audio_chunk.astype(np.float32) / INT16_MAX
        elif source_format == pyaudio.paInt24:
            float32_chunk = audio_chunk.astype(np.float32) / INT24_MAX
        elif source_format == pyaudio.paFloat32:
            float32_chunk = audio_chunk
        else:
            raise ValueError(f"Unsupported source format: {source_format}")
        
        # Clip the float32 values to [-1.0, 1.0]
        float32_chunk = np.clip(float32_chunk, -1.0, 1.0)
        
        # Convert from float32 to target format
        if target_format == pyaudio.paInt16:
            return (float32_chunk * INT16_MAX).astype(np.int16)
        elif target_format == pyaudio.paInt24:
            return (float32_chunk * INT24_MAX).astype(np.int32)
        elif target_format == pyaudio.paFloat32:
            return float32_chunk.astype(np.float32)
        else:
            raise ValueError(f"Unsupported target format: {target_format}")
    
    @classmethod
    def convert_format(cls, 
                      audio_chunk: np.ndarray, 
                      source_config: AudioConfig, 
                      target_config: AudioConfig) -> np.ndarray:
        """
        Convert audio between different configurations
        
        Args:
            audio_chunk: Input audio as numpy array
            source_config: Source audio configuration
            target_config: Target audio configuration
            
        Returns:
            Audio as numpy array in target format
        """
        # Convert channels if needed
        if source_config.channels != target_config.channels:
            if source_config.channels > target_config.channels:
                # Convert to mono by averaging channels
                audio_chunk = audio_chunk.reshape(-1, source_config.channels).mean(axis=1)
            else:
                # Convert to stereo by duplicating mono channel
                audio_chunk = np.tile(audio_chunk, (target_config.channels, 1)).T
            
        # Resample if different rate
        if source_config.rate != target_config.rate:
            audio_chunk = signal.resample(
                audio_chunk, 
                int(len(audio_chunk) * target_config.rate / source_config.rate)
            )
        
        # Convert bit format if needed
        if source_config.format != target_config.format:
            audio_chunk = cls.convert_bit_format(audio_chunk, source_config, target_config)
                
        return audio_chunk

    @classmethod
    def convert_bytes(cls,
                     audio_bytes: bytes,
                     source_config: AudioConfig,
                     target_config: AudioConfig) -> bytes:
        """
        Convert audio bytes between different configurations
        
        Args:
            audio_bytes: Input audio as bytes
            source_config: Source audio configuration
            target_config: Target audio configuration
            
        Returns:
            Audio as bytes in target format
        """
        # Convert bytes to numpy array based on source format
        dtype = np.int16 if source_config.format == pyaudio.paInt16 else np.float32
        audio_chunk = np.frombuffer(audio_bytes, dtype=dtype)
        
        # Convert format
        converted_chunk = cls.convert_format(audio_chunk, source_config, target_config)
        
        # Return as bytes
        return converted_chunk.tobytes()

class AudioInputRecorder:
    def __init__(
            self, 
            config: Optional[AudioConfig] = None,
            raw: bool = False
        ):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=config.format,
            channels=config.channels,
            rate=config.rate,
            input=True,
            frames_per_buffer=config.chunk,
            input_device_index=config.input_device
        )
        self.config = config or AudioConfig()
        self.raw = raw
        
    def read_chunk(self, raw: bool = False) -> dict[str, Any] | bytes:
        """
        Read and convert audio chunk
        
        Args:
            config: AudioConfig instance that contains audio format settings.
                    If none specified, default AudioConfig is used
            raw: If True, returns raw bit data as numpy array
                 If False, returns JSON-schema for Wyoming protocol (Default)
        """
        return self.stream.read(self.config.chunk, exception_on_overflow=False)
    
    def read_stream(self) -> Generator[bytes, None, None]:
        """
        Generator that yields audio chunks (raw bytes or Wyoming protocol depending on setup)
        """
        while True:
            try:
                yield self.read_chunk(self.raw)
            except Exception as e:
                print(f"Error reading audio stream: {e}")
                break
    
    def close(self):
        """Clean up audio resources"""
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
