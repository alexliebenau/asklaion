import math 
import pyaudio
import logging
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

def map_sample_size(bits: int) -> int:
    format_map = {
            16: pyaudio.paInt16,
            24: pyaudio.paInt24,
            32: pyaudio.paFloat32
        }
    if bits not in format_map:
        raise ValueError(f'Invalid sample_size: {bits}. Choose 16, 24, or 32 bits')
    return format_map[bits]

def map_to_numpy(sample_size: int):
    return {
            pyaudio.paInt16: np.int16,
            pyaudio.paInt24: np.int32,  # 24-bit not native in numpy
            pyaudio.paFloat32: np.float32
        }[map_sample_size(sample_size)]


@dataclass
class AudioConfig:
    sample_size: int = 16  # bits (16, 24, or 32)
    channels: int = 1
    rate: int = 16000       # Hz
    chunk: int = 2048       # bytes per chunk
    silence_threshold: int = 80000
    silence_duration: int = 1  # seconds
    input_device: int = None

    @property
    def format(self) -> int:
        return map_sample_size(self.sample_size)
        
    @property
    def numpy_format(self):
        return map_to_numpy(self.sample_size)
    
    @property
    def samples_per_chunk(self) -> int:
        bytes_per_sample = (self.sample_size // 8) * self.channels
        return self.chunk // bytes_per_sample
    
    @property
    def chunk_duration(self) -> float:
        return self.samples_per_chunk / self.rate
    
    @property
    def silence_consecutive(self) -> int:
        return math.ceil(self.silence_duration / self.chunk_duration)
    

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
            config: Optional[AudioConfig] = None
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
    def read_chunk(self) -> dict[str, Any] | bytes:
        """
        Read and convert audio chunk
        
        Args:
            config: AudioConfig instance that contains audio format settings.
                    If none specified, default AudioConfig is used
        """
        return self.stream.read(self.config.chunk, exception_on_overflow=False)
    
    def read_stream(self) -> Generator[bytes, None, None]:
        """
        Generator that yields audio chunks (raw bytes)
        """
        while True:
            try:
                yield self.read_chunk()
            except Exception as e:
                logging.error(f"Error reading audio stream: {e}")
                break

    def set_silence_threshold(self):
        """Calibrate silence threshold using peak amplitude detection"""
        logging.warning(f'Setting mic threshold. Please stay silent for {self.config.silence_duration}s')
        max_amplitude = 0
        chunks_needed = math.ceil(self.config.rate / self.config.chunk * self.config.silence_duration)
        dtype = self.config.numpy_format
        
        for _ in range(chunks_needed):
            audio_frame = self.stream.read(self.config.chunk, exception_on_overflow=False)
            audio_array = np.frombuffer(audio_frame, dtype=dtype)
            
            # Track maximum amplitude with 50% safety margin
            current_max = np.max(np.abs(audio_array)) * 2
            if current_max > max_amplitude:
                max_amplitude = current_max
                
        self.config.silence_threshold = int(max_amplitude)
        logging.info(f'Silence threshold set to: {self.config.silence_threshold} ({self.input_level(self.config.silence_threshold):.2f}%, Range: 0-{self._max_possible_amplitude})')

    @property
    def _max_possible_amplitude(self):
        """Get maximum possible value for current format"""
        if self.config.numpy_format == np.int16:
            return 32767
        elif self.config.numpy_format == np.int32:  # For 24-bit packed in 32-bit
            return 8388607
        elif self.config.numpy_format == np.float32:
            return 1.0
        return 0
    
    def input_level(self, level: int) -> float:
        return 100 / self._max_possible_amplitude * level
    
    def close(self):
        """Clean up audio resources"""
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
