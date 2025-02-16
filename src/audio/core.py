import os
import pyaudio
import numpy as np
from dataclasses import dataclass

@dataclass
class AudioConfig:
    format: int = pyaudio.paInt16
    channels: int = 1
    rate: int = 16000
    chunk: int = 2048
    silence_threshold: int = 400
    silence_consecutive: int = 5
    input_device: int = None

class DefaultAudioSetup:
    def __init__(
            self, 
            config: AudioConfig
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
        self.config = config
        
    def read_chunk(self):
        """Read and convert audio chunk to numpy array"""
        data = self.stream.read(self.config.chunk, exception_on_overflow=False)
        return np.frombuffer(data, dtype=np.int16)
    
    def close(self):
        """Clean up audio resources"""
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()