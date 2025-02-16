import json
import asyncio
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from src.models.core import BaseModel
from src.audio import AudioInputRecorder, AudioConverter, AudioConfig

#############################
# Handle Wyoming Protocol
#############################

WYOMING_CHUNK_TYPE_NAME = 'wyoming_audio_chunk'

def get_wyoming_chunk(data: bytes, 
                      config: AudioConfig) -> str:
    
    return json.dumps({
            "type": WYOMING_CHUNK_TYPE_NAME,
            "data": {
                "audio": {
                    "bytes": str(data),  # Raw audio bytes
                    "rate": config.rate,
                    "width": 2,  # 16-bit = 2 bytes
                    "channels": config.channels
                }
            }
        })

def get_bytes(wyoming_chunk: dict[str, Any], 
              target_config: AudioConfig
              ) -> bytes:
    
    if wyoming_chunk.get('type') is not WYOMING_CHUNK_TYPE_NAME:
        raise RuntimeError(f"Invalid type of recieved packet: {wyoming_chunk.get('type')}")
    
    data: dict[str, Any] = wyoming_chunk.get('data')
    if not data:
        raise RuntimeError(f'No data in package!')

    audio: dict[str, Any] = wyoming_chunk.get('data')
    if not audio:
        raise RuntimeError(f'No audio in package!')
    
    audio_bytes = bytes(audio.pop('bytes'))
    source_config = AudioConfig(**audio)
    return AudioConverter.convert_bytes(audio_bytes, source_config, target_config)


class Transcriber:
    def __init__(
        self, 
        model: BaseModel,
        model_audio_config: AudioConfig,

    ):
        self.model = model
        self.audio_config = model_audio_config

    def setup(self):
        self.model.setup()

    def transcribe(self, packet: str) -> dict:
        """
        Use pre-loaded model for transcription

        Recieves packet in encoded for Wyoming Protocol
        """
        audio_bytes = get_bytes(json.loads(packet), self.audio_config)
        self.model.query(np.array(audio_bytes))


class RealTimeTranscriber:
    def __init__(self, 
                 transcriber: Transcriber, 
                 audio_input: AudioInputRecorder):
        self.transcriber = transcriber
        self.audio_setup = audio_input
        self._audio_buffer = []
        self.full_transcription = ""
        self._processing_queue = asyncio.Queue()
        self._executor = ThreadPoolExecutor()
        self._running = False
        self._silent_chunks = 0

    async def _vad_detect(self, audio_frame):
        """Non-blocking VAD using thread executor"""
        return await asyncio.get_event_loop().run_in_executor(
            self._executor,
            lambda: np.max(np.abs(audio_frame)) < self.audio_setup.config.silence_threshold
        )

    async def _process_queue(self):
        while self._running or not self._processing_queue.empty():
            audio_data = await self._processing_queue.get()
            try:
                # Added error handling for transcription
                transcription = await asyncio.to_thread(
                    self.transcriber.transcribe, 
                    get_wyoming_chunk(audio_data)
                )
                self.full_transcription += transcription['text'] + ' '
                print(f"\r{transcription['text']}", end="", flush=True)
            except Exception as e:
                print(f"\nTranscription error: {str(e)}")
            finally:
                self._processing_queue.task_done()

    async def _audio_capture_loop(self):
        print("Real-time transcription ready - speak now...")
        while self._running:
            try:
                print('Recording...')
                audio_frame = await asyncio.get_event_loop().run_in_executor(
                    self._executor,
                    self.audio_setup.read_chunk
                )
                audio_bytes = np.frombuffer(audio_frame)
                if await self._vad_detect(audio_bytes):
                    print('Silent chunk detected')
                    self._silent_chunks += 1
                    if (self._silent_chunks > self.audio_setup.config.silence_consecutive 
                        and len(self._audio_buffer) > 0):
                        audio_data = np.concatenate(self._audio_buffer)
                        await self._processing_queue.put(audio_data)
                        self._audio_buffer.clear()
                        self._silent_chunks = 0
                else:
                    self._silent_chunks = 0
                    self._audio_buffer.append(audio_bytes)
                    
            except asyncio.CancelledError:
                break

    async def process_audio_input(self):
        """Main async entry point"""
        try:
            self._running = True
            async with asyncio.TaskGroup() as tg:
                tg.create_task(self._process_queue())
                tg.create_task(self._audio_capture_loop())

        except KeyboardInterrupt:
            print("Stopping recording...")
            if len(self._audio_buffer) > 0:  # Process remaining audio
                await self._processing_queue.put(np.concatenate(self._audio_buffer))
            await self._processing_queue.join()
            self._executor.shutdown()
            self.audio_setup.close()