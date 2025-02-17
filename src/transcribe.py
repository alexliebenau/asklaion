import json
import base64
import asyncio
import logging
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from src.models.core import BaseModel
from src.audio import AudioInputRecorder, AudioConverter, AudioConfig, map_to_numpy

#############################
# Packet encoding
#############################

PACKET_TYPE_NAME = 'audio_chunk'

def get_json_packet(data: bytes, 
                    config: AudioConfig
    ) -> str:
    
    return json.dumps({
            "type": PACKET_TYPE_NAME,
            "data": {
                "audio": {
                    "bytes": base64.b64encode(data).decode("ascii"),  # Raw audio bytes encoded for JSON
                    "rate": config.rate,
                    "sample_size": config.sample_size,  # 16-bit = 2 bytes
                    "channels": config.channels
                }
            }
        })

def get_audio_frame(packet: dict[str, Any], 
                    target_config: AudioConfig
    ) -> np.ndarray:
    
    # if packet.get('type') is not PACKET_TYPE_NAME:
    #     raise RuntimeError(f"Invalid type of recieved packet: {packet.get('type')}")
    data: dict[str, Any] = packet.get('data')
    if not data:
        raise RuntimeError(f'No data in package!')

    audio: dict[str, Any] = data.get('audio')
    if not audio:
        raise RuntimeError(f'No audio in package!')

    # FIXME: Decoding bug?? 
    audio_bytes = base64.b64decode(audio.pop('bytes'))
    audio_array = np.fromstring(audio_bytes, dtype=map_to_numpy(audio['sample_size']))
    source_config = AudioConfig(**audio)
    return AudioConverter.convert_format(audio_array, source_config, target_config)


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

        Recieves packet containing data
        """
        print('Processing')
        audio_bytes = get_audio_frame(json.loads(packet), self.audio_config)
        return self.model.query(np.array(audio_bytes))


class RealTimeTranscriber:
    def __init__(self, 
                 transcriber: Transcriber, 
                 audio_input: AudioInputRecorder,
    ):
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
                transcription = await asyncio.to_thread(
                    self.transcriber.transcribe, 
                    get_json_packet(audio_data, self.audio_setup.config)
                )
                self.full_transcription += transcription['text'] + ' '
                # Print transcription on new line with flush
                print(f"\n\x1b[2K{transcription['text']}", flush=True)  
            except Exception as e:
                print(f"\n\x1b[2KTranscription error: {str(e)}", flush=True)
                logging.error("Error", exc_info=True)
            finally:
                self._processing_queue.task_done()

    async def _audio_capture_loop(self):
        print("Real-time transcription ready - speak now...")
        dtype = self.audio_setup.config.numpy_format
        while self._running:
            try:
                audio_frame = await asyncio.get_event_loop().run_in_executor(
                    self._executor,
                    self.audio_setup.read_chunk
                )
                audio_array = np.frombuffer(audio_frame, dtype=dtype)
                
                # Calculate current audio level metrics
                current_level = np.mean(np.abs(audio_array))
                format_level = lambda level: f'{level:.2f} / {self.audio_setup.mic_level(current_level):.2f}%'
                
                # Print status with flushing and carriage return
                print(
                    f"\rSilent chunks: {self._silent_chunks} of {self.audio_setup.config.silence_consecutive}| "
                    f"Current level: {format_level(current_level)} (Thresh: {format_level(self.audio_setup.config.silence_threshold)})",
                    end="", 
                    flush=True
                )
                
                if await self._vad_detect(audio_array):
                    self._silent_chunks += 1
                    if (self._silent_chunks > self.audio_setup.config.silence_consecutive 
                        and len(self._audio_buffer) > 0):
                        audio_data = np.concatenate(self._audio_buffer)
                        await self._processing_queue.put(audio_data)
                        self._audio_buffer.clear()
                        self._silent_chunks = 0
                        # Clear line before queue message
                        logging.info("\n\x1b[2KSent audio package off to be processed!", flush=True)  

                        # FIXME: for debugging. Don't forget to remove!
                        logging.info('Terminating loop after first sent package for debugging purposes')
                        break
                else:
                    # only append to buffer if mic threshold is exceeded
                    self._silent_chunks = 0
                    self._audio_buffer.append(audio_array)
                    
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
            
           