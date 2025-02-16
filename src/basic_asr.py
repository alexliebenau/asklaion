import numpy as np
from pathlib import Path

import mlx
import mlx_whisper.load_models as load_models

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any



# FIXME: check these, not sure if they exist
QUANTIZED_MODELS = {
    "fast": "mlx-community/whisper-tiny-8bit-mlx",
    "balanced": "mlx-community/whisper-base-4bit-mlx",
    "accurate": "mlx-community/whisper-large-v3-8bit-mlx"
}

DEFAULT_MODELS = {
    "fast": "mlx-community/whisper-tiny-mlx",
    "balanced": "mlx-community/whisper-base.en-mlx",
    "accurate": "mlx-community/whisper-large-v3-mlx" 
}

MODEL_PRESETS = {
    "fast": {
        "beam_size": 1,
        "word_timestamps": False,
        "temperature": 0.0
    },
    "balanced": {
        "beam_size": 3,
        "word_timestamps": True,
        "temperature": 0.0
    },
    "accurate": {
        "beam_size": 5,
        "word_timestamps": True,
        "temperature": 0.2
    }
}

MODEL_CACHE = Path.cwd() / "models" / "mlx_whisper_models"



class RealTimeTranscriber:
    def __init__(self, transcriber: Transcriber, audio_setup: AudioSetup):
        self.transcriber = transcriber
        self.audio_setup = audio_setup
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
                    audio_data
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
                
                if await self._vad_detect(audio_frame):
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
                    self._audio_buffer.append(audio_frame)
                    
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

if __name__ == "__main__":
    # Preload components
    config = AudioConfig()
    transcriber = create_transcriber(preset='fast', quantized=False)
    audio_setup = AudioSetup(config)
    
    # Start transcription system
    rt_transcriber = RealTimeTranscriber(transcriber, audio_setup)
    asyncio.run(rt_transcriber.process_audio_input())
    print(rt_transcriber.full_transcription)
