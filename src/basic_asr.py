import asyncio

from src.audio import AudioConfig, AudioInputRecorder
from src.transcribe import Transcriber, RealTimeTranscriber
from src.models.mlx_whisper import MLXWhisperModel
from src.models.core import ModelPerformance

if __name__ == "__main__":
    print('Setting up audio input')
    input_audio_config = AudioConfig()
    audio_recorder = AudioInputRecorder(input_audio_config)
    audio_recorder.set_silence_threshold() # move to init?

    print('Setting up model audio input')
    model_input_config = AudioConfig()
    model = MLXWhisperModel(ModelPerformance.FAST)

    print('Start transcription system')
    rt_transcriber = RealTimeTranscriber(
        transcriber=Transcriber(
            model,
            model_input_config,
        ), 
        audio_input=audio_recorder
        )
    
    asyncio.run(rt_transcriber.process_audio_input())
    print(rt_transcriber.full_transcription)
