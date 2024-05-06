import asyncio
from typing import Optional
import numpy as np
import wave
from pydub import AudioSegment
import io

from fastapi import WebSocket
from vocode.streaming.output_device.base_output_device import BaseOutputDevice
from vocode.streaming.output_device.speaker_output import SpeakerOutput
from vocode.streaming.telephony.constants import (
    VONAGE_AUDIO_ENCODING,
    VONAGE_CHUNK_SIZE,
    VONAGE_SAMPLING_RATE,
)

class VonageOutputDevice(BaseOutputDevice):
    def __init__(
        self,
        ws: Optional[WebSocket] = None,
        output_to_speaker: bool = False,
        background_audio_path: str = "/Users/cam/Repos/lyngoAI/phone_receptionist/vocode/vocode/streaming/synthesizer/filler_audio/typing-noise.wav",
        background_volume: float = 0.4,
    ):
        super().__init__(
            sampling_rate=VONAGE_SAMPLING_RATE, audio_encoding=VONAGE_AUDIO_ENCODING
        )
        self.ws = ws
        self.output_to_speaker = output_to_speaker
        self.background_audio_data = self.load_and_prepare_background(background_audio_path)
        self.background_volume = background_volume
        self.foreground_volume = 1.0 - background_volume
        self.queue: asyncio.Queue[Optional[bytes]] = asyncio.Queue()
        self.audio_play_task = asyncio.create_task(self.play_audio())

        if output_to_speaker:
            self.output_speaker = SpeakerOutput.from_default_device(
                sampling_rate=VONAGE_SAMPLING_RATE, blocksize=VONAGE_CHUNK_SIZE // 2
            )

    def load_and_prepare_background(self, path):
        audio = AudioSegment.from_file(path)
        audio = audio.set_frame_rate(VONAGE_SAMPLING_RATE).set_channels(1)
        buffer = io.BytesIO()
        audio.export(buffer, format="wav")
        buffer.seek(0)
        wav_file = wave.open(buffer, 'rb')
        frames = wav_file.readframes(wav_file.getnframes())
        background = np.frombuffer(frames, dtype=np.int16)
        return np.tile(background, 5)  # Ensure it's long enough for continuous looping

    async def play_audio(self):
        index = 0
        while True:
            try:
                foreground_chunk = await asyncio.wait_for(self.queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                foreground_chunk = None

            if foreground_chunk:
                foreground = np.frombuffer(foreground_chunk, dtype=np.int16)
            else:
                foreground = np.zeros(VONAGE_CHUNK_SIZE, dtype=np.int16)

            # Proper background audio management
            if index + len(foreground) >= len(self.background_audio_data):
                index = 0  # Reset to start to prevent index overflow

            background = self.background_audio_data[index:index + len(foreground)]
            mixed_audio = (foreground * self.foreground_volume + background * self.background_volume)
            mixed_audio_bytes = np.clip(mixed_audio, -32768, 32767).astype(np.int16).tobytes()

            if self.output_to_speaker:
                self.output_speaker.consume_nonblocking(mixed_audio_bytes)
            if self.ws:
                await self.send_chunked_bytes(mixed_audio_bytes)

            index += len(foreground)
            await asyncio.sleep(0.01)  # Sleep to manage timing

    async def send_chunked_bytes(self, bytes_data):
        for i in range(0, len(bytes_data), VONAGE_CHUNK_SIZE):
            subchunk = bytes_data[i:i + VONAGE_CHUNK_SIZE]
            await self.ws.send_bytes(subchunk)

    def consume_nonblocking(self, chunk: bytes):
        self.queue.put_nowait(chunk)

    def terminate(self):
        self.audio_play_task.cancel()
        if hasattr(self, 'background_audio'):
            self.background_audio.close()

