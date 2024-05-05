import asyncio
import numpy as np
from typing import Optional
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
        self.background_audio = self.load_and_prepare_background(background_audio_path)
        self.background_volume = background_volume
        self.foreground_volume = 1.0 - background_volume
        self.queue: asyncio.Queue[bytes] = asyncio.Queue()
        self.process_task = asyncio.create_task(self.process_audio())

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
        return np.tile(background, 10)  # Loop the background audio sufficiently

    def mix_audio(self, foreground_chunk: bytes) -> bytes:
        foreground = np.frombuffer(foreground_chunk, dtype=np.int16)
        foreground_length = len(foreground)

        # Get the required length of background audio
        background = self.background_audio[:foreground_length]

        # Mix audio
        mixed_audio = (foreground * self.foreground_volume + background * self.background_volume)
        return np.clip(mixed_audio, -32768, 32767).astype(np.int16).tobytes()

    async def process_audio(self):
        while True:
            try:
                chunk = await asyncio.wait_for(self.queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                # Use silent foreground when there is no actual foreground chunk
                chunk = bytes(VONAGE_CHUNK_SIZE)

            mixed_chunk = self.mix_audio(chunk)
            # Play or send audio
            if self.output_to_speaker:
                self.output_speaker.consume_nonblocking(mixed_chunk)
            if self.ws:
                for i in range(0, len(mixed_chunk), VONAGE_CHUNK_SIZE):
                    subchunk = mixed_chunk[i:i + VONAGE_CHUNK_SIZE]
                    await self.ws.send_bytes(subchunk)

            # Roll the background buffer to simulate continuous play
            self.background_audio = np.roll(self.background_audio, -VONAGE_CHUNK_SIZE)
    def consume_nonblocking(self, chunk: bytes):
        self.queue.put_nowait(chunk)

    def terminate(self):
        self.process_task.cancel()
        if hasattr(self, 'background_audio'):
            self.background_audio.close()

