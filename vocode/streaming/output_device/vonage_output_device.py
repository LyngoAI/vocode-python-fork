import asyncio
import numpy as np
from typing import Optional
import wave
from pydub import AudioSegment
import io


from fastapi import WebSocket
from vocode.streaming.models.audio_encoding import AudioEncoding
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
        self.active = True
        self.queue: asyncio.Queue[bytes] = asyncio.Queue()
        self.process_task = asyncio.create_task(self.process())
        self.output_to_speaker = output_to_speaker
        self.background_audio = self.load_and_convert_background(background_audio_path)
        self.background_volume = background_volume
        self.foreground_volume = 1.0 - background_volume
        if output_to_speaker:
            self.output_speaker = SpeakerOutput.from_default_device(
                sampling_rate=VONAGE_SAMPLING_RATE, blocksize=VONAGE_CHUNK_SIZE // 2
            )
    
    # WORKING!!!!
    # def generate_sine_wave(self, frequency, length, rate):
    #     """Generate a sine wave of a specific frequency and length at a given sample rate."""
    #     t = np.linspace(0, length, int(rate * length), endpoint=False)
    #     wave = np.sin(frequency * 2 * np.pi * t)
    #     return (wave * np.iinfo(np.int16).max).astype(np.int16).tobytes()

    # def mix_audio(self, foreground_chunk: bytes) -> bytes:
    #     foreground = np.frombuffer(foreground_chunk, dtype=np.int16)
    #     background_chunk = self.generate_sine_wave(440, 1, VONAGE_SAMPLING_RATE)
    #     background = np.frombuffer(background_chunk[:len(foreground_chunk)], dtype=np.int16)
    #     print(f"Foreground length: {len(foreground)}, Background length: {len(background)}")  # Debugging output
    #     mixed_audio = (foreground * self.foreground_volume + background * self.background_volume)
    #     return np.clip(mixed_audio, -32768, 32767).astype(np.int16).tobytes()
    
    def load_and_convert_background(self, path):
        audio = AudioSegment.from_file(path)
        audio = audio.set_frame_rate(VONAGE_SAMPLING_RATE).set_channels(1)
        buffer = io.BytesIO()
        audio.export(buffer, format="wav")
        buffer.seek(0)
        return wave.open(buffer, 'rb')


    def stereo_to_mono(self, stereo_data):
    # Assumes stereo_data is an np.array of int16 type
        return stereo_data.reshape((-1, 2)).mean(axis=1).astype(np.int16)

    def mix_audio(self, foreground_chunk: bytes) -> bytes:
        foreground = np.frombuffer(foreground_chunk, dtype=np.int16)
        background_chunk = self.background_audio.readframes(len(foreground))
        background = np.frombuffer(background_chunk, dtype=np.int16)

        if len(background) < len(foreground):
            self.background_audio.rewind()
            extra_frames = len(foreground) - len(background)
            background_chunk += self.background_audio.readframes(extra_frames)
            background = np.frombuffer(background_chunk, dtype=np.int16)

        mixed_audio = (foreground * self.foreground_volume + background * self.background_volume)
        return np.clip(mixed_audio, -32768, 32767).astype(np.int16).tobytes()







    async def process(self):
        while self.active:
            chunk = await self.queue.get()
            mixed_chunk = self.mix_audio(chunk)
            if self.output_to_speaker:
                self.output_speaker.consume_nonblocking(mixed_chunk)
            for i in range(0, len(mixed_chunk), VONAGE_CHUNK_SIZE):
                subchunk = mixed_chunk[i : i + VONAGE_CHUNK_SIZE]
                if self.ws:
                    await self.ws.send_bytes(subchunk)


    def consume_nonblocking(self, chunk: bytes):
        self.queue.put_nowait(chunk)

    def maybe_send_mark_nonblocking(self, message_sent):
            pass
    def terminate(self):
        self.background_audio.close()
        self.process_task.cancel()

