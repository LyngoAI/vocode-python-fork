import asyncio
from typing import Optional
from fastapi import WebSocket
import numpy as np
import wave
import io
from pydub import AudioSegment

from vocode.streaming.output_device.base_output_device import BaseOutputDevice
from vocode.streaming.telephony.constants import VONAGE_AUDIO_ENCODING, VONAGE_CHUNK_SIZE, VONAGE_SAMPLING_RATE

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
        self.output_to_speaker = True
        self.background_audio_data = self.load_and_prepare_background(background_audio_path)
        self.background_volume = background_volume
        self.foreground_volume = 1.0 - background_volume
        self.queue = asyncio.Queue()
        self.foreground_chunk = None
        self.new_foreground_available = False
        self.audio_play_task = asyncio.create_task(self.play_audio())
        self.monitor_task = asyncio.create_task(self.monitor_foreground_audio())

    def load_and_prepare_background(self, path):
        audio = AudioSegment.from_file(path)
        audio = audio.set_frame_rate(44100).set_channels(1)
        buffer = io.BytesIO()
        audio.export(buffer, format="wav")
        buffer.seek(0)
        wav_file = wave.open(buffer, 'rb')
        frames = wav_file.readframes(wav_file.getnframes())
        background = np.frombuffer(frames, dtype=np.int16)
        return np.tile(background, int(44100 * 60 / len(background)))  # Tile to cover 60 seconds

    async def play_audio(self):
        background_index = 0
        background_length = len(self.background_audio_data)
        while True:
            # Check if new foreground audio is available
            if self.new_foreground_available:
                foreground = np.frombuffer(self.foreground_chunk, dtype=np.int16)
                self.new_foreground_available = False  # Reset the flag
            else:
                # If no new foreground audio, use a zero array of the same length
                foreground = np.zeros(VONAGE_CHUNK_SIZE, dtype=np.int16)

            end_index = background_index + len(foreground)
            if end_index >= background_length:
                # Handle wrapping of the background buffer
                end_index %= background_length
                background = np.concatenate((self.background_audio_data[background_index:], self.background_audio_data[:end_index]))
            else:
                background = self.background_audio_data[background_index:end_index]

            # Mix the foreground and background audio
            mixed_audio = np.add(background * self.background_volume, foreground * self.foreground_volume).astype(np.int16)
            mixed_audio_bytes = np.clip(mixed_audio, -32768, 32767).astype(np.int16).tobytes()

            # Send mixed audio to output
            if self.output_to_speaker:
                self.output_speaker.consume_nonblocking(mixed_audio_bytes)
            if self.ws:
                await self.send_chunked_bytes(mixed_audio_bytes)

            background_index = end_index
            await asyncio.sleep(0.01) 

    async def monitor_foreground_audio(self):
        while True:
            foreground_chunk = await self.queue.get()
            if foreground_chunk is None:
                break  # Exit the loop if shutdown signal received
            self.foreground_chunk = foreground_chunk
            self.new_foreground_available = True

    async def send_chunked_bytes(self, bytes_data):
        for i in range(0, len(bytes_data), VONAGE_CHUNK_SIZE):
            subchunk = bytes_data[i:i + VONAGE_CHUNK_SIZE]
            if self.ws:
                await self.ws.send_bytes(subchunk)

    def consume_nonblocking(self, chunk: bytes):
        self.queue.put_nowait(chunk)

    def terminate(self):
        self.queue.put_nowait(None)  # Signal to stop the monitor
        self.audio_play_task.cancel()
        self.monitor_task.cancel()
        if self.output_to_speaker and hasattr(self, 'output_speaker'):
            self.output_speaker.close()
