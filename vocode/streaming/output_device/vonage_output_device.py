import asyncio
from typing import Optional
import numpy as np
import io
import os
import wave
from pydub import AudioSegment
from fastapi import WebSocket
from vocode.streaming.output_device.speaker_output import SpeakerOutput
from vocode.streaming.output_device.base_output_device import BaseOutputDevice
from vocode.streaming.telephony.constants import VONAGE_AUDIO_ENCODING, VONAGE_CHUNK_SIZE, VONAGE_SAMPLING_RATE

BACKGROUND_NOISE_PATH = os.path.join(os.path.dirname(__file__), "background_noise")
BACKGROUND_NOISE_FILE_PATH = "%s/coffee-shop.wav" % BACKGROUND_NOISE_PATH
class VonageOutputDevice(BaseOutputDevice):
    def __init__(
        self,
        ws: Optional[WebSocket] = None,
        output_to_speaker: bool = False,
        background_audio_path: str = BACKGROUND_NOISE_FILE_PATH,
        background_volume: float = 0.4,
    ):
        super().__init__(
            sampling_rate=VONAGE_SAMPLING_RATE, audio_encoding=VONAGE_AUDIO_ENCODING
        )
        self.ws = ws
        self.active = True
        self.foreground_buffer = np.array([], dtype=np.int16)
        self.foreground_chunks = []
        self.queue = asyncio.Queue()
        self.background_audio = self.load_and_convert_background(background_audio_path)
        self.background_volume = background_volume
        self.foreground_volume = 1.0 - background_volume
        self.output_to_speaker = output_to_speaker and SpeakerOutput.from_default_device(
            sampling_rate=VONAGE_SAMPLING_RATE, blocksize=VONAGE_CHUNK_SIZE // 2
        )
        self.background_task = asyncio.create_task(self.play_background())
        self.foreground_task = asyncio.create_task(self.handle_foreground())
        self.is_last_chunk = False

    # Load in audio and convert it to single channel to match output from synthesizer
    def load_and_convert_background(self, path):
        audio = AudioSegment.from_file(path).set_frame_rate(VONAGE_SAMPLING_RATE).set_channels(1)
        buffer = io.BytesIO()
        audio.export(buffer, format="wav")
        buffer.seek(0)
        return wave.open(buffer, 'rb')

    # Mix synthesizer and background audio
    def mix_audio(self, foreground_chunk: bytes) -> bytes:
        foreground = np.frombuffer(foreground_chunk, dtype=np.int16)
        background_chunk = self.background_audio.readframes(len(foreground))
        background = np.frombuffer(background_chunk, dtype=np.int16)

        # Mixed audio needs to be matched length
        if len(background) < len(foreground):
            self.background_audio.rewind()
            extra_frames = len(foreground) - len(background)
            background_chunk += self.background_audio.readframes(extra_frames)
            background = np.frombuffer(background_chunk, dtype=np.int16)
        mixed_audio = (foreground * self.foreground_volume + background * self.background_volume)
        return np.clip(mixed_audio, -32768, 32767).astype(np.int16).tobytes()

    async def play_background(self):
        while True:
            if self.foreground_chunks:
                foreground_chunk = self.foreground_chunks.pop(0)  # Get the oldest chunk
                mixed_chunk = self.mix_audio(foreground_chunk)
                sleep_time = 0.5
            # If its last chunk then just send background audio
            elif self.is_last_chunk:
                # Calculate the remaining frames
                remaining_frames = self.background_audio.getnframes() - self.background_audio.tell()
                # Determine how many frames to read, limited by the remaining number
                num_frames_to_read = min(8000, remaining_frames)
                
                # check at least 500 frames are left
                if num_frames_to_read <= 500:
                    self.background_audio.rewind()
                    num_frames_to_read = 8000

                # Read the available frames
                background_chunk = self.background_audio.readframes(num_frames_to_read)
                background = np.frombuffer(background_chunk, dtype=np.int16)

                # Mix audio (ensure data size)
                mixed_chunk = np.clip(background, -32768, 32767).astype(np.int16).tobytes()
                sleep_time = 0.5
            else:
                await asyncio.sleep(0.1)
                continue

            if self.output_to_speaker:
                self.output_speaker.consume_nonblocking(mixed_chunk)
            if self.ws:
                for i in range(0, len(mixed_chunk), VONAGE_CHUNK_SIZE):
                    subchunk = mixed_chunk[i : i + VONAGE_CHUNK_SIZE]
                    await self.ws.send_bytes(subchunk)

            # Bit of a hack to dynamically change sleep time
            # If we kept sleep time the same then audio becomes choppy, so increase it when mixing audio and decrease it when just background 
            await asyncio.sleep(sleep_time)

    # Loop to process any audio from the synthesizer and add to a list
    async def handle_foreground(self):
        while self.active:
            foreground_chunk, self.is_last_chunk = await self.queue.get()  # Receive chunks from the queue
            self.foreground_chunks.append(foreground_chunk)
    
    def consume_nonblocking(self, chunk: bytes, is_last_chunk: bool):
        self.queue.put_nowait((chunk, is_last_chunk))  # Add chunks to the queue for processing

    def terminate(self):
        self.active = False
        self.background_audio.close()
        self.background_task.cancel()
        self.foreground_task.cancel()
