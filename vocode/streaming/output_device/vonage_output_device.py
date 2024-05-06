import asyncio
from typing import Optional
import numpy as np
import io
import wave
from pydub import AudioSegment
from fastapi import WebSocket
from vocode.streaming.output_device.speaker_output import SpeakerOutput
from vocode.streaming.models.audio_encoding import AudioEncoding
from vocode.streaming.output_device.base_output_device import BaseOutputDevice
from vocode.streaming.telephony.constants import VONAGE_AUDIO_ENCODING, VONAGE_CHUNK_SIZE, VONAGE_SAMPLING_RATE
import time

class VonageOutputDevice(BaseOutputDevice):
    def __init__(
        self,
        ws: Optional[WebSocket] = None,
        output_to_speaker: bool = False,
        background_audio_path: str = "/Users/cam/Repos/lyngoAI/phone_receptionist/vocode/vocode/streaming/synthesizer/filler_audio/coffee-shop.wav",
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
        self.last_message_time = time.time()  
        self.is_last_chunk = False

    def load_and_convert_background(self, path):
        audio = AudioSegment.from_file(path).set_frame_rate(VONAGE_SAMPLING_RATE).set_channels(1)
        buffer = io.BytesIO()
        audio.export(buffer, format="wav")
        buffer.seek(0)
        return wave.open(buffer, 'rb')

    # async def play_background(self):
    #     while True:
    #         background_chunk = np.frombuffer(self.background_audio.readframes(VONAGE_CHUNK_SIZE), dtype=np.int16)
    #         if background_chunk.size == 0:
    #             self.background_audio.rewind()
    #             background_chunk = np.frombuffer(self.background_audio.readframes(VONAGE_CHUNK_SIZE), dtype=np.int16)

    #         # if self.foreground_buffer.size > 0:
    #         #     print("new foreground")
    #         #     mix_length = min(len(background_chunk), len(self.foreground_buffer))
    #         #     foreground_chunk = self.foreground_buffer[:mix_length]
    #         #     self.foreground_buffer = self.foreground_buffer[mix_length:]
    #         #     print(f"Length of background chunk: {len(background_chunk)}")
    #         #     print(f"Length of foreground chunk: {len(foreground_chunk)}")

    #         if self.foreground_chunks:

    #             foreground_chunk = self.foreground_chunks.pop(0)
    #             print("new foreground")
    #             print(f"Length of background chunk: {len(background_chunk)}")
    #             print(f"Length of foreground chunk: {len(foreground_chunk)}")
    #             if len(background_chunk) < len(foreground_chunk):
    #                 self.background_audio.rewind()
    #                 extra_frames = len(foreground_chunk) - len(background)
    #                 background_chunk += self.background_audio.readframes(extra_frames)
    #                 background = np.frombuffer(background_chunk, dtype=np.int16)

    #             mixed_audio = (foreground_chunk * self.foreground_volume + background_chunk * self.background_volume)
    #             mixed_audio = np.clip(mixed_audio, -32768, 32767).astype(np.int16)
    #             mixed_chunk = mixed_audio.tobytes()

    #             # if mix_length < len(background_chunk):
    #             #     mixed_chunk += background_chunk[mix_length:].tobytes()
    #         else:
    #             mixed_chunk = background_chunk.tobytes()

    #         if self.output_to_speaker:
    #             self.output_speaker.consume_nonblocking(mixed_chunk)
    #         for i in range(0, len(mixed_chunk), VONAGE_CHUNK_SIZE):
    #             subchunk = mixed_chunk[i : i + VONAGE_CHUNK_SIZE]
    #             if self.ws:
    #                 await self.ws.send_bytes(subchunk)
            
    #         await asyncio.sleep(0.01)  # Yield back to the event loop
    def mix_audio(self, foreground_chunk: bytes) -> bytes:
        foreground = np.frombuffer(foreground_chunk, dtype=np.int16)
        background_chunk = self.background_audio.readframes(len(foreground))
        background = np.frombuffer(background_chunk, dtype=np.int16)

        if len(background) < len(foreground):
            self.background_audio.rewind()
            extra_frames = len(foreground) - len(background)
            background_chunk += self.background_audio.readframes(extra_frames)
            background = np.frombuffer(background_chunk, dtype=np.int16)
        print(f"Length of background chunk: {len(background)}")
        print(f"Length of foreground chunk: {len(foreground)}")
        mixed_audio = (foreground * self.foreground_volume + background * self.background_volume)
        return np.clip(mixed_audio, -32768, 32767).astype(np.int16).tobytes()

    async def play_background(self):
        new_message = False
        while True:
            if self.foreground_chunks:
                foreground_chunk = self.foreground_chunks.pop(0)  # Get the oldest chunk
                # foreground_chunk = await self.queue.get()
                mixed_chunk = self.mix_audio(foreground_chunk)
                sleep_time = 0.5
                new_message = True
            elif self.is_last_chunk:
                # Just read background for the buffer size if no foreground is available
                background_chunk = self.background_audio.readframes(8000)
                background = np.frombuffer(background_chunk, dtype=np.int16)
                mixed_chunk = np.clip(background, -32768, 32767).astype(np.int16).tobytes() #background.tobytes()
                sleep_time = 0.5
                new_message = False
            else:
                await asyncio.sleep(0.1)
                continue

            if self.output_to_speaker:
                self.output_speaker.consume_nonblocking(mixed_chunk)
            if self.ws:
                for i in range(0, len(mixed_chunk), VONAGE_CHUNK_SIZE):
                    subchunk = mixed_chunk[i : i + VONAGE_CHUNK_SIZE]
                    await self.ws.send_bytes(subchunk)

            # current_time = time.time()
            # time_elapsed = (current_time - self.last_message_time) * 1000
            # # print(time_elapsed)
            # if time_elapsed > 5000 and not new_message:
            #     await asyncio.sleep(0.01)  # Yield back to the event loop
            # else:
            #     await asyncio.sleep(0.5)

            await asyncio.sleep(sleep_time)
            new_message = False

    async def handle_foreground(self):
        while self.active:
            foreground_chunk, self.is_last_chunk = await self.queue.get()  # Receive chunks from the queue
            print("Is last chunk: ", self.is_last_chunk)
            current_time = time.time()
            self.last_message_time = current_time
            # foreground_array = np.frombuffer(foreground_chunk, dtype=np.int16)
            self.foreground_chunks.append(foreground_chunk)
            # self.foreground_buffer = np.concatenate((self.foreground_buffer, foreground_array))
    
    def consume_nonblocking(self, chunk: bytes, is_last_chunk: bool):
        self.queue.put_nowait((chunk, is_last_chunk))  # Add chunks to the queue for processing

    def terminate(self):
        self.active = False
        self.background_audio.close()
        self.background_task.cancel()
        self.foreground_task.cancel()
